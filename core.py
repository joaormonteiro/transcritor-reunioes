import sys
import os

if sys.platform == "win32":
    # ctranslate2 (usado pelo whisperx) traz um cudnn64_9.dll "dispatcher" sem as
    # DLLs de backend (cudnn_ops64_9.dll etc.) — elas existem na pasta do torch.
    # Sem isso, o carregamento do modelo falha com "Could not load symbol cudnnGetLibConfig".
    import torch
    _torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.isdir(_torch_lib):
        os.add_dll_directory(_torch_lib)
        os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")

import pyaudiowpatch as pyaudio
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal
import threading
import time
from fractions import Fraction
from datetime import datetime
from dotenv import load_dotenv
import whisperx
from whisperx.diarize import DiarizationPipeline

if getattr(sys, "frozen", False):
    _app_dir = os.path.dirname(sys.executable)
else:
    _app_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_app_dir, ".env"))
HF_TOKEN = os.getenv("HF_TOKEN")

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = np.float32


class Recorder:
    """Grava mic + áudio do sistema (loopback) em paralelo até stop() ser chamado."""

    def __init__(self):
        self.mic_chunks = []
        self.loopback_chunks = []
        self.recording = False
        self._t_mic = None
        self._t_loop = None
        self.loopback_found = True
        self._loop_channels = 2
        self._loop_rate = 48000

    def _record_mic(self):
        def callback(indata, frames, time, status):
            if self.recording:
                self.mic_chunks.append(indata.copy())
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                             dtype=DTYPE, callback=callback):
            while self.recording:
                sd.sleep(100)

    def _record_loopback(self):
        p = pyaudio.PyAudio()
        try:
            device_info = p.get_default_wasapi_loopback()
        except (OSError, LookupError):
            self.loopback_found = False
            p.terminate()
            return

        # WASAPI loopback não faz resample/downmix sozinho: precisa abrir no
        # formato nativo do dispositivo (normalmente estéreo, 44100/48000Hz) —
        # abrir forçando mono/16kHz direto captura só ruído/silêncio.
        self._loop_channels = int(device_info["maxInputChannels"]) or 2
        self._loop_rate = int(device_info["defaultSampleRate"])

        def callback(in_data, frame_count, time_info, status):
            if self.recording:
                self.loopback_chunks.append(np.frombuffer(in_data, dtype=DTYPE))
            return (None, pyaudio.paContinue)

        # Modo callback (não-bloqueante): com leitura bloqueante (stream.read),
        # se o áudio do sistema fica em silêncio o read() trava esperando dados
        # e nunca percebe que recording=False — o app fica preso ao "Parar".
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=self._loop_channels,
            rate=self._loop_rate,
            input=True,
            input_device_index=device_info["index"],
            frames_per_buffer=1024,
            stream_callback=callback,
        )
        stream.start_stream()
        while self.recording:
            time.sleep(0.1)
        stream.stop_stream()
        stream.close()
        p.terminate()

    def start(self):
        self.mic_chunks = []
        self.loopback_chunks = []
        self.recording = True
        self._t_mic = threading.Thread(target=self._record_mic, daemon=True)
        self._t_loop = threading.Thread(target=self._record_loopback, daemon=True)
        self._t_mic.start()
        self._t_loop.start()

    def wait(self):
        """Bloqueia até recording=False e as threads terminarem (ex: após Ctrl+C)."""
        self._t_mic.join()
        self._t_loop.join()

    def stop_and_save(self, path):
        self.recording = False
        self.wait()

        mic = np.concatenate(self.mic_chunks).reshape(-1) if self.mic_chunks else np.array([], dtype=DTYPE)
        loop_raw = np.concatenate(self.loopback_chunks) if self.loopback_chunks else np.array([], dtype=DTYPE)

        loop = np.array([], dtype=DTYPE)
        if len(loop_raw):
            channels = max(self._loop_channels, 1)
            usable_len = (len(loop_raw) // channels) * channels
            loop_mono = loop_raw[:usable_len].reshape(-1, channels).mean(axis=1).astype(DTYPE)
            if self._loop_rate != SAMPLE_RATE:
                ratio = Fraction(SAMPLE_RATE, self._loop_rate).limit_denominator(1000)
                loop_mono = scipy.signal.resample_poly(
                    loop_mono, ratio.numerator, ratio.denominator
                ).astype(DTYPE)
            loop = loop_mono

        if len(mic) and len(loop):
            min_len = min(len(mic), len(loop))
            mixed = (mic[:min_len] + loop[:min_len]) / 2
        else:
            mixed = mic if len(mic) else loop

        peak = np.max(np.abs(mixed)) if len(mixed) else 0
        if peak > 0:
            mixed = mixed / peak * 0.9

        wav.write(path, SAMPLE_RATE, mixed)
        return path


def transcribe(wav_path, output_path, device="cuda", compute_type="float16", progress=None):
    def report(msg):
        if progress:
            progress(msg)

    report("Carregando modelo Whisper...")
    model = whisperx.load_model("medium", device, compute_type=compute_type)
    audio = whisperx.load_audio(wav_path)
    result = model.transcribe(audio, language="pt")

    report("Alinhando timestamps...")
    model_a, metadata = whisperx.load_align_model(language_code="pt", device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device)

    report("Identificando speakers...")
    diarize_model = DiarizationPipeline(token=HF_TOKEN, device=device)
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    lines = []
    current_speaker = None
    current_text = ""

    for seg in result["segments"]:
        speaker = seg.get("speaker", "Speaker ?")
        text = seg["text"].strip()
        if speaker != current_speaker:
            if current_text:
                lines.append(f"[{current_speaker}] {current_text}")
            current_speaker = speaker
            current_text = text
        else:
            current_text += " " + text

    if current_text:
        lines.append(f"[{current_speaker}] {current_text}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(datetime.now().strftime("# Reunião %Y-%m-%d %H:%M\n\n"))
        f.write("\n".join(lines))

    report(f"Transcrição salva: {output_path}")
    return output_path
