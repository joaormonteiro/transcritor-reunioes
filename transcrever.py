import pyaudiowpatch as pyaudio
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import threading
import signal
import os
from datetime import datetime
from dotenv import load_dotenv
import whisperx

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = np.float32

mic_chunks = []
loopback_chunks = []
recording = True


def record_mic():
    def callback(indata, frames, time, status):
        if recording:
            mic_chunks.append(indata.copy())
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype=DTYPE, callback=callback):
        while recording:
            sd.sleep(100)


def record_loopback():
    p = pyaudio.PyAudio()
    loopback_device = None
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get('isLoopbackDevice'):
            loopback_device = i
            break
    if loopback_device is None:
        print("Aviso: loopback não encontrado — apenas microfone será gravado.")
        p.terminate()
        return
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=loopback_device,
        frames_per_buffer=1024
    )
    while recording:
        data = stream.read(1024, exception_on_overflow=False)
        loopback_chunks.append(np.frombuffer(data, dtype=DTYPE))
    stream.stop_stream()
    stream.close()
    p.terminate()


def mix_and_save(path):
    mic = np.concatenate(mic_chunks) if mic_chunks else np.array([], dtype=DTYPE)
    loop = np.concatenate(loopback_chunks) if loopback_chunks else np.array([], dtype=DTYPE)

    if len(mic) and len(loop):
        min_len = min(len(mic), len(loop))
        mixed = (mic[:min_len] + loop[:min_len]) / 2
    else:
        mixed = mic if len(mic) else loop

    peak = np.max(np.abs(mixed))
    if peak > 0:
        mixed = mixed / peak * 0.9

    wav.write(path, SAMPLE_RATE, mixed)


def transcribe(wav_path, output_path):
    device = "cpu"  # trocar por "cuda" se tiver GPU NVIDIA

    print("Carregando modelo Whisper...")
    model = whisperx.load_model("medium", device, compute_type="int8")
    audio = whisperx.load_audio(wav_path)
    result = model.transcribe(audio, language="pt")

    print("Alinhando timestamps...")
    model_a, metadata = whisperx.load_align_model(language_code="pt", device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device)

    print("Identificando speakers...")
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
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

    print(f"\nTranscrição salva: {output_path}")


if __name__ == "__main__":
    now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    wav_path = f"reuniao_{now}.wav"
    md_path = f"reuniao_{now}.md"

    print("Gravando mic + sistema... Ctrl+C para parar e transcrever.")

    t_mic = threading.Thread(target=record_mic)
    t_loop = threading.Thread(target=record_loopback)

    def stop(sig, frame):
        global recording
        recording = False
        print("\nParando gravação...")

    signal.signal(signal.SIGINT, stop)

    t_mic.start()
    t_loop.start()
    t_mic.join()
    t_loop.join()

    print("Mixando áudio...")
    mix_and_save(wav_path)

    print("Transcrevendo (pode levar alguns minutos)...")
    transcribe(wav_path, md_path)

    os.remove(wav_path)
    print(f"WAV deletado. Pronto: {md_path}")
