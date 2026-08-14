import os
import signal
from datetime import datetime

from core import Recorder, transcribe

if __name__ == "__main__":
    now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    wav_path = f"reuniao_{now}.wav"
    md_path = f"reuniao_{now}.md"

    print("Gravando mic + sistema... Ctrl+C para parar e transcrever.")

    recorder = Recorder()
    recorder.start()

    def stop(sig, frame):
        print("\nParando gravação...")
        recorder.recording = False

    signal.signal(signal.SIGINT, stop)
    recorder.wait()

    print("Mixando áudio...")
    recorder.stop_and_save(wav_path)

    print("Transcrevendo (pode levar alguns minutos)...")
    transcribe(wav_path, md_path, progress=print)

    os.remove(wav_path)
    print(f"WAV deletado. Pronto: {md_path}")
