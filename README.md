# Meeting Transcriber

A Windows desktop app that records a meeting from **both** the microphone and the
system audio at the same time, then produces a speaker-labeled transcript in
Markdown, one file per meeting.

Built for online calls where only your own mic would otherwise be captured: it
taps the system output via **WASAPI loopback**, mixes it with the mic, and runs
the result through [WhisperX](https://github.com/m-bain/whisperX) for
transcription, forced alignment and speaker diarization.

## How it works

1. **Record.** Two threads capture in parallel: the mic (`sounddevice`, 16 kHz
   mono) and the system output (`pyaudiowpatch` WASAPI loopback, opened in the
   device's *native* format because loopback does no resample or downmix on its
   own).
2. **Mix.** The loopback stream is downmixed to mono and resampled to 16 kHz
   (`scipy.signal.resample_poly`), aligned with the mic by length, averaged, and
   peak-normalized to 0.9.
3. **Transcribe.** WhisperX `medium` model, language `pt`, then timestamp
   alignment.
4. **Diarize.** The `pyannote` pipeline (needs a Hugging Face token) assigns a
   speaker to each word.
5. **Write.** Consecutive segments from the same speaker are merged into
   `[SPEAKER_00] ...` blocks and saved to `reunioes/reuniao_<timestamp>.md`. The
   intermediate `.wav` is deleted.

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

copy .env.example .env   # then put your Hugging Face token in it
```

```ini
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

The HF token is needed for the diarization model. GPU is assumed
(`device="cuda"`, `compute_type="float16"`); edit `core.py` for CPU.

## Run

```bash
python app.py
```

A small Tkinter window: **Record** to start, **Stop and Transcribe** to finish.
Output lands in the `reunioes/` folder next to the app.

```bash
python app.py --selftest
```

Runs the transcription path on a synthetic tone and writes the result (or the
full traceback) to `selftest.log`, used to debug the packaged build with no UI.

## Windows notes

- **cuDNN DLLs.** `ctranslate2` (used by WhisperX) ships a `cudnn64_9.dll`
  dispatcher without its backend DLLs. `core.py` adds `torch/lib` to the DLL
  search path at import time so model loading doesn't fail with
  *"Could not load symbol cudnnGetLibConfig"*.
- **Loopback capture uses callback mode**, not blocking reads: with a blocking
  `stream.read`, a silent system output would block forever and the app would
  hang on "Stop".

## Stack

Python, Tkinter, pyaudiowpatch (WASAPI loopback), sounddevice, scipy, WhisperX,
pyannote.
