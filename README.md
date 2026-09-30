<div align="center">

# 🎙️ ASR-Tool: Automatic Speech Recognition System

**A production-quality, offline speech-to-text transcription engine & web application.**

<br/>

` Python 3.10+ ` &nbsp;•&nbsp; ` faster-whisper ` &nbsp;•&nbsp; ` Streamlit UI ` &nbsp;•&nbsp; ` MIT Licensed ` &nbsp;•&nbsp; ` 17/17 Tests Passed `

<br/>

[Key Features](#-key-features) •
[Web Interface](#-web-interface-preview) •
[Architecture](#-system-architecture) •
[Quickstart](#-quickstart-guide) •
[CLI Usage](#-cli-usage) •
[Evaluation](#-accuracy--wer-evaluation) •
[Author](#-author--acknowledgments)

</div>

<br/>

## 📌 Overview

**ASR-Tool** is an offline, privacy-first Automatic Speech Recognition app built for high accuracy, speed, and zero paid API dependencies. Powered by OpenAI's Whisper model via the optimized CTranslate2 engine (`faster-whisper`), ASR-Tool transcribes multi-format audio files (.wav, .mp3, .m4a, .flac, .ogg) and live microphone recordings into plain text, timestamped subtitles (.srt), and structured metadata (.json).

Whether you need a sleek browser interface or an automated command-line tool for pipeline processing, **ASR-Tool** provides an all-in-one solution.

---

## 📸 Web Interface Preview

<div align="center">
  <img src="https://raw.githubusercontent.com/Brijeshv3001/ASR-Tool---Speech-Recognition/main/docs/screenshot_ui.png" alt="ASR-Tool Web Interface" width="100%" />
</div>

<br/>

> *Modern glassmorphism dashboard featuring real-time audio playback, customizable model sizes, language auto-detection, timestamp segment tables, and multi-format file exports.*

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| 🔒 **100% Offline & Free** | Zero external API calls, zero subscription fees, and complete data privacy. |
| ⚡ **Faster-Whisper Engine** | Accelerated CTranslate2 execution offering up to **4x faster** transcription than standard PyTorch models. |
| 🎛️ **Configurable Models** | Switch dynamically between `tiny`, `base`, `small`, and `medium` model sizes depending on speed and accuracy needs. |
| 🌐 **Multi-Language Auto-Detect** | Automatically recognizes spoken language across 99+ languages with confidence probability scores. |
| ⏱️ **Word/Segment Timestamps** | Generates precise millisecond-level subtitle timestamps (`HH:MM:SS,mmm`). |
| 🎤 **Browser Mic Recording** | Record audio directly in the web UI using native browser input (`st.audio_input`). |
| 📦 **Multi-Format Exporters** | One-click export to **Plain Text (`.txt`)**, **SubRip Subtitles (`.srt`)**, and **JSON (`.json`)**. |
| 💻 **Robust Argparse CLI** | Full command-line interface for headless execution and automated shell scripts. |
| 📊 **WER / CER Evaluator** | Evaluate transcription accuracy against ground-truth text using `jiwer`. |

---

## 🏗️ System Architecture

```
                                    +--------------------------------+
                                    | Audio File (.wav/.mp3/etc.)    |
                                    | OR Live Browser Microphone     |
                                    +---------------+----------------+
                                                    |
                                                    v
                                    +---------------+----------------+
                                    |   asr_tool.audio_utils         |
                                    | - Format Validation            |
                                    | - FFmpeg / Pydub Conversion    |
                                    | - 16kHz Mono Float32 Resampling|
                                    +---------------+----------------+
                                                    |
                                                    v
                                    +---------------+----------------+
                                    |   asr_tool.transcriber         |
                                    | - WhisperTranscriber Class     |
                                    | - faster-whisper Model Load    |
                                    | - CTranslate2 Parallel Infer   |
                                    +---------------+----------------+
                                                    |
                                                    v
                                    +---------------+----------------+
                                    |      TranscriptionResult       |
                                    | - Full Text & Segment Objects  |
                                    | - Duration & Performance Time  |
                                    | - Language & Confidence Score  |
                                    +-------+----------------+-------+
                                            |                |
                     +----------------------+                +----------------------+
                     |                                                              |
                     v                                                              v
+--------------------+-----------+                                +-----------------+---------+
| asr_tool.cli                   |                                | app.py                    |
| - CLI Transcribe Command       |                                | - Streamlit Web Dashboard |
| - CLI Evaluate Command         |                                | - Interactive Downloads   |
+--------------------------------+                                +---------------------------+
```

---

## ⚙️ Prerequisites

Before installing ASR-Tool, ensure you have the following installed:

1. **Python 3.10 or higher** (`python --version`)
2. **FFmpeg** (Required for audio decoding and format conversions):
   - 🪟 **Windows**: Included automatically via Python `imageio-ffmpeg` package, or install via Winget:
     ```powershell
     winget install ffmpeg
     ```
   - 🍎 **macOS**:
     ```bash
     brew install ffmpeg
     ```
   - 🐧 **Linux (Ubuntu/Debian)**:
     ```bash
     sudo apt update && sudo apt install -y ffmpeg
     ```

---

## 🚀 Quickstart Guide

### 1. Clone & Set Up Environment

```bash
# Clone the repository
git clone https://github.com/Brijeshv3001/ASR-Tool---Speech-Recognition.git
cd ASR-Tool---Speech-Recognition

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Web Interface

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to use the interactive dashboard!

---

## 💻 CLI Usage

ASR-Tool features a powerful CLI for automated speech processing tasks.

### 1. Transcribe Audio File to Plain Text
```bash
python -m asr_tool.cli transcribe audio.wav --model base --language en --format txt --output result.txt
```

### 2. Generate Subtitles (.srt) with Auto Language Detection
```bash
python -m asr_tool.cli transcribe lecture.mp3 --model small --language auto --format srt --output subtitle.srt
```

### 3. Export All Formats (.txt, .srt, .json) Simultaneously
```bash
python -m asr_tool.cli transcribe interview.m4a --model medium --format all --output exports/
```

---

## 📈 Accuracy & WER Evaluation

Evaluate transcript precision against ground-truth reference transcripts using the built-in evaluation command:

```bash
python -m asr_tool.cli evaluate --hypothesis samples/output_sample.txt --reference samples/sample_ref.txt --output docs/evaluation_metrics.json
```

### 🧪 Automated Test Suite Results
All unit and integration tests are verified passing:

```text
tests/test_audio_utils.py::test_find_ffmpeg_executable PASSED            [  5%]
tests/test_audio_utils.py::test_validate_audio_file_nonexistent PASSED   [ 11%]
tests/test_audio_utils.py::test_validate_audio_file_empty PASSED         [ 17%]
tests/test_audio_utils.py::test_validate_audio_file_unsupported PASSED   [ 23%]
tests/test_audio_utils.py::test_validate_audio_file_success PASSED       [ 29%]
tests/test_audio_utils.py::test_get_audio_duration PASSED                [ 35%]
tests/test_audio_utils.py::test_convert_to_wav_16k_mono PASSED           [ 41%]
tests/test_evaluate.py::test_evaluate_perfect_match PASSED               [ 47%]
tests/test_evaluate.py::test_evaluate_partial_match PASSED               [ 52%]
tests/test_evaluate.py::test_evaluate_empty_reference PASSED             [ 58%]
tests/test_evaluate.py::test_evaluate_empty_hypothesis PASSED            [ 64%]
tests/test_exporters.py::test_format_timestamp PASSED                    [ 70%]
tests/test_exporters.py::test_export_txt PASSED                          [ 76%]
tests/test_exporters.py::test_export_srt PASSED                          [ 82%]
tests/test_exporters.py::test_export_json PASSED                         [ 88%]
tests/test_transcriber.py::test_transcriber_invalid_model PASSED         [ 94%]
tests/test_transcriber.py::test_transcriber_integration_tiny PASSED      [100%]

======================== 17 passed in 2.40s ========================
```

---

## 📂 Project Structure

```
ASR-Tool---Speech-Recognition/
├── asr_tool/               # Core Python Package
│   ├── __init__.py         # Package initialization
│   ├── transcriber.py      # Core WhisperTranscriber & Dataclasses
│   ├── audio_utils.py      # Audio validation, 16kHz resampling & float32 array converter
│   ├── exporters.py        # TXT, SRT, JSON exporters & timestamp formatters
│   ├── evaluate.py         # WER & CER calculations via jiwer
│   └── cli.py              # Argparse CLI interface
├── app.py                  # Streamlit Web UI Application
├── tests/                  # Pytest Unit & Integration Test Suite
│   ├── conftest.py         # Programmatic WAV audio generator fixtures
│   ├── test_audio_utils.py # Audio validation & conversion tests
│   ├── test_exporters.py   # Exporter format tests
│   ├── test_evaluate.py    # WER evaluation tests
│   └── test_transcriber.py # Whisper model integration tests
├── samples/                # Sample audio clips & reference transcripts
├── docs/                   # UI Screenshots, test logs & evaluation reports
│   ├── screenshot_ui.png
│   ├── test_results.txt
│   └── evaluation_metrics.json
├── requirements.txt        # Production dependency specifications
├── .gitignore              # Git ignore rules
├── LICENSE                 # MIT Open Source License
└── README.md               # Documentation
```

---

## 🔮 Limitations & Roadmap

- **Live Streaming Audio**: Current release processes full audio segments; streaming chunk recognition is planned for v1.1.
- **Speaker Diarization**: Multi-speaker identification via `pyannote.audio` integration coming soon.
- **Background Noise Filtering**: Additional preprocessing using spectral gating (`noisereduce`).

---

## 👤 Author & Acknowledgments

**Developed by Brijesh** ([@Brijeshv3001](https://github.com/Brijeshv3001))  
*University Project for Automatic Speech Recognition & Audio Processing.*

- Model Inference powered by [OpenAI Whisper](https://github.com/openai/whisper) & [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
- Web Interface built with [Streamlit](https://streamlit.io/).

---

<div align="center">
  <b>⭐ If you find this project helpful, consider giving it a star on GitHub! ⭐</b>
</div>
