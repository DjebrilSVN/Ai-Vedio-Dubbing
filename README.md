# AI Cinematic Dubbing Studio 

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue)
![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-Spaces-yellow)
![License](https://img.shields.io/badge/license-MIT-green)

An end-to-end, automated AI video dubbing pipeline that translates, voice-clones, synchronizes, and subtitles video content.

Unlike basic text-to-speech scripts, this project transforms raw video into a studio-quality **"Cinematic Dub"**. It preserves the original speaker's vocal identity, isolates and preserves background sound effects, applies audio "ducking," and precisely times the new dialogue to match the original video pacing.

---

## 📑 Table of Contents
1. About the Project
2. Key Features
3. The Tech Stack
4. Prerequisites
5. Deployment Options (How to Run)
6. Project Structure
7. Roadmap
8. Contributing
9. License

---

## 🔍 About the Project

The goal of this project is to democratize high-quality video localization. Typically, dubbing requires a studio, voice actors, and audio engineers. This pipeline automates the entire workflow using state-of-the-art open-source AI models.

It listens to video, separates dialogue from background sounds, translates content intelligently, clones the speaker's voice in another language, and merges everything back with perfect synchronization.

---

## ✨ Key Features

- **Zero-Shot Voice Cloning:** Replicates the timbre, pitch, and emotion of the original speaker.
- **Smart Source Separation:** Preserves background music and sound effects.
- **Dynamic Time-Stretching ("Gap Squisher"):** Prevents overlap using adaptive speed control.
- **Audio Ducking:** Reduces background audio during speech.
- **Noise Cancellation:** Removes AI artifacts.
- **Hardcoded Subtitles:** Generates and embeds subtitles directly into the video.

---

## ⚙️ The Tech Stack

- **Source Separation:** Demucs
- **Transcription & Diarization:** WhisperX, Pyannote
- **Translation:** Meta NLLB-200-distilled-1.3B
- **Voice Synthesis:** Coqui XTTS-v2
- **Media Processing:** FFmpeg, PyDub

---

## 💻 Prerequisites

- OS: Linux (Ubuntu recommended) or Windows (WSL2)
- GPU: NVIDIA GPU with 12GB+ VRAM (16GB recommended)
- CUDA installed
- Python 3.10+
- Docker & Docker Compose

---

## 🏗️ Deployment Options

### 🌟 1. Hugging Face App

```bash
git clone https://github.com/yourusername/ai-cinematic-dubbing.git
cd ai-cinematic-dubbing
pip install -r requirements-hf.txt
python app.py
```

---

### 🐳 2. Docker Microservices

- Service A: Transcription (port 8001)
- Service B: Translation (port 8002)
- Service C: Voice Synthesis (port 8003)

```bash
docker-compose up --build
```

---

### 📓 3. Jupyter Notebooks

- notebooks/01_Transcription.ipynb
- notebooks/02_Translation.ipynb
- notebooks/03_PostProduction.ipynb

---

## 📁 Project Structure

```
ai-cinematic-dubbing/
│
├── app.py
├── requirements-hf.txt
├── docker-compose.yml
│
├── services/
│   ├── service_a_hearing/
│   ├── service_b_intelligence/
│   └── service_c_synthesis/
│
├── notebooks/
│   ├── 01_Transcription.ipynb
│   ├── 02_Translation.ipynb
│   └── 03_PostProduction.ipynb
│
├── assets/
└── README.md
```

---

## 🗺️ Roadmap

- [x] English → French dubbing
- [x] Background audio preservation
- [x] Gradio UI
- [x] Add 10+ languages
- [x] Multi-speaker lip sync (Wav2Lip)
- [x] Optimize for low VRAM

---

## 🤝 Contributing

1. Fork the project
2. Create a branch (git checkout -b feature/AmazingFeature)
3. Commit changes
4. Push to GitHub
5. Open a Pull Request

---

## 📝 License

MIT License

Note: AI models may have separate licenses (e.g., CPML). Check before commercial use.
