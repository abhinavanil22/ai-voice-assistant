# Gemini Voice Assistant (Mini Project)

A simple voice assistant built with Python that uses **Google Gemini** for:
- Speech-to-text (transcribing your voice)
- Text generation (LLM brain)
- Text-to-speech (replying back in audio)

## Features

- Record from microphone (5 seconds per turn)
- Transcribe speech using Gemini
- Chat with context (remembers previous turns in this run)
- Respond with synthesized speech
- All API keys stored securely in `.env` (not committed to git)

## Requirements

- Python 3.10+ (recommended)
- `pip` package manager
- A valid **GEMINI_API_KEY**

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate    # on Linux / macOS
venv\Scripts\activate       # on Windows

# 3. Install dependencies
pip install -r requirements.txt