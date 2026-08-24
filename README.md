# Author Voice & Talking-Avatar Pipeline

An end-to-end grounded RAG knowledge chatbot and **Interactive Talking Avatar Twin** for **Shahbuddin Rathod** — a famous Gujarati humorist and philosopher. 

The system extracts structured knowledge from all 20 of his published books, performs semantic retrieval (ChromaDB + multilingual-e5-base), streams neural text-to-speech, and drives an **interactive talking avatar** with lip-sync.

---

## 🌟 Key Features

1. **Grounded 20-Book Knowledge Base**: Answers questions strictly grounded in Shahbuddin Rathod's 20 books with exact page citations.
2. **Neural Speech Synthesis**: Powered by `edge-tts` (English & Gujarati voices), local `piper` TTS, or browser speech synthesis fallback.
3. **Pluggable Talking Avatar Pipeline**:
   - **SadTalkerEngine**: Photorealistic MP4 video generation from portrait images + audio (when SadTalker is installed).
   - **Wav2LipEngine**: Alternative video lip-sync engine.
   - **CssAvatarEngine**: Zero-dependency Web Audio API amplitude lip-sync engine (never breaks the site if video AI models are not installed).
4. **Interactive Web Application**: Responsive dark-mode UI with state machine badges, 3D mouse-tracking perspective, eye blinking, DOM XSS safety, Replay/Stop controls, and portrait photo uploading.
5. **Robust Automated Testing**: Smoke test suite verifying endpoints, startup validations, and fallbacks.

---

## 🚀 Quick Start Guide

### 1. Clone & Environment Setup

```bash
git clone https://github.com/Bhavya773-coder/Author_clone.git
cd Author_clone

# Create virtual environment
python -m venv .venv

# Activate environment
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Ollama Backend Setup

Install [Ollama](https://ollama.com) and download the LLM model:

```bash
# Start Ollama server
ollama serve

# In a separate terminal, pull Qwen 2.5:
ollama pull qwen2.5:3b
```

### 3. Launch Web Server

```bash
python -X utf8 scripts/server.py --port 8000
```

Open your browser and visit:
👉 **[http://localhost:8000/](http://localhost:8000/)**

---

## ⚙️ Environment Configuration

You can customize server and engine options using environment variables:

| Variable | Default Value | Description |
|----------|---------------|-------------|
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Local Ollama HTTP endpoint |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Local Ollama model name |
| `TTS_PROVIDER` | `edge-tts` | TTS provider (`edge-tts`, `piper`, `browser`) |
| `TTS_VOICE` | `en-US-AriaNeural` | Default neural voice preset |
| `AVATAR_ENGINE` | `sadtalker` | Avatar engine (`sadtalker`, `wav2lip`, `css`) |
| `AVATAR_PORTRAIT` | `web/avatar_character.jpg` | Path to default avatar portrait image |
| `AVATAR_OUTPUT_DIR` | `web/avatar_cache` | Path to store rendered avatar MP4 cache |
| `SADTALKER_PATH` | Path to SadTalker repo | Directory containing SadTalker `inference.py` |
| `WAV2LIP_PATH` | Path to Wav2Lip repo | Directory containing Wav2Lip `inference.py` |

Example launching with custom settings on Windows PowerShell:
```powershell
$env:TTS_VOICE="en-US-ChristopherNeural"
$env:AVATAR_ENGINE="sadtalker"
python -X utf8 scripts/server.py --port 8000
```

---

## 🎭 SadTalker / Wav2Lip Photorealistic Video Setup (Optional)

If you want real MP4 video generation instead of Web Audio lip-sync fallback:

1. **Install SadTalker**:
   ```bash
   git clone https://github.com/OpenTalker/SadTalker.git SadTalker
   cd SadTalker
   pip install -r requirements.txt
   # Download SadTalker pre-trained checkpoints to SadTalker/checkpoints/
   ```

2. **Set Environment Variable**:
   ```bash
   export SADTALKER_PATH="/absolute/path/to/SadTalker"
   ```

3. **GPU / CPU Expectations**:
   - **NVIDIA GPU (CUDA)**: Renders a 5-second video clip in ~3–8 seconds.
   - **CPU Mode**: Renders in ~30–60 seconds.
   - *Note:* The web app displays audio and CSS lip-sync immediately while asynchronously rendering video in the background!

---

## 🧪 Running Automated Tests

Run the automated smoke test suite to verify server endpoints, speech synthesis, and fallbacks:

```bash
python scripts/test_server.py --url http://127.0.0.1:8000
```

Sample output:
```
======================================================================
🧪 AUTHOR AI SERVER SMOKE TEST SUITE — Target: http://127.0.0.1:8000
======================================================================
  ✅ PASS: GET / (Web UI) Status: 200
  ✅ PASS: POST /api/chat Job ID: job_1787573...
  ✅ PASS: POST /api/tts MIME: audio/mpeg, Bytes: 24891
  ✅ PASS: POST /api/avatar/speak Enqueued Job ID: job_1787573...
  ✅ PASS: GET /api/avatar/status/<job_id> Status: done, Engine: CssAvatarEngine
  ✅ PASS: Error Handling (Empty Query) Returned HTTP 400 as expected
======================================================================
📊 TEST SUMMARY: 5 PASSED | 0 FAILED
======================================================================
```

---

## 📜 Safety & Consent Disclosure

> **Notice:** AI character twin demonstration. Only use photos/voices with explicit permission. AI responses are synthesized for demonstration and do not represent official statements of real persons.
