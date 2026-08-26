# Real-Time Talking Avatar (MuseTalk 1.5 + WebRTC)

This document describes the self-hosted, genuinely real-time talking-avatar
pipeline: architecture, installation, launch, configuration, testing and
known limitations.

---

## 1. Architecture

```
┌──────────────────────────── Browser (web/index.html) ───────────────────────────┐
│  Chat UI ──► POST /api/chat ──► text answer + citations                         │
│       │                                                                         │
│       └──► POST /api/avatar/session/{id}/speak ─┐                               │
│                                                 │                               │
│  avatarVideo.srcObject ◄══ ONE WebRTC MediaStream (audio+video, one timeline) ══╪══┐
└──────────────────────────────────────────────────────────────────────────────────┘  │
                                                                                      │
Main server :8000 (scripts/server.py)        Real-time avatar service :8001           │
  RAG / Ollama LLM                           (scripts/realtime_avatar_server.py)      │
  /api/chat, /api/tts                        /api/avatar/health | capabilities        │
  (legacy MP4 jobs kept only for             /api/avatar/portrait (Pillow-validated,  │
   offline export)                            face-cropped, safe filenames)           │
                                             /api/avatar/session → offer → speak      │
                                                    │                               │
                                    ┌───────────────┴────────────────┐            │
                                    │  audio_pipeline.py             │            │
                                    │  • TTS generated EXACTLY ONCE  │            │
                                    │    (bounded LRU cache)         │            │
                                    │  • FFmpeg decode → 16 kHz f32  │            │
                                    │    (MuseTalk) + 48 kHz s16     │            │
                                    │    (WebRTC/Opus)               │            │
                                    └───────────────┬────────────────┘            │
                                                    │ PCM                         │
                                    ┌───────────────▼────────────────┐            │
                                    │  musetalk_worker.py            │            │
                                    │  • ONE persistent worker       │            │
                                    │  • MuseTalk 1.5 loaded once    │            │
                                    │  • FP16 + inference_mode       │            │
                                    │  • bounded queue (default 3)   │            │
                                    │  • single GPU lock             │            │
                                    └───────────────┬────────────────┘            │
                                                    │ RGB frames @25 fps          │
                                    ┌───────────────▼────────────────┐            │
                                    │  aiortc tracks                 │            │
                                    │  video frame i ↔ audio samples │            │
                                    │  [i·1920, (i+1)·1920) — same   │            │
                                    │  index ⇒ no drift              │            │
                                    └────────────────────────────────┘───────────┘

Idle: idle_animator.py pre-generates ONE seamless 12 s loop per portrait
(breathing + irregular blinking + micro head-sway) — no GPU inference while
nobody is speaking. On speech end the stream returns to the idle loop.

Fallback: if MuseTalk/CUDA is unavailable, /api/avatar/health reports
"degraded", the frontend badge switches to "CSS Fallback" and the existing
Web-Audio amplitude lip-sync avatar is used automatically.
```

## 2. Files added / modified

**Added**
| File | Purpose |
|---|---|
| `scripts/realtime_avatar_server.py` | FastAPI service: sessions, WebRTC offer/answer, speak/stop, portrait upload, health/capabilities |
| `scripts/musetalk_worker.py` | Persistent MuseTalk 1.5 worker (load-once, FP16, bounded queue, GPU lock, dry-run dev shim) |
| `scripts/audio_pipeline.py` | Single-generation TTS cache + FFmpeg PCM decode (16 kHz f32 / 48 kHz s16) |
| `scripts/portrait_preprocessing.py` | Pillow validation, EXIF fix, RGB, face/eye detection, hat-preserving head-and-shoulders crop |
| `scripts/idle_animator.py` | Pre-generated seamless idle loop (breathing, blinking, sway) |
| `scripts/test_realtime_avatar.py` | 12-test suite: preprocessing, single-TTS, decode, idle loop, full WebRTC speak/stop flow |
| `scripts/install_avatar.sh` | One-shot installer (PyTorch-CUDA, deps, MuseTalk, checkpoints, FFmpeg) |
| `requirements-avatar.txt` | Separated avatar dependencies |
| `docs/REALTIME_AVATAR.md` | This document |

**Modified**
| File | Change |
|---|---|
| `scripts/server.py` | Added missing `import shutil`; TTS pre-warm through the single-generation cache; `/api/tts` uses the same cache |
| `scripts/avatar_engine.py` | Engines accept pre-generated audio / reuse the cache (TTS once per answer); fixed MP3-bytes-in-`.wav` Wav2Lip bug; SadTalker documented as offline-export only |
| `web/index.html` | WebRTC client (`avatarVideo.srcObject`), real portrait upload + crop-adjust UI, full state machine, Stop/Replay, reconnect, cleanup, CSS fallback |
| `.gitignore` | Excludes avatar caches, portraits, MuseTalk checkpoints |

## 3. Installation

```bash
# one-shot (PyTorch cu121, deps, MuseTalk repo + checkpoints, FFmpeg)
bash scripts/install_avatar.sh

# or manually:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-avatar.txt
git clone https://github.com/TMElyralab/MuseTalk.git MuseTalk
pip install -r MuseTalk/requirements.txt
# checkpoints: see scripts/install_avatar.sh steps 5
sudo apt-get install -y ffmpeg
```

> `pip install -r requirements.txt` does **not** install CUDA-compatible
> PyTorch — install it from the PyTorch CUDA wheel index as above.

## 4. Launch

```bash
ollama serve                                                # terminal 1
python -X utf8 scripts/server.py --port 8000                # terminal 2 (chat UI + RAG)
python -X utf8 scripts/realtime_avatar_server.py            # terminal 3 (avatar, port 8001)
# open http://localhost:8000
```

## 5. Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `REALTIME_AVATAR_ENABLED` | `1` | Master switch for the realtime service |
| `REALTIME_AVATAR_PORT` | `8001` | Avatar service port |
| `MUSETALK_PATH` | `./MuseTalk` | MuseTalk repo path |
| `MUSETALK_MODEL_DIR` | `$MUSETALK_PATH/models` | Checkpoint directory |
| `AVATAR_DEVICE` | `cuda:0` | CUDA device selection |
| `AVATAR_FP16` | `1` | Half precision |
| `AVATAR_TARGET_FPS` | `25` | Video frame rate |
| `AVATAR_BATCH_SIZE` | `4` | MuseTalk frames per inference batch (lower = lower first-frame latency) |
| `AVATAR_BBOX_SHIFT` | `0` | MuseTalk face-crop bbox shift (tune if the mouth region is clipped) |
| `AVATAR_HEAD_MOTION` | `1` | Whole-frame animation during speech (sway + blinking) |
| `AVATAR_DEBUG_DUMP_DIR` | unset | If set, dumps the first 60 generated frames of each utterance |
| `AVATAR_MAX_QUEUE` | `3` | Bounded speech queue |
| `AVATAR_PORTRAIT_DIR` | `web/avatar_portraits` | Uploaded portrait store |
| `AVATAR_ALLOWED_ORIGINS` | `http://localhost:8000,...` | CORS allow-list (never `*` in production) |
| `AVATAR_MAX_UPLOAD_BYTES` | `10485760` | Portrait upload limit (10 MB) |
| `AVATAR_MAX_SPEECH_CHARS` | `5000` | Reject overly long speech requests |
| `AVATAR_SESSION_TTL_S` | `900` | Idle session expiry |
| `FFMPEG_PATH` | `ffmpeg` | FFmpeg binary |
| `AVATAR_ENGINE_MODE` | `musetalk` | `dryrun` = no-GPU pipeline test shim (NOT for production) |

## 6. Tests

```bash
# Full pipeline without a GPU (dry-run engine, stubbed TTS):
AVATAR_ENGINE_MODE=dryrun python -X utf8 scripts/test_realtime_avatar.py

# Existing legacy smoke tests:
python -X utf8 scripts/server.py --port 8000 &
python -X utf8 scripts/test_server.py
```

Verified in CI-style run (dry-run): portrait validation/crop, single-TTS
cache, FFmpeg decode, seamless idle loop, WebRTC idle stream (~25 fps video,
20 ms audio packets), synchronized speech (A/V span drift < 500 ms assertion),
Stop < 500 ms, Replay cache-hit, session deletion.

## 7. Expected resource usage (24 GB VRAM GPU)

| Component | VRAM |
|---|---|
| MuseTalk 1.5 (UNet FP16 + VAE + Whisper tiny) | ~3–4 GB |
| Qwen 2.5 3B via Ollama (q4) | ~2–3 GB |
| Headroom | > 15 GB |

Model-load time, TTS time, first-frame latency, achieved FPS, speech duration
and peak VRAM are logged per job by `musetalk_worker.py` and queryable via
`GET /api/avatar/session/{id}/stats` and `/api/avatar/health`.

## 8. Known limitations

1. **GPU numbers pending.** Latency/FPS/VRAM must be measured on the target
   24 GB machine; the sandbox used for development has no GPU. The dry-run
   suite validates the whole pipeline except MuseTalk inference itself.
2. **MuseTalk API drift.** `MuseTalkEngine` targets the MuseTalk 1.5
   `realtime_inference.py` API (`load_all_model`, `Audio2Feature`,
   `get_landmark_and_bbox`). If a future MuseTalk release renames these, the
   adapter in `musetalk_worker.py` is the single place to adjust.
3. **Whole-answer TTS (MVP).** Speech starts after the complete answer is
   synthesized (~1–3 s). Sentence-level LLM/TTS streaming is the planned
   phase 2 and does not block this release.
4. **Idle is procedural.** The idle loop is CPU-warp based (breathing/blink/
   sway), not generative — deliberate, to keep the GPU free.
5. **Haar detection limits.** Very dark/strongly shadowed portraits may need
   the manual crop-adjust controls (arrow/zoom buttons) after upload.
6. **Single GPU job at a time** by design (`AVATAR_MAX_QUEUE` bounds waiting
   jobs); concurrent multi-user GPU rendering is future work.
7. **WebRTC over WAN** needs STUN/TURN configuration; localhost/LAN work
   out of the box.
