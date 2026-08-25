#!/usr/bin/env python3
"""
Real-Time Avatar Service (FastAPI + aiortc + MuseTalk 1.5)
==========================================================
Standalone real-time talking-avatar service. Audio (Edge-TTS) and
MuseTalk-generated video share ONE WebRTC connection and ONE timeline, so
lip-sync cannot drift.

Endpoints
---------
GET    /api/avatar/health                     engine & service status
GET    /api/avatar/capabilities               engine, fps, current portrait preview URL
POST   /api/avatar/portrait                   multipart upload -> validated, face-cropped, stored
GET    /api/avatar/portrait/preview/{id}      processed portrait preview (never a fs path)
POST   /api/avatar/session                    create avatar session
POST   /api/avatar/session/{sid}/offer        WebRTC SDP offer -> answer (A/V on one connection)
POST   /api/avatar/session/{sid}/speak        {text, voice} -> TTS once, stream via WebRTC
POST   /api/avatar/session/{sid}/stop         interrupt speech (< ~500 ms)
DELETE /api/avatar/session/{sid}              tear down session & peer connection

Launch:  python -X utf8 scripts/realtime_avatar_server.py
"""

import os
import sys
import time
import uuid
import queue
import asyncio
import hashlib
import logging
import threading
from fractions import Fraction
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from audio_pipeline import (
    synthesize_speech_once, decode_to_musetalk_pcm, decode_to_webrtc_pcm, audio_cache_stats,
)
from portrait_preprocessing import preprocess_portrait, PortraitError, OUTPUT_SIZE
from idle_animator import build_idle_loop
from musetalk_worker import WORKER, TARGET_FPS, SAMPLES_PER_FRAME, AUDIO_SAMPLE_RATE

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RealtimeAvatarServer")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REALTIME_AVATAR_ENABLED = os.environ.get("REALTIME_AVATAR_ENABLED", "1") == "1"
PORT = int(os.environ.get("REALTIME_AVATAR_PORT", "8001"))
PORTRAIT_DIR = Path(os.environ.get("AVATAR_PORTRAIT_DIR",
                                   str(PROJECT_ROOT / "web" / "avatar_portraits"))).resolve()
PORTRAIT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_PORTRAIT_PATH = Path(os.environ.get(
    "AVATAR_PORTRAIT", str(PROJECT_ROOT / "web" / "avatar_character.jpg"))).resolve()
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get(
    "AVATAR_ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",") if o.strip()]
SESSION_TTL_S = int(os.environ.get("AVATAR_SESSION_TTL_S", "900"))
MAX_SPEECH_CHARS = int(os.environ.get("AVATAR_MAX_SPEECH_CHARS", "5000"))
WEBRTC_SAMPLE_RATE = 48000
SAMPLES_PER_PACKET = 960          # 20 ms @ 48 kHz
SAMPLES_PER_VIDEO_FRAME = WEBRTC_SAMPLE_RATE // TARGET_FPS  # 1920 @ 25 fps

# Current avatar source (server-side state; the browser only sees portrait_ids)
CURRENT_PORTRAIT_ID: Optional[str] = None
PORTRAITS: Dict[str, dict] = {}   # portrait_id -> {image, key, eyes, path, idle_loop, created}


def _portrait_key(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()[:20]


def register_portrait(raw_bytes: bytes, manual_offset=None, manual_zoom=None) -> dict:
    """Preprocess and register a portrait; returns {portrait_id, preview_url, warnings,...}."""
    global CURRENT_PORTRAIT_ID
    result = preprocess_portrait(raw_bytes, manual_offset=manual_offset, manual_zoom=manual_zoom)
    portrait_id = uuid.uuid4().hex[:16]
    out_path = PORTRAIT_DIR / f"{portrait_id}.jpg"  # safe server-generated filename
    result["image"].save(out_path, "JPEG", quality=92)

    key = _portrait_key(out_path.read_bytes())
    PORTRAITS[portrait_id] = {
        "image": result["image"],
        "key": key,
        "eyes": result["eyes"],
        "path": out_path,
        "idle_loop": None,   # built lazily on first session use
        "created": time.time(),
    }
    CURRENT_PORTRAIT_ID = portrait_id
    logger.info("Portrait registered: id=%s face=%s crop=%s warnings=%s",
                portrait_id, result["face_box"], result["crop_box"], result["warnings"])
    return {
        "portrait_id": portrait_id,
        "preview_url": f"/api/avatar/portrait/preview/{portrait_id}",
        "face_box": result["face_box"],
        "crop_box": result["crop_box"],
        "eyes_detected": result["eyes"] is not None,
        "warnings": result["warnings"],
    }


def get_current_portrait() -> dict:
    global CURRENT_PORTRAIT_ID
    if CURRENT_PORTRAIT_ID is None or CURRENT_PORTRAIT_ID not in PORTRAITS:
        if not DEFAULT_PORTRAIT_PATH.exists():
            raise PortraitError(f"Default portrait missing at {DEFAULT_PORTRAIT_PATH}")
        register_portrait(DEFAULT_PORTRAIT_PATH.read_bytes())
    return PORTRAITS[CURRENT_PORTRAIT_ID]


# ---------------------------------------------------------------------------
# WebRTC tracks — one shared timeline for audio & video
# ---------------------------------------------------------------------------
try:
    import av  # noqa: F401
    from aiortc import MediaStreamTrack  # noqa: F401
    _AIORTC_AVAILABLE = True
except ImportError:
    _AIORTC_AVAILABLE = False
    MediaStreamTrack = object  # type: ignore


def _require_aiortc():
    return _AIORTC_AVAILABLE


class _PacedTrack:
    """Common pacing helper: pts derived from a packet counter, paced by wall clock."""

    def _init_pacing(self):
        self._counter = 0
        self._start: Optional[float] = None

    def _next_pts(self, step: int, rate: int) -> Tuple[int, Fraction]:
        pts = self._counter
        self._counter += step
        return pts, Fraction(1, rate)

    async def _pace(self, pts: int, rate: int) -> None:
        now = time.time()
        if self._start is None:
            self._start = now - pts / rate
        target = self._start + pts / rate
        delay = target - now
        if delay > 0:
            await asyncio.sleep(delay)


class AvatarVideoTrack(MediaStreamTrack, _PacedTrack):
    """Streams idle-loop frames, switching to MuseTalk frames during speech."""

    kind = "video"

    def __init__(self, session: "AvatarSession"):
        super().__init__()
        self._init_pacing()
        self.session = session
        self._last_frame: Optional[np.ndarray] = None
        self._idle_idx = 0

    async def recv(self):
        import av
        pts, time_base = self._next_pts(90000 // TARGET_FPS, 90000)
        await self._pace(pts, 90000)

        frame: Optional[np.ndarray] = None
        speech = self.session.speech_state
        if speech is not None:
            # Align video frame to the audio timeline by frame index
            idx = self._counter // (90000 // TARGET_FPS) - 1
            if speech.ended_at(idx * SAMPLES_PER_VIDEO_FRAME):
                # Utterance finished -> smoothly return to the idle loop
                logger.info("Session %s speech ended (stats=%s) — returning to idle",
                            self.session.session_id,
                            {k: round(v, 3) if isinstance(v, float) else v
                             for k, v in speech.stats.items()})
                self.session.speech_state = None
                speech = None
            else:
                frame = speech.get_video_frame(idx)

        if frame is None:
            loop = self.session.idle_loop
            if loop:
                frame = loop[self._idle_idx % len(loop)]
                self._idle_idx += 1

        if frame is None:
            frame = np.zeros((OUTPUT_SIZE[1], OUTPUT_SIZE[0], 3), dtype=np.uint8)

        self._last_frame = frame
        vf = av.VideoFrame.from_ndarray(frame, format="rgb24")
        vf.pts = pts
        vf.time_base = time_base
        return vf


class AvatarAudioTrack(MediaStreamTrack, _PacedTrack):
    """Streams 20 ms speech PCM packets; silence when idle."""

    kind = "audio"

    def __init__(self, session: "AvatarSession"):
        super().__init__()
        self._init_pacing()
        self.session = session

    async def recv(self):
        import av
        pts, time_base = self._next_pts(SAMPLES_PER_PACKET, WEBRTC_SAMPLE_RATE)
        await self._pace(pts, WEBRTC_SAMPLE_RATE)

        speech = self.session.speech_state
        if speech is not None:
            samples = speech.get_audio_packet(pts)
        else:
            samples = np.zeros(SAMPLES_PER_PACKET, dtype=np.int16)

        af = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")
        af.pts = pts
        af.time_base = time_base
        af.sample_rate = WEBRTC_SAMPLE_RATE
        return af


class SpeechState:
    """Holds one utterance's audio + generated frames, addressed by timeline index."""

    def __init__(self, pcm48: np.ndarray, frame_queue: "queue.Queue",
                 cancel_event: threading.Event, stats: dict):
        self.pcm48 = pcm48
        self.frame_queue = frame_queue
        self.cancel_event = cancel_event
        self.stats = stats
        self.frames: Dict[int, np.ndarray] = {}
        self.generator_done = False
        self._pump = threading.Thread(target=self._pump_frames, daemon=True)
        self._pump.start()

    def _pump_frames(self):
        while True:
            try:
                item = self.frame_queue.get(timeout=SESSION_TTL_S)
            except queue.Empty:
                break
            if item is None:
                self.generator_done = True
                break
            idx, frame = item
            self.frames[idx] = frame
            # bound memory: drop frames far behind the playhead
            if len(self.frames) > TARGET_FPS * 120:
                oldest = min(self.frames)
                self.frames.pop(oldest, None)

    def get_video_frame(self, idx: int) -> Optional[np.ndarray]:
        if idx in self.frames:
            return self.frames[idx]
        # hold nearest available earlier frame to avoid flicker when renderer lags
        if self.frames:
            earlier = [k for k in self.frames if k < idx]
            if earlier:
                return self.frames[max(earlier)]
            return None  # not started yet -> caller shows idle frame
        return None if not self.generator_done else None

    def get_audio_packet(self, pts: int) -> np.ndarray:
        start = pts
        end = pts + SAMPLES_PER_PACKET
        if start >= len(self.pcm48):
            return np.zeros(SAMPLES_PER_PACKET, dtype=np.int16)
        packet = self.pcm48[start:end]
        if len(packet) < SAMPLES_PER_PACKET:
            packet = np.pad(packet, (0, SAMPLES_PER_PACKET - len(packet)))
        return packet

    def ended_at(self, pts: int) -> bool:
        """True once the audio timeline has passed the end of the utterance
        and the frame generator has finished — the session returns to idle."""
        return pts >= len(self.pcm48) and self.generator_done


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
class AvatarSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created = time.time()
        self.last_activity = time.time()
        self.pc = None
        self.idle_loop = None
        self.speech_state: Optional[SpeechState] = None
        self.cancel_event: Optional[threading.Event] = None
        self.stats: dict = {}
        self.lock = threading.Lock()

    def touch(self):
        self.last_activity = time.time()

    def ensure_idle_loop(self):
        if self.idle_loop is None:
            portrait = get_current_portrait()
            if portrait.get("idle_loop") is None:
                portrait["idle_loop"] = build_idle_loop(portrait["image"], portrait.get("eyes"))
            self.idle_loop = portrait["idle_loop"]

    def start_speech(self, pcm16k: np.ndarray, pcm48: np.ndarray) -> dict:
        """Cancel any current speech, then enqueue a new generation job."""
        self.stop_speech()
        with self.lock:
            cancel_event = threading.Event()
            frame_queue: "queue.Queue" = queue.Queue(maxsize=TARGET_FPS * 10)
            stats: dict = {"speech_duration_s": len(pcm16k) / AUDIO_SAMPLE_RATE,
                           "t_enqueued": time.time()}
            portrait = get_current_portrait()
            ok = WORKER.submit_speak(portrait["image"], portrait["key"],
                                      pcm16k, frame_queue, cancel_event, stats)
            if not ok:
                raise HTTPException(status_code=503,
                                    detail="Avatar engine unavailable or queue full — use CSS fallback.")
            self.speech_state = SpeechState(pcm48, frame_queue, cancel_event, stats)
            self.cancel_event = cancel_event
            self.stats = stats
            self.idle_idx = 0
            return stats

    def stop_speech(self):
        with self.lock:
            if self.cancel_event is not None:
                self.cancel_event.set()
            self.cancel_event = None
            self.speech_state = None


SESSIONS: Dict[str, AvatarSession] = {}


def get_session(sid: str) -> AvatarSession:
    session = SESSIONS.get(sid)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown or expired session_id")
    session.touch()
    return session


async def session_reaper():
    while True:
        await asyncio.sleep(60)
        now = time.time()
        expired = [sid for sid, s in SESSIONS.items() if now - s.last_activity > SESSION_TTL_S]
        for sid in expired:
            logger.info("Expiring inactive avatar session %s", sid)
            await close_session(sid)


async def close_session(sid: str):
    session = SESSIONS.pop(sid, None)
    if session is None:
        return
    session.stop_speech()
    if session.pc is not None:
        try:
            await session.pc.close()
        except Exception as e:
            logger.warning("Error closing peer connection for %s: %s", sid, e)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Real-Time Avatar Service", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,   # never "*" in production
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


class OfferBody(BaseModel):
    sdp: str
    type: str


class SpeakBody(BaseModel):
    text: str
    voice: str = "en-US-AriaNeural"


@app.on_event("startup")
async def _startup():
    if not REALTIME_AVATAR_ENABLED:
        logger.warning("REALTIME_AVATAR_ENABLED=0 — service running in disabled mode.")
        return
    if not _require_aiortc():
        logger.error("aiortc/PyAV not installed — WebRTC endpoints will fail. pip install aiortc av")
    t0 = time.time()
    WORKER.start()   # loads MuseTalk ONCE
    logger.info("Avatar worker start-up took %.1fs (available=%s)", time.time() - t0, WORKER.available)
    try:
        get_current_portrait()   # pre-process default portrait at startup
    except PortraitError as e:
        logger.error("Default portrait preprocessing failed: %s", e)
    asyncio.create_task(session_reaper())


@app.get("/api/avatar/health")
async def health():
    return {
        "status": "ok" if WORKER.available else "degraded",
        "realtime_enabled": REALTIME_AVATAR_ENABLED,
        "engine": WORKER.status(),
        "sessions": len(SESSIONS),
        "audio_cache": audio_cache_stats(),
    }


@app.get("/api/avatar/capabilities")
async def capabilities():
    current_id = CURRENT_PORTRAIT_ID
    return {
        "engine": WORKER.status(),
        "target_fps": TARGET_FPS,
        "webrtc": _require_aiortc(),
        "portrait_id": current_id,
        "portrait_preview_url": f"/api/avatar/portrait/preview/{current_id}" if current_id else None,
        "fallback": "css",   # frontend uses existing CSS avatar when engine unavailable
    }


@app.post("/api/avatar/portrait")
async def upload_portrait(file: UploadFile = File(...),
                          offset_x: float = 0.0, offset_y: float = 0.0, zoom: float = 1.0):
    """
    Upload (or re-upload with manual crop adjustment) a portrait.
    offset_x/offset_y in [-1, 1] shift the crop; zoom > 1 zooms in.
    """
    offset_x = max(-1.0, min(1.0, offset_x))
    offset_y = max(-1.0, min(1.0, offset_y))
    zoom = max(0.5, min(3.0, zoom))
    raw = await file.read()
    try:
        info = register_portrait(raw, manual_offset=(offset_x, offset_y), manual_zoom=zoom)
    except PortraitError as e:
        raise HTTPException(status_code=422, detail=str(e))
    # invalidate idle loops of active sessions so the new face appears
    for s in SESSIONS.values():
        s.idle_loop = None
    return info


@app.get("/api/avatar/portrait/preview/{portrait_id}")
async def portrait_preview(portrait_id: str):
    entry = PORTRAITS.get(portrait_id)
    if entry is None or not entry["path"].exists():
        raise HTTPException(status_code=404, detail="Unknown portrait_id")
    return FileResponse(entry["path"], media_type="image/jpeg")


@app.post("/api/avatar/session")
async def create_session():
    sid = uuid.uuid4().hex[:20]
    session = AvatarSession(sid)
    try:
        session.ensure_idle_loop()
    except PortraitError as e:
        raise HTTPException(status_code=500, detail=str(e))
    SESSIONS[sid] = session
    logger.info("Avatar session created: %s", sid)
    return {"session_id": sid, "ttl_s": SESSION_TTL_S}


@app.post("/api/avatar/session/{sid}/offer")
async def webrtc_offer(sid: str, body: OfferBody):
    if not _require_aiortc():
        raise HTTPException(status_code=503, detail="aiortc/PyAV not installed on the server")
    from aiortc import RTCPeerConnection, RTCSessionDescription

    session = get_session(sid)
    if session.pc is not None:
        await session.pc.close()   # reconnect: drop the old peer connection

    pc = RTCPeerConnection()
    session.pc = pc
    pc.addTrack(AvatarVideoTrack(session))
    pc.addTrack(AvatarAudioTrack(session))

    @pc.on("connectionstatechange")
    async def _on_state():
        logger.info("Session %s WebRTC state: %s", sid, pc.connectionState)
        if pc.connectionState in ("failed", "closed"):
            session.touch()

    await pc.setRemoteDescription(RTCSessionDescription(sdp=body.sdp, type=body.type))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


@app.post("/api/avatar/session/{sid}/speak")
async def speak(sid: str, body: SpeakBody):
    session = get_session(sid)
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty speech text")
    if len(text) > MAX_SPEECH_CHARS:
        raise HTTPException(status_code=400,
                            detail=f"Speech text too long (max {MAX_SPEECH_CHARS} chars)")
    if not WORKER.available:
        raise HTTPException(status_code=503,
                            detail=f"Avatar engine unavailable: {WORKER.last_error}. Use CSS fallback.")

    t0 = time.time()
    audio_bytes, mime = synthesize_speech_once(text, body.voice)   # TTS generated EXACTLY ONCE
    if not audio_bytes:
        raise HTTPException(status_code=502, detail="TTS generation failed (edge-tts unreachable?)")
    t_tts = time.time() - t0

    try:
        pcm16k, _ = decode_to_musetalk_pcm(audio_bytes)
        pcm48, _ = decode_to_webrtc_pcm(audio_bytes)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    t_decode = time.time() - t0 - t_tts

    stats = session.start_speech(pcm16k, pcm48)
    stats["tts_time_s"] = t_tts
    stats["decode_time_s"] = t_decode
    logger.info("speak: tts=%.2fs decode=%.2fs dur=%.1fs session=%s",
                t_tts, t_decode, stats["speech_duration_s"], sid)
    return {
        "status": "speaking",
        "speech_duration_s": stats["speech_duration_s"],
        "tts_time_s": round(t_tts, 3),
        "decode_time_s": round(t_decode, 3),
        "audio_cached": t_tts < 0.05,   # cache hits are ~instant
    }


@app.post("/api/avatar/session/{sid}/stop")
async def stop(sid: str):
    session = get_session(sid)
    session.stop_speech()
    return {"status": "stopped"}


@app.get("/api/avatar/session/{sid}/stats")
async def session_stats(sid: str):
    session = get_session(sid)
    return session.stats or {"status": "no speech yet"}


@app.delete("/api/avatar/session/{sid}")
async def delete_session(sid: str):
    if sid not in SESSIONS:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    await close_session(sid)
    return {"status": "closed"}


def main():
    import uvicorn
    logger.info("Starting Real-Time Avatar Service on http://0.0.0.0:%d", PORT)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")


if __name__ == "__main__":
    main()
