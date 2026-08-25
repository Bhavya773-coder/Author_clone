#!/usr/bin/env python3
"""
Pluggable Talking Avatar Engine for Author AI
=============================================
Provides pluggable avatar rendering engines:
1. SadTalkerEngine (Generates photorealistic MP4 video from image + audio if installed)
2. Wav2LipEngine (Alternative video lip-sync engine if configured)
3. CssAvatarEngine (Graceful zero-dependency fallback using Web Audio API amplitude lip-sync)

Includes caching by hash(text + voice + portrait + engine), job queueing, worker pool, and path sanitization.
"""

import os
import sys
import hashlib
import time
import shutil
import subprocess
import threading
import queue
import logging
from pathlib import Path
from typing import Dict, Any

from tts_engine import clean_text_for_speech
from audio_pipeline import synthesize_speech_once

logger = logging.getLogger("AvatarEngine")

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_PORTRAIT = os.environ.get("AVATAR_PORTRAIT", str(PROJECT_ROOT / "web" / "avatar_character.jpg"))
AVATAR_OUTPUT_DIR = Path(os.environ.get("AVATAR_OUTPUT_DIR", str(PROJECT_ROOT / "web" / "avatar_cache"))).resolve()
PREFERRED_ENGINE = os.environ.get("AVATAR_ENGINE", "sadtalker").lower().strip()

# Ensure cache directory exists
AVATAR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job registry
JOBS: Dict[str, Dict[str, Any]] = {}
JOB_QUEUE = queue.Queue()
WORKER_THREAD = None


def get_job_hash(text: str, voice: str, portrait_path: str, engine_name: str) -> str:
    """Generate deterministic hash for caching generated avatar videos."""
    clean_text = clean_text_for_speech(text)
    raw = f"{clean_text}|{voice}|{portrait_path}|{engine_name}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def is_sadtalker_available() -> bool:
    """Check if SadTalker environment/repo is installed and configured."""
    st_path = os.environ.get("SADTALKER_PATH")
    if st_path and Path(st_path).exists() and (Path(st_path) / "inference.py").exists():
        return True
    # Check if sadtalker command or inference script exists in project
    local_st = PROJECT_ROOT / "SadTalker" / "inference.py"
    return local_st.exists()


def is_wav2lip_available() -> bool:
    """Check if Wav2Lip environment/repo is installed and configured."""
    w2l_path = os.environ.get("WAV2LIP_PATH")
    if w2l_path and Path(w2l_path).exists() and (Path(w2l_path) / "inference.py").exists():
        return True
    local_w2l = PROJECT_ROOT / "Wav2Lip" / "inference.py"
    return local_w2l.exists()


class CssAvatarEngine:
    """Fallback engine returning Web Audio amplitude lip-sync signal without video rendering overhead."""
    name = "CssAvatarEngine"

    def render(self, text: str, voice: str, portrait_path: str, output_mp4: Path) -> Dict[str, Any]:
        return {
            "engine": self.name,
            "status": "done",
            "is_video": False,
            "message": "Using Web Audio API real-time lip-sync engine (SadTalker/Wav2Lip not installed)"
        }


class SadTalkerEngine:
    """Generates photorealistic talking-head video using SadTalker.
    NOTE: retained ONLY for optional offline/high-quality MP4 export —
    never used for the live avatar stream (MuseTalk + WebRTC handles that)."""
    name = "SadTalkerEngine"

    def render(self, text: str, voice: str, portrait_path: str, output_mp4: Path,
               audio_bytes: bytes = None) -> Dict[str, Any]:
        if not is_sadtalker_available():
            raise RuntimeError("SadTalker is not installed or SADTALKER_PATH is invalid.")

        # 1. Reuse pre-generated audio when provided (TTS runs exactly once per answer)
        if not audio_bytes:
            audio_bytes, _mime = synthesize_speech_once(text, voice, provider="edge-tts")
        if not audio_bytes:
            raise RuntimeError("TTS generation failed for SadTalker input.")

        temp_audio = output_mp4.parent / f"temp_{output_mp4.stem}.mp3"
        temp_audio.write_bytes(audio_bytes)

        st_path = Path(os.environ.get("SADTALKER_PATH", str(PROJECT_ROOT / "SadTalker"))).resolve()
        st_script = st_path / "inference.py"

        # Sanitize path arguments to prevent command injection
        cmd = [
            sys.executable,
            str(st_script),
            "--driven_audio", str(temp_audio),
            "--source_image", str(portrait_path),
            "--result_dir", str(output_mp4.parent),
            "--still",
            "--preprocess", "full"
        ]

        logger.info(f"Running SadTalker: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

        if temp_audio.exists():
            temp_audio.unlink()

        if result.returncode != 0:
            raise RuntimeError(f"SadTalker process error: {result.stderr[-500:]}")

        # Locate generated MP4 and move to output_mp4
        generated_files = list(output_mp4.parent.glob("*.mp4"))
        if generated_files:
            latest = max(generated_files, key=os.path.getctime)
            if latest != output_mp4:
                shutil.move(str(latest), str(output_mp4))
            return {"engine": self.name, "status": "done", "is_video": True}
        else:
            raise RuntimeError("SadTalker executed but generated no MP4 output.")


class Wav2LipEngine:
    """Generates lip-synced video using Wav2Lip."""
    name = "Wav2LipEngine"

    def render(self, text: str, voice: str, portrait_path: str, output_mp4: Path,
               audio_bytes: bytes = None) -> Dict[str, Any]:
        if not is_wav2lip_available():
            raise RuntimeError("Wav2Lip is not installed or WAV2LIP_PATH is invalid.")

        if not audio_bytes:
            audio_bytes, _mime = synthesize_speech_once(text, voice, provider="edge-tts")
        if not audio_bytes:
            raise RuntimeError("TTS generation failed for Wav2Lip input.")

        # Edge-TTS returns MP3 bytes — decode to a REAL RIFF/WAV via FFmpeg.
        # (The old code wrote MP3 bytes into a .wav file, which corrupts Wav2Lip input.)
        from audio_pipeline import _run_ffmpeg_decode
        import wave
        pcm = _run_ffmpeg_decode(audio_bytes, 16000, "s16le")
        temp_audio = output_mp4.parent / f"temp_w2l_{output_mp4.stem}.wav"
        with wave.open(str(temp_audio), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(pcm.tobytes())

        w2l_path = Path(os.environ.get("WAV2LIP_PATH", str(PROJECT_ROOT / "Wav2Lip"))).resolve()
        w2l_script = w2l_path / "inference.py"
        checkpoint = w2l_path / "checkpoints" / "wav2lip_gan.pth"

        cmd = [
            sys.executable,
            str(w2l_script),
            "--checkpoint_path", str(checkpoint),
            "--face", str(portrait_path),
            "--audio", str(temp_audio),
            "--outfile", str(output_mp4)
        ]

        logger.info(f"Running Wav2Lip: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if temp_audio.exists():
            temp_audio.unlink()

        if result.returncode != 0 or not output_mp4.exists():
            raise RuntimeError(f"Wav2Lip process error: {result.stderr[-500:]}")

        return {"engine": self.name, "status": "done", "is_video": True}


def select_avatar_engine(preferred: str = None):
    """Select appropriate avatar rendering engine with robust fallbacks."""
    pref = (preferred or PREFERRED_ENGINE).lower().strip()

    if pref in ("sadtalker", "sadtalkerengine") and is_sadtalker_available():
        return SadTalkerEngine()
    if pref in ("wav2lip", "wav2lipengine") and is_wav2lip_available():
        return Wav2LipEngine()
    if is_sadtalker_available():
        return SadTalkerEngine()
    if is_wav2lip_available():
        return Wav2LipEngine()

    return CssAvatarEngine()


def process_avatar_job(job_id: str, text: str, voice: str, portrait_path: str,
                       preferred_engine: str, audio_bytes: bytes = None):
    """Background worker function executing avatar video generation with fallback."""
    job = JOBS.get(job_id)
    if not job:
        return

    job["status"] = "processing"
    engine = select_avatar_engine(preferred_engine)
    job["engine"] = engine.name

    job_hash = get_job_hash(text, voice, portrait_path, engine.name)
    cached_mp4 = AVATAR_OUTPUT_DIR / f"{job_hash}.mp4"

    if cached_mp4.exists() and engine.name != "CssAvatarEngine":
        logger.info(f"Serving cached avatar video for job {job_id}")
        job["status"] = "done"
        job["video_url"] = f"/api/avatar/video/{job_hash}"
        job["is_video"] = True
        return

    try:
        res = engine.render(text, voice, portrait_path, cached_mp4, audio_bytes=audio_bytes)
        job["status"] = "done"
        job["is_video"] = res.get("is_video", False)
        if res.get("is_video") and cached_mp4.exists():
            job["video_url"] = f"/api/avatar/video/{job_hash}"
        else:
            job["video_url"] = None
            job["message"] = res.get("message", "CSS audio lip-sync fallback active")
    except Exception as e:
        logger.error(f"Avatar job {job_id} failed with engine {engine.name}: {e}")
        # Fall back to CssAvatarEngine on error
        job["status"] = "done"
        job["engine"] = "CssAvatarEngine"
        job["is_video"] = False
        job["video_url"] = None
        job["error_fallback"] = str(e)
        job["message"] = f"Video engine ({engine.name}) failed: {e}. Falling back to Web Audio API lip-sync."


def _worker_loop():
    """Continuous background worker thread for avatar jobs."""
    while True:
        try:
            job_item = JOB_QUEUE.get()
            if job_item is None:
                break
            job_id, text, voice, portrait_path, engine, audio_bytes = job_item
            process_avatar_job(job_id, text, voice, portrait_path, engine, audio_bytes)
            JOB_QUEUE.task_done()
        except Exception as e:
            logger.error(f"Worker queue error: {e}")


def start_worker_thread():
    global WORKER_THREAD
    if WORKER_THREAD is None or not WORKER_THREAD.is_alive():
        WORKER_THREAD = threading.Thread(target=_worker_loop, daemon=True)
        WORKER_THREAD.start()


# Initialize worker thread on module load
start_worker_thread()


def enqueue_avatar_job(text: str, voice: str = "en-US-AriaNeural", portrait_path: str = None, engine: str = None) -> str:
    """Create and enqueue an avatar rendering job."""
    clean_text = clean_text_for_speech(text)
    portrait = str(Path(portrait_path).resolve()) if portrait_path else DEFAULT_PORTRAIT

    job_id = f"job_{int(time.time()*1000)}_{os.urandom(4).hex()}"
    selected_engine = (engine or PREFERRED_ENGINE).lower()

    # Check cache first
    engine_obj = select_avatar_engine(selected_engine)
    job_hash = get_job_hash(clean_text, voice, portrait, engine_obj.name)
    cached_mp4 = AVATAR_OUTPUT_DIR / f"{job_hash}.mp4"

    if cached_mp4.exists() and engine_obj.name != "CssAvatarEngine":
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "done",
            "engine": engine_obj.name,
            "is_video": True,
            "video_url": f"/api/avatar/video/{job_hash}"
        }
        return job_id

    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "engine": engine_obj.name,
        "video_url": None
    }

    JOB_QUEUE.put((job_id, clean_text, voice, portrait, selected_engine))
    return job_id


def get_avatar_job_status(job_id: str) -> Dict[str, Any]:
    """Retrieve current status of an avatar job."""
    return JOBS.get(job_id, {"status": "error", "error": "Job ID not found"})
