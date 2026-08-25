#!/usr/bin/env python3
"""
Shared Audio Pipeline for Author AI
===================================
Responsibilities:
1. Single-generation TTS audio cache (bounded LRU) so each answer's audio
   is produced EXACTLY ONCE and reused by the browser player, the avatar
   worker and the Replay button.
2. FFmpeg-based decoding of Edge-TTS MP3 output into the normalized PCM
   format MuseTalk expects (mono, 16 kHz, float32), plus 48 kHz s16 PCM
   for WebRTC playback.
3. Safe temporary-file handling (all temp audio is deleted after use).

Nothing in this module ever trusts a client-supplied filesystem path.
"""

import os
import hashlib
import logging
import subprocess
import tempfile
import threading
from collections import OrderedDict
from typing import Optional, Tuple

import numpy as np

from tts_engine import generate_tts_audio, clean_text_for_speech

logger = logging.getLogger("AudioPipeline")

FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "ffmpeg")

# ---------------------------------------------------------------------------
# Bounded TTS audio cache
# ---------------------------------------------------------------------------
_AUDIO_CACHE: "OrderedDict[str, Tuple[bytes, str]]" = OrderedDict()
_AUDIO_CACHE_LOCK = threading.Lock()
_AUDIO_CACHE_MAX_ITEMS = int(os.environ.get("AVATAR_AUDIO_CACHE_MAX", "32"))
_AUDIO_CACHE_MAX_BYTES = int(os.environ.get("AVATAR_AUDIO_CACHE_MAX_BYTES", str(64 * 1024 * 1024)))
_audio_cache_bytes = 0


def audio_cache_key(text: str, voice: str) -> str:
    clean = clean_text_for_speech(text)
    return hashlib.sha256(f"{clean}|{voice}".encode("utf-8")).hexdigest()[:24]


def _cache_evict_if_needed() -> None:
    global _audio_cache_bytes
    while _AUDIO_CACHE and (
        len(_AUDIO_CACHE) > _AUDIO_CACHE_MAX_ITEMS or _audio_cache_bytes > _AUDIO_CACHE_MAX_BYTES
    ):
        _, (old_bytes, _) = _AUDIO_CACHE.popitem(last=False)
        _audio_cache_bytes -= len(old_bytes)
        logger.info("Audio cache eviction (size=%d items, %.1f MB)",
                    len(_AUDIO_CACHE), _audio_cache_bytes / 1e6)


def get_cached_audio(text: str, voice: str) -> Optional[Tuple[bytes, str]]:
    key = audio_cache_key(text, voice)
    with _AUDIO_CACHE_LOCK:
        if key in _AUDIO_CACHE:
            _AUDIO_CACHE.move_to_end(key)
            logger.info("Audio cache HIT key=%s (TTS NOT regenerated)", key)
            return _AUDIO_CACHE[key]
    return None


def synthesize_speech_once(text: str, voice: str, provider: str = None) -> Tuple[Optional[bytes], str]:
    """
    Generate TTS audio for (text, voice) exactly once.
    Returns (audio_bytes, mime). (None, 'browser') means browser fallback.
    """
    global _audio_cache_bytes
    cached = get_cached_audio(text, voice)
    if cached is not None:
        return cached

    audio_bytes, mime = generate_tts_audio(text, voice, provider=provider)
    if audio_bytes:
        key = audio_cache_key(text, voice)
        with _AUDIO_CACHE_LOCK:
            if key not in _AUDIO_CACHE:
                _AUDIO_CACHE[key] = (audio_bytes, mime)
                _audio_cache_bytes += len(audio_bytes)
                _cache_evict_if_needed()
                logger.info("Audio cached key=%s (%d bytes)", key, len(audio_bytes))
    return audio_bytes, mime


def clear_audio_cache() -> None:
    global _audio_cache_bytes
    with _AUDIO_CACHE_LOCK:
        _AUDIO_CACHE.clear()
        _audio_cache_bytes = 0


def audio_cache_stats() -> dict:
    with _AUDIO_CACHE_LOCK:
        return {"items": len(_AUDIO_CACHE), "bytes": _audio_cache_bytes,
                "max_items": _AUDIO_CACHE_MAX_ITEMS, "max_bytes": _AUDIO_CACHE_MAX_BYTES}


# ---------------------------------------------------------------------------
# FFmpeg decoding helpers
# ---------------------------------------------------------------------------
def _run_ffmpeg_decode(input_bytes: bytes, sample_rate: int, sample_fmt: str) -> np.ndarray:
    """
    Decode arbitrary audio bytes (mp3/wav/...) to raw PCM via FFmpeg stdin/stdout.
    sample_fmt: 'f32le' or 's16le'. Always mono.
    """
    if sample_fmt not in ("f32le", "s16le"):
        raise ValueError(f"unsupported sample_fmt {sample_fmt}")
    cmd = [
        FFMPEG_PATH, "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", sample_fmt,
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, input=input_bytes, capture_output=True, timeout=120)
    except FileNotFoundError as e:
        raise RuntimeError(f"FFmpeg not found at '{FFMPEG_PATH}'. Install ffmpeg or set FFMPEG_PATH.") from e
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg decode failed: {proc.stderr.decode('utf-8', 'ignore')[:400]}")
    dtype = np.float32 if sample_fmt == "f32le" else np.int16
    pcm = np.frombuffer(proc.stdout, dtype=dtype)
    if pcm.size == 0:
        raise RuntimeError("FFmpeg produced empty PCM output (corrupt audio input?)")
    return pcm


def decode_to_musetalk_pcm(audio_bytes: bytes, sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
    """Decode TTS audio into mono float32 PCM at the MuseTalk input rate (16 kHz)."""
    return _run_ffmpeg_decode(audio_bytes, sample_rate, "f32le"), sample_rate


def decode_to_webrtc_pcm(audio_bytes: bytes, sample_rate: int = 48000) -> Tuple[np.ndarray, int]:
    """Decode TTS audio into mono s16 PCM at the WebRTC/Opus rate (48 kHz)."""
    return _run_ffmpeg_decode(audio_bytes, sample_rate, "s16le"), sample_rate


def get_audio_duration_seconds(audio_bytes: bytes) -> float:
    """Return duration (seconds) of arbitrary encoded audio via ffprobe-less decode."""
    pcm = _run_ffmpeg_decode(audio_bytes, 8000, "s16le")
    return pcm.size / 8000.0
