#!/usr/bin/env python3
"""
TTS Abstraction Layer for Author AI
===================================
Provides text cleaning, citation removal, and speech synthesis providers:
1. edge-tts (Default online high-quality neural voices)
2. local/piper (Optional offline TTS if configured)
3. browser (Frontend SpeechSynthesis fallback signal)
"""

import os
import re
import sys
import asyncio
import logging
from pathlib import Path

logger = logging.getLogger("TTSEngine")

DEFAULT_VOICE = os.environ.get("TTS_VOICE", "en-US-AriaNeural")
DEFAULT_PROVIDER = os.environ.get("TTS_PROVIDER", "edge-tts")


def clean_text_for_speech(text: str) -> str:
    """Clean citations, markdown formatting, emojis, and special chars for TTS."""
    if not text:
        return ""
    # Strip book source citations e.g. [Book-01, p.5] or ["Show Must Go On", p.12]
    clean = re.sub(r'\[(?:Source:|[Book"\w\s\-]+,?\s*p\.\d+[^\]]*)\]', '', text, flags=re.IGNORECASE)
    clean = re.sub(r'\[Book-\d+,\s*p\.\d+[^\]]*\]', '', clean)
    # Strip markdown bold/italic/code/headers
    clean = re.sub(r'[\*\#\_`~]', '', clean)
    # Strip URLs
    clean = re.sub(r'https?://\S+', '', clean)
    # Normalize extra spaces and newlines
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean


def get_available_voices():
    """Return dictionary of supported voice presets."""
    return {
        "en-US-AriaNeural": "Aria (Expressive Female)",
        "en-US-ChristopherNeural": "Christopher (Deep Male Narrator)",
        "en-US-GuyNeural": "Guy (Natural Male)",
        "en-IN-PrabhatNeural": "Prabhat (Indian Accent Male)",
        "gu-IN-NiranjanNeural": "Niranjan (Native Gujarati Accent)",
        "gu-IN-DhwaniNeural": "Dhwani (Gujarati Female Accent)"
    }


def generate_tts_audio(text: str, voice: str = None, provider: str = None) -> tuple[bytes | None, str]:
    """
    Synthesize audio bytes using requested provider.
    Returns (audio_bytes, mime_type).
    If provider fails or is 'browser', returns (None, 'browser').
    """
    clean_text = clean_text_for_speech(text)
    if not clean_text:
        return None, "empty"

    selected_voice = voice or DEFAULT_VOICE
    selected_provider = (provider or DEFAULT_PROVIDER).lower().strip()

    if selected_provider == "browser":
        return None, "browser"

    if selected_provider == "edge-tts":
        try:
            import edge_tts

            async def _edge_synth():
                communicate = edge_tts.Communicate(clean_text, selected_voice)
                audio_data = bytearray()
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_data.extend(chunk["data"])
                return bytes(audio_data)

            # edge-tts is asyncio-based, but this function is also called from
            # inside ALREADY-RUNNING event loops (FastAPI endpoints), where
            # neither asyncio.run() nor loop.run_until_complete() is legal in
            # the calling thread. Running the coroutine in a dedicated thread
            # with its OWN event loop works from every calling context.
            import threading
            holder: dict = {}

            def _synth_in_thread():
                try:
                    holder["audio"] = asyncio.run(_edge_synth())
                except BaseException as exc:  # network/SSL errors etc.
                    holder["error"] = exc

            synth_thread = threading.Thread(target=_synth_in_thread, daemon=True)
            synth_thread.start()
            synth_thread.join(timeout=60)
            if "error" in holder:
                raise holder["error"]
            audio_bytes = holder.get("audio", b"")

            if audio_bytes and len(audio_bytes) > 0:
                return audio_bytes, "audio/mpeg"
        except ImportError:
            logger.warning("edge-tts library not installed. Falling back to browser TTS.")
        except Exception as e:
            logger.error(f"edge-tts error: {e}")

    elif selected_provider in ("piper", "local"):
        piper_cmd = os.environ.get("PIPER_CMD")
        if piper_cmd and Path(piper_cmd).exists():
            import subprocess
            try:
                res = subprocess.run([piper_cmd, "--output_stdout"], input=clean_text.encode("utf-8"), capture_output=True, timeout=15)
                if res.returncode == 0 and res.stdout:
                    return res.stdout, "audio/wav"
            except Exception as e:
                logger.error(f"Piper local TTS error: {e}")

    # Final fallback signal
    return None, "browser"
