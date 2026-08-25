#!/usr/bin/env python3
"""
Standalone MuseTalk render test (offline, no browser, no WebRTC)
================================================================
Renders MuseTalk's raw output for a short utterance into an MP4 so you can
SEE exactly what the engine generates. This isolates the engine from the
streaming/frontend pipeline:

    # render from any audio file:
    python -X utf8 scripts/test_musetalk_render.py --audio sample.mp3

    # or synthesize speech first (edge-tts):
    python -X utf8 scripts/test_musetalk_render.py --text "Hello, this is a mouth movement test."

    # custom portrait / output:
    python -X utf8 scripts/test_musetalk_render.py --audio sample.mp3 \
        --portrait web/avatar_character.jpg --out render_test.mp4

Interpreting the result:
* Mouth moves in render_test.mp4  -> engine is fine; the bug is downstream
  (WebRTC/display path). Send us the terminal-3 log.
* Mouth frozen in render_test.mp4 -> the MuseTalk invocation itself is broken
  on this machine (API mismatch). The console prints the exact traceback
  and a motion metric to share.
"""

import os
import sys
import time
import queue
import argparse
import threading
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from audio_pipeline import decode_to_musetalk_pcm, synthesize_speech_once
from musetalk_worker import MuseTalkEngine, TARGET_FPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", help="Path to an audio file (mp3/wav/...)")
    ap.add_argument("--text", help="Text to synthesize with edge-tts instead")
    ap.add_argument("--voice", default="en-US-AriaNeural")
    ap.add_argument("--portrait", default=str(PROJECT_ROOT / "web" / "avatar_character.jpg"))
    ap.add_argument("--out", default="render_test.mp4")
    ap.add_argument("--max-seconds", type=float, default=8.0)
    args = ap.parse_args()

    if not args.audio and not args.text:
        ap.error("provide --audio or --text")

    # 1. Get audio bytes
    if args.audio:
        audio_bytes = Path(args.audio).read_bytes()
    else:
        audio_bytes, mime = synthesize_speech_once(args.text, args.voice)
        if not audio_bytes:
            print("TTS failed — use --audio with a local file instead.")
            sys.exit(2)

    # 2. Decode to MuseTalk PCM
    pcm, sr = decode_to_musetalk_pcm(audio_bytes)
    pcm = pcm[: int(sr * args.max_seconds)]
    print(f"Audio: {len(pcm)/sr:.1f}s @ {sr} Hz -> {int(np.ceil(len(pcm)/ (sr//TARGET_FPS)))} frames @ {TARGET_FPS} fps")

    # 3. Load engine ONCE and render
    from PIL import Image
    engine = MuseTalkEngine()
    print("Loading MuseTalk (one-time)...")
    engine.load()
    print(f"Loaded in {engine.load_time_s:.1f}s")

    portrait = Image.open(args.portrait).convert("RGB")
    engine.prepare_portrait(portrait, portrait_key="render-test")

    frame_q: "queue.Queue" = queue.Queue()
    cancel = threading.Event()
    stats: dict = {}
    t0 = time.time()
    engine.stream_frames(pcm, frame_q, cancel, stats)
    frames = {}
    while not frame_q.empty():
        item = frame_q.get()
        if item is not None:
            frames[item[0]] = item[1]

    print(f"Rendered {len(frames)} frames in {time.time()-t0:.1f}s")
    for k in ("first_frame_latency_s", "fps", "peak_vram_gb", "motion_mean", "job_error"):
        if k in stats:
            print(f"  {k}: {stats[k]}")

    if not frames:
        print("NO FRAMES PRODUCED — see error above.")
        sys.exit(3)

    # 4. Write MP4
    import cv2
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), TARGET_FPS, (w, h))
    for i in range(max(frames) + 1):
        f = frames.get(i, frames[max(k for k in frames if k <= i)])
        writer.write(f[:, :, ::-1])  # RGB -> BGR
    writer.release()
    print(f"Wrote {args.out} — open it and check whether the mouth moves.")


if __name__ == "__main__":
    main()
