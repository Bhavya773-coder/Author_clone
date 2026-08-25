#!/usr/bin/env python3
"""
Idle Avatar Animator
====================
Generates a short, seamless idle loop for a portrait ONCE (CPU only, cached),
so the character looks subtly alive between answers without any continuous
generative inference:

* gentle breathing (sub-pixel vertical bob + 0.6% scale),
* natural blinking at irregular intervals (deterministic per loop),
* very small head sway.

Frames are plain RGB numpy arrays at AVATAR_TARGET_FPS and loop perfectly.
"""

import os
import hashlib
import logging
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger("IdleAnimator")

TARGET_FPS = int(os.environ.get("AVATAR_TARGET_FPS", "25"))
IDLE_LOOP_SECONDS = float(os.environ.get("AVATAR_IDLE_LOOP_SECONDS", "12"))


def _warp_frame(img: np.ndarray, dy: float, scale: float, sway_x: float) -> np.ndarray:
    """
    Sub-pixel affine transform (centre-anchored scale + translate) for
    breathing/sway. Float precision keeps the loop perfectly seamless —
    integer-pixel shifts would break the wrap.
    """
    pil = Image.fromarray(img)
    w, h = pil.size
    cx, cy = w / 2.0, h / 2.0
    s = max(scale, 1e-6)
    # output (x, y) samples input at ((x-cx)/s + cx - sway_x, (y-cy)/s + cy - dy)
    a, b, c = 1.0 / s, 0.0, cx - cx / s - sway_x
    d, e, f = 0.0, 1.0 / s, cy - cy / s - dy
    warped = pil.transform((w, h), Image.AFFINE, (a, b, c, d, e, f), resample=Image.BICUBIC)
    return np.array(warped)


def _apply_blink(img: np.ndarray, eye_y: int, eye_half_h: int, closedness: float,
                 eye_x_range: Tuple[int, int]) -> np.ndarray:
    """
    Vertical squash of the eye band to simulate an eyelid closing.
    closedness: 0.0 (open) .. 1.0 (closed).
    """
    if closedness <= 0.0:
        return img
    h, w = img.shape[:2]
    y0 = max(0, eye_y - eye_half_h)
    y1 = min(h, eye_y + eye_half_h)
    if y1 - y0 < 4:
        return img
    x0, x1 = max(0, eye_x_range[0]), min(w, eye_x_range[1])
    band = img[y0:y1, x0:x1].astype(np.float32)
    band_h = band.shape[0]
    # Compress rows toward the band centre and darken lid rows slightly
    centre = band_h / 2.0
    out = np.empty_like(band)
    for r in range(band_h):
        src = centre + (r - centre) / max(1e-6, (1.0 - 0.85 * closedness))
        src = min(max(src, 0), band_h - 1)
        r0 = int(src)
        r1 = min(r0 + 1, band_h - 1)
        frac = src - r0
        out[r] = band[r0] * (1 - frac) + band[r1] * frac
    # slight darkening of lid skin as it closes
    lid_mask = np.abs(np.arange(band_h) - centre) / centre
    out *= (1.0 - 0.12 * closedness * lid_mask[:, None, None])
    img = img.copy()
    img[y0:y1, x0:x1] = out.astype(np.uint8)
    return img


def _blink_envelope(phase: float) -> float:
    """Triangle-ish blink envelope: phase 0..1 -> closedness 0..1..0."""
    if phase < 0.35:
        return phase / 0.35
    if phase < 0.5:
        return 1.0
    return max(0.0, 1.0 - (phase - 0.5) / 0.5)


def build_idle_loop(
    portrait: Image.Image,
    eyes: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    fps: int = TARGET_FPS,
    seconds: float = IDLE_LOOP_SECONDS,
) -> List[np.ndarray]:
    """
    Build a seamless idle loop (list of RGB uint8 frames).
    `eyes` are coordinates in the ORIGINAL upload; we map them into the
    normalized crop by scaling relative positions is complex, so if eyes are
    unavailable we blink at 38% of frame height (typical eye line of a
    head-and-shoulders crop).
    """
    n_frames = max(int(fps * seconds), fps * 2)
    base = np.array(portrait)
    h, w = base.shape[:2]

    if eyes:
        # Approximate: eyes tuple was measured pre-crop; use relative Y of eye
        # line within the face region mapped onto the normalized output.
        eye_y = int(h * 0.38)
        ex = (eyes[0][0], eyes[1][0])
        span = max(20, abs(ex[1] - ex[0]) if ex[1] != ex[0] else int(w * 0.3))
        eye_x_range = (int(w / 2 - span * 0.75), int(w / 2 + span * 0.75))
    else:
        eye_y = int(h * 0.38)
        eye_x_range = (int(w * 0.28), int(w * 0.72))
    eye_half_h = max(6, int(h * 0.022))

    # Blink schedule: deterministic irregular blinks inside the loop
    rng = np.random.default_rng(42)
    blink_starts = []
    t = rng.uniform(1.0, 2.5)
    while t < seconds - 1.0:
        blink_starts.append(t)
        t += rng.uniform(2.0, 5.5)
    blink_duration = 0.28  # seconds

    frames: List[np.ndarray] = []
    for i in range(n_frames):
        t = i / fps
        # Seamless breathing: sine over the FULL loop so first/last match
        loop_phase = 2 * np.pi * (i / n_frames)
        breath = np.sin(loop_phase * 2)  # two breaths per loop
        dy = breath * (h * 0.004)
        scale = 1.0 + breath * 0.003
        sway = np.sin(loop_phase) * (w * 0.002)

        frame = _warp_frame(base, dy, scale, sway)

        for bs in blink_starts:
            if bs <= t < bs + blink_duration:
                closed = _blink_envelope((t - bs) / blink_duration)
                frame = _apply_blink(frame, eye_y, eye_half_h, closed, eye_x_range)
                break
        frames.append(frame)

    logger.info("Idle loop built: %d frames @ %d fps (%.1fs)", n_frames, fps, seconds)
    return frames


def idle_loop_cache_key(portrait_bytes: bytes, fps: int) -> str:
    return hashlib.sha256(portrait_bytes + str(fps).encode()).hexdigest()[:20]
