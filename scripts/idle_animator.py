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


# Safety zoom so sub-pixel motion NEVER samples outside the source frame.
# Out-of-bounds samples come back black from PIL and paint dark "lines" along
# the frame edges — this margin (plus the clamps below) eliminates them.
_BASE_ZOOM = 1.03


def _warp_frame(img: np.ndarray, dy: float, scale: float, sway_x: float) -> np.ndarray:
    """
    Sub-pixel affine transform (centre-anchored scale + translate) for
    breathing/sway. Float precision keeps the loop perfectly seamless —
    integer-pixel shifts would break the wrap.

    The effective zoom is always >= _BASE_ZOOM and translations are clamped
    to the margin that zoom provides, so sampling stays inside the image.
    """
    pil = Image.fromarray(img)
    w, h = pil.size
    cx, cy = w / 2.0, h / 2.0
    s = max(scale, 1.0) * _BASE_ZOOM
    # Clamp translation to ~80% of the zoom margin so we never sample outside.
    mx = (s - 1.0) / 2.0 * w * 0.8
    my = (s - 1.0) / 2.0 * h * 0.8
    dy = float(min(max(dy, -my), my))
    sway_x = float(min(max(sway_x, -mx), mx))
    # output (x, y) samples input at ((x-cx)/s + cx - sway_x, (y-cy)/s + cy - dy)
    a, b, c = 1.0 / s, 0.0, cx - cx / s - sway_x
    d, e, f = 0.0, 1.0 / s, cy - cy / s - dy
    warped = pil.transform((w, h), Image.AFFINE, (a, b, c, d, e, f), resample=Image.BICUBIC)
    return np.array(warped)


def _apply_blink(img: np.ndarray, eye_y: int, eye_half_h: int, closedness: float,
                 eye_x_range: Tuple[int, int]) -> np.ndarray:
    """
    Feathered vertical squash of the eye region to simulate an eyelid closing.
    closedness: 0.0 (open) .. 1.0 (closed).

    The squashed band is blended back with a raised-cosine weight that falls
    to zero at the band edges, so there is no visible seam (the old hard-edged
    rectangle is what made blinks look like flat "2D" cut-outs). Darkening is
    gentle and follows the same weight, so it also fades out at the edges.
    """
    if closedness <= 0.0:
        return img
    h, w = img.shape[:2]
    pad = max(3, eye_half_h // 2)
    y0 = max(0, eye_y - eye_half_h - pad)
    y1 = min(h, eye_y + eye_half_h + pad)
    if y1 - y0 < 6:
        return img
    x0, x1 = max(0, eye_x_range[0]), min(w, eye_x_range[1])
    if x1 - x0 < 6:
        return img
    band = img[y0:y1, x0:x1].astype(np.float32)
    band_h = band.shape[0]
    centre = band_h / 2.0
    squash = max(1e-6, 1.0 - 0.85 * closedness)
    out = np.empty_like(band)
    for r in range(band_h):
        src = centre + (r - centre) / squash
        src = min(max(src, 0.0), band_h - 1.0)
        r0 = int(src)
        r1 = min(r0 + 1, band_h - 1)
        frac = src - r0
        out[r] = band[r0] * (1.0 - frac) + band[r1] * frac
    # Raised-cosine vertical weight: 0 at band edges -> 1 at the centre.
    rows = np.arange(band_h, dtype=np.float32)
    wy = np.sin(np.pi * (rows + 0.5) / band_h) ** 2
    wy3 = wy[:, None, None]
    # Very slight lid shading, feathered identically so it can't seam.
    out *= (1.0 - 0.05 * closedness * wy3)
    blended = band * (1.0 - wy3) + out * wy3
    img = img.copy()
    img[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return img


def _blink_envelope(phase: float) -> float:
    """Smooth blink curve: fast close, short hold, soft open (no kinks)."""
    phase = min(max(phase, 0.0), 1.0)
    return float(np.sin(np.pi * phase) ** 0.8)


# Public aliases — reused by the MuseTalk worker's speech animation layer so
# the whole frame stays alive (sway + blinking) while the avatar talks.
warp_frame = _warp_frame
apply_blink = _apply_blink
blink_envelope = _blink_envelope


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
