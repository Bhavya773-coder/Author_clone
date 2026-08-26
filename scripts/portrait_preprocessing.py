#!/usr/bin/env python3
"""
Portrait Preprocessing for the Real-Time Avatar
===============================================
Turns a user-uploaded photograph into an avatar-ready source image:

* Validates the file with Pillow (never trusts the extension).
* Enforces format (JPG/JPEG/PNG/WebP) and file-size limits.
* Corrects EXIF orientation and converts to RGB.
* Detects the largest valid frontal face (OpenCV Haar cascade).
* Crops to head & shoulders, preserving a full hat where possible,
  with padding above the head and without cutting off the chin.
* Produces a normalized portrait-orientation avatar source.
* Raises PortraitError with a clear message when no usable face is found.
* Never beautifies, re-renders or otherwise alters the person's identity.

Also exposes eye coordinates (when detectable) for the idle blink animator.
"""

import io
import os
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger("PortraitPreprocessing")

MAX_UPLOAD_BYTES = int(os.environ.get("AVATAR_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))  # 10 MB
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
OUTPUT_SIZE = (512, 640)          # (width, height) portrait aspect for the avatar stage
MAX_INPUT_DIMENSION = 4096        # reject absurdly large images early


class PortraitError(Exception):
    """Raised for any user-facing portrait validation/detection failure."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_and_open(raw_bytes: bytes) -> Image.Image:
    """Validate upload bytes and return an RGB PIL image with EXIF orientation applied."""
    if not raw_bytes:
        raise PortraitError("Empty file uploaded.")
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise PortraitError(
            f"File too large ({len(raw_bytes) / 1e6:.1f} MB). Limit is {MAX_UPLOAD_BYTES // 1024 // 1024} MB."
        )
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img.load()
    except Exception:
        raise PortraitError("File is not a valid image. Upload a JPG, JPEG, PNG or WebP photo.")

    fmt = (img.format or "").upper()
    if fmt not in ALLOWED_FORMATS:
        raise PortraitError(f"Unsupported image format '{fmt or 'unknown'}'. Use JPG, JPEG, PNG or WebP.")
    if max(img.size) > MAX_INPUT_DIMENSION:
        raise PortraitError(f"Image dimensions too large (max {MAX_INPUT_DIMENSION}px per side).")

    img = ImageOps.exif_transpose(img)   # correct EXIF orientation
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


# ---------------------------------------------------------------------------
# Face & eye detection
# ---------------------------------------------------------------------------
_CASCADE_DIR = None


def _load_cascades():
    """Load Haar cascades bundled with opencv. Returns (face_cascade, eye_cascade|None)."""
    global _CASCADE_DIR
    try:
        import cv2
    except ImportError:
        raise PortraitError("opencv-python-headless is not installed; cannot detect faces.")

    if _CASCADE_DIR is None:
        _CASCADE_DIR = Path(cv2.data.haarcascades)

    face_path = _CASCADE_DIR / "haarcascade_frontalface_default.xml"
    eye_path = _CASCADE_DIR / "haarcascade_eye.xml"
    face_cascade = cv2.CascadeClassifier(str(face_path))
    if face_cascade.empty():
        raise PortraitError("Face detector model files are missing (corrupt OpenCV install).")
    eye_cascade = cv2.CascadeClassifier(str(eye_path))
    if eye_cascade.empty():
        eye_cascade = None
    return face_cascade, eye_cascade


def detect_face(img: Image.Image) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
    """
    Detect the largest valid frontal face.
    Returns ((x, y, w, h), gray_numpy_image). Raises PortraitError if none found.
    """
    import cv2
    face_cascade, _ = _load_cascades()
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Detect at original scale; fall back to an upscaled pass for small faces.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        up = cv2.resize(gray, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
        faces_up = face_cascade.detectMultiScale(up, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
        if len(faces_up) > 0:
            faces = (faces_up / 1.6).astype(int)
    if len(faces) == 0:
        raise PortraitError(
            "No usable frontal face detected in the photo. "
            "Use a well-lit, front-facing portrait where the face is clearly visible."
        )
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return (int(x), int(y), int(w), int(h)), gray


def detect_eyes(img: Image.Image, face_box: Tuple[int, int, int, int]) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Detect eye centres (in full-image coordinates). Used for idle blink animation.
    Returns ((lx, ly), (rx, ry)) or None.
    """
    import cv2
    try:
        _, eye_cascade = _load_cascades()
    except PortraitError:
        return None
    if eye_cascade is None:
        return None
    x, y, w, h = face_box
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    roi = gray[y:y + int(h * 0.62), x:x + w]
    eyes = eye_cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=4, minSize=(12, 12))
    if len(eyes) < 2:
        return None
    eyes = sorted(eyes, key=lambda e: e[0])[:2]
    (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) = eyes[0], eyes[1]
    left = (x + ex1 + ew1 // 2, y + ey1 + eh1 // 2)
    right = (x + ex2 + ew2 // 2, y + ey2 + eh2 // 2)
    return left, right


# ---------------------------------------------------------------------------
# Avatar crop
# ---------------------------------------------------------------------------
def compute_avatar_crop(
    img_size: Tuple[int, int],
    face_box: Tuple[int, int, int, int],
    manual_offset: Optional[Tuple[float, float]] = None,
    manual_zoom: Optional[float] = None,
) -> Tuple[int, int, int, int]:
    """
    Compute a head-and-shoulders crop box (x0, y0, x1, y1) that:
      * includes the full face,
      * extends ~1 face-height above the face top to preserve a hat,
      * extends below the chin so the chin is never clipped,
      * targets the OUTPUT_SIZE aspect ratio (4:5 portrait),
      * stays inside the image (padded with edge colour later if needed).

    manual_offset: (dx, dy) in [-1, 1] relative shifts for manual adjustment.
    manual_zoom: >1 zooms in, <1 zooms out.
    """
    img_w, img_h = img_size
    fx, fy, fw, fh = face_box
    aspect = OUTPUT_SIZE[0] / OUTPUT_SIZE[1]  # 0.8

    # Vertical span: hat space above (1.0 * fh), face, neck/shoulders below (1.1 * fh)
    top_pad = 1.0 * fh
    bottom_pad = 1.1 * fh
    crop_h = fh + top_pad + bottom_pad
    crop_w = crop_h * aspect

    cx = fx + fw / 2
    cy = fy - top_pad + crop_h / 2  # crop top anchored above the hat

    if manual_zoom and manual_zoom > 0:
        crop_w /= manual_zoom
        crop_h /= manual_zoom
    if manual_offset:
        cx += manual_offset[0] * crop_w * 0.25
        cy += manual_offset[1] * crop_h * 0.25

    x0 = cx - crop_w / 2
    y0 = cy - crop_h / 2

    # Soft-clamp inside the image while keeping the aspect ratio
    if crop_w > img_w * 1.35 or crop_h > img_h * 1.35:
        scale = min(img_w * 1.35 / crop_w, img_h * 1.35 / crop_h)
        crop_w *= scale
        crop_h *= scale
        x0 = cx - crop_w / 2
        y0 = cy - crop_h / 2

    x0 = max(min(x0, img_w - crop_w * 0.5), -crop_w * 0.35)
    y0 = max(min(y0, img_h - crop_h * 0.5), -crop_h * 0.35)

    return (int(round(x0)), int(round(y0)), int(round(x0 + crop_w)), int(round(y0 + crop_h)))


def apply_crop(img: Image.Image, box: Tuple[int, int, int, int]) -> Image.Image:
    """Crop with edge-colour padding if the box extends beyond the image, then normalize size."""
    x0, y0, x1, y1 = box
    img_w, img_h = img.size
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x1 - img_w), max(0, y1 - img_h)

    crop = img.crop((max(0, x0), max(0, y0), min(img_w, x1), min(img_h, y1)))
    if pad_l or pad_t or pad_r or pad_b:
        # Pad with the average edge colour (keeps a black-hat/black-bg portrait black)
        arr = np.array(crop)
        edge = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]]).mean(axis=0).astype(np.uint8)
        crop = ImageOps.expand(crop, border=(pad_l, pad_t, pad_r, pad_b), fill=tuple(edge.tolist()))
    return crop.resize(OUTPUT_SIZE, Image.LANCZOS)


def preprocess_portrait(
    raw_bytes: bytes,
    manual_offset: Optional[Tuple[float, float]] = None,
    manual_zoom: Optional[float] = None,
) -> dict:
    """
    Full pipeline: bytes -> validated, cropped, normalized avatar source.
    Returns dict with keys: image (PIL RGB), face_box, crop_box, eyes, warnings.
    """
    img = validate_and_open(raw_bytes)
    face_box, _ = detect_face(img)
    eyes = detect_eyes(img, face_box)
    box = compute_avatar_crop(img.size, face_box, manual_offset, manual_zoom)
    processed = apply_crop(img, box)

    # Map eye centres from source-image coordinates into the processed
    # 512x640 crop, so downstream animators (idle loop, speech blink layer)
    # can use them directly on the avatar frames.
    eyes_processed = None
    if eyes:
        x0, y0, x1, y1 = box
        sx = OUTPUT_SIZE[0] / (x1 - x0)
        sy = OUTPUT_SIZE[1] / (y1 - y0)
        eyes_processed = tuple(
            (int(round((ex - x0) * sx)), int(round((ey - y0) * sy))) for ex, ey in eyes
        )

    warnings = []
    fx, fy, fw, fh = face_box
    if fw < img.size[0] * 0.12:
        warnings.append("Face is relatively small in the frame; crop may include extra background.")
    if eyes is None:
        warnings.append("Eyes could not be located; idle blinking will use an estimated position.")

    return {
        "image": processed,
        "face_box": face_box,
        "crop_box": box,
        "eyes": eyes_processed,       # coordinates in the processed 512x640 crop
        "eyes_source": eyes,          # coordinates in the original upload
        "warnings": warnings,
        "source_size": img.size,
    }
