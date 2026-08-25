#!/usr/bin/env python3
"""
Persistent MuseTalk 1.5 Worker
==============================
One process-wide worker that keeps MuseTalk 1.5 loaded in GPU memory and
turns 16 kHz mono PCM speech into lip-synced RGB frames in real time.

The inference path faithfully follows MuseTalk 1.5's own
`scripts/realtime_inference.py`:

    audio wav -> Audio2Feature.audio2feat -> feature2chunks(fps)
              -> datagen(chunks, latents_cycle, batch_size)
              -> pe(whisper_batch) -> unet -> vae.decode_latents
              -> resize 256x256 crop to the face bbox from
                 get_landmark_and_bbox() and blend it back into the
                 original portrait frame

Design guarantees:
* Models are loaded EXACTLY ONCE (never reloaded per answer).
* Portrait preparation is cached per portrait and recomputed only when the
  portrait changes.
* FP16 + torch.inference_mode everywhere supported.
* A bounded speech queue (AVATAR_MAX_QUEUE) and a single GPU lock ensure only
  one avatar-generation job uses the GPU at a time.
* GPU-unavailable / OOM errors surface cleanly via `worker.last_error` and
  the /api/avatar/health endpoint, so the frontend can fall back to the CSS
  avatar.
* Logs model-load time, per-job first-frame latency, achieved FPS, speech
  duration and peak VRAM.

If MuseTalk is not installed the worker reports itself unavailable; the
server then falls back to the CSS avatar. Set AVATAR_ENGINE_MODE=dryrun to
exercise the full WebRTC pipeline without a GPU (development shim only —
mouth motion is amplitude-driven, NOT phoneme-accurate).
"""

import os
import sys
import time
import wave
import queue
import logging
import tempfile
import threading
from pathlib import Path
from typing import Optional, List

import numpy as np
from PIL import Image

logger = logging.getLogger("MuseTalkWorker")

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

MUSETALK_PATH = os.environ.get("MUSETALK_PATH", str(PROJECT_ROOT / "MuseTalk"))
MUSETALK_MODEL_DIR = os.environ.get("MUSETALK_MODEL_DIR", str(Path(MUSETALK_PATH) / "models"))
AVATAR_DEVICE = os.environ.get("AVATAR_DEVICE", "cuda:0")
AVATAR_FP16 = os.environ.get("AVATAR_FP16", "1") == "1"
TARGET_FPS = int(os.environ.get("AVATAR_TARGET_FPS", "25"))
MAX_QUEUE = int(os.environ.get("AVATAR_MAX_QUEUE", "3"))
BATCH_SIZE = int(os.environ.get("AVATAR_BATCH_SIZE", "4"))
BBOX_SHIFT = int(os.environ.get("AVATAR_BBOX_SHIFT", "0"))
ENGINE_MODE = os.environ.get("AVATAR_ENGINE_MODE", "musetalk").lower().strip()

AUDIO_SAMPLE_RATE = 16000
SAMPLES_PER_FRAME = AUDIO_SAMPLE_RATE // TARGET_FPS  # 640 samples/frame @ 25 fps


class GpuUnavailableError(RuntimeError):
    pass


def _write_temp_wav(pcm_f32: np.ndarray, sample_rate: int) -> str:
    """Write float32 PCM to a REAL RIFF/WAV temp file (never raw bytes in .wav)."""
    pcm_s16 = (np.clip(pcm_f32, -1.0, 1.0) * 32767).astype(np.int16)
    fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with wave.open(tmp_wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_s16.tobytes())
    return tmp_wav


# ---------------------------------------------------------------------------
# Engine adapters
# ---------------------------------------------------------------------------
class MuseTalkEngine:
    """Adapter around the real MuseTalk 1.5 realtime inference API."""

    name = "MuseTalk15"

    def __init__(self):
        self.device = None
        self.vae = None
        self.unet = None
        self.pe = None
        self.audio_processor = None
        self.timesteps = None
        self.weight_dtype = None
        self.load_time_s: Optional[float] = None
        # cached per-portrait avatar materials
        self._portrait_key: Optional[str] = None
        self._base_frame_bgr: Optional[np.ndarray] = None  # full portrait frame (BGR)
        self._face_bbox = None                             # (x1, y1, x2, y2)
        self._latents_cycle: Optional[list] = None

    # -- loading ----------------------------------------------------------
    def load(self) -> None:
        if not Path(MUSETALK_PATH).exists():
            raise GpuUnavailableError(
                f"MuseTalk repository not found at MUSETALK_PATH={MUSETALK_PATH}. "
                "Clone https://github.com/TMElyralab/MuseTalk and set MUSETALK_PATH."
            )
        try:
            import torch
        except ImportError as e:
            raise GpuUnavailableError(f"PyTorch is not installed: {e}")
        if not torch.cuda.is_available() and AVATAR_DEVICE.startswith("cuda"):
            raise GpuUnavailableError("CUDA is not available to PyTorch on this machine.")

        t0 = time.time()
        if str(Path(MUSETALK_PATH).resolve()) not in sys.path:
            sys.path.insert(0, str(Path(MUSETALK_PATH).resolve()))

        try:
            from musetalk.utils.utils import load_all_model  # MuseTalk 1.5 API
        except ImportError as e:
            raise GpuUnavailableError(
                f"Could not import MuseTalk modules from {MUSETALK_PATH}: {e}. "
                "Ensure MuseTalk 1.5 requirements are installed."
            )

        self.device = torch.device(AVATAR_DEVICE)
        self.weight_dtype = torch.float16 if AVATAR_FP16 else torch.float32

        logger.info("Loading MuseTalk 1.5 checkpoints from %s ...", MUSETALK_MODEL_DIR)
        # load_all_model returns (audio_processor, vae, unet, pe) — same call
        # signature as MuseTalk's own inference.py / realtime_inference.py
        audio_processor, vae, unet, pe = load_all_model(
            unet_model_path=str(Path(MUSETALK_MODEL_DIR) / "musetalkV15" / "unet.pth"),
            vae_type="sd-vae",
            unet_config=str(Path(MUSETALK_MODEL_DIR) / "musetalkV15" / "musetalk.json"),
            device=self.device,
        )
        self.audio_processor = audio_processor
        self.vae = vae
        self.unet = unet
        self.pe = pe

        self.timesteps = torch.tensor([0], device=self.device)
        if AVATAR_FP16:
            self.pe = self.pe.half()
            self.vae.model = self.vae.model.half()
            self.unet.model = self.unet.model.half()
        for m in (self.pe, self.vae.model, self.unet.model):
            m.requires_grad_(False)

        self.load_time_s = time.time() - t0
        vram = torch.cuda.max_memory_allocated(self.device) / 1e9 if self.device.type == "cuda" else 0.0
        logger.info("MuseTalk loaded in %.1fs (peak VRAM so far: %.2f GB)", self.load_time_s, vram)

    # -- portrait preparation (cached) ------------------------------------
    def prepare_portrait(self, portrait: Image.Image, portrait_key: str) -> None:
        """
        Extract the face bbox + VAE latents for the avatar source, exactly like
        MuseTalk realtime_inference.py's avatar preparation. Runs only when
        the portrait changes.
        """
        if self._portrait_key == portrait_key and self._latents_cycle is not None:
            return
        from musetalk.utils.preprocessing import get_landmark_and_bbox, coord_placeholder

        t0 = time.time()
        tmp_dir = PROJECT_ROOT / "web" / "avatar_cache"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_img = tmp_dir / "_musetalk_source.png"
        portrait.save(tmp_img)

        coord_list, frame_list = get_landmark_and_bbox([str(tmp_img)], BBOX_SHIFT)
        if not coord_list or coord_list[0] is coord_placeholder or (
            isinstance(coord_list[0], str) and coord_list[0] == coord_placeholder
        ):
            raise GpuUnavailableError(
                "MuseTalk could not detect a face in the prepared portrait. "
                "Re-upload a clearer front-facing portrait."
            )

        bbox = [int(v) for v in coord_list[0]]
        frame_bgr = frame_list[0]                      # MuseTalk frames are BGR (cv2)
        x1, y1, x2, y2 = bbox
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            raise GpuUnavailableError(
                f"Invalid face bbox {bbox} for frame shape {frame_bgr.shape}."
            )

        import cv2
        resized = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        with torch_inference():
            latents = self.vae.get_latents_for_unet(resized)

        self._base_frame_bgr = frame_bgr.copy()
        self._face_bbox = bbox
        self._latents_cycle = [latents]
        self._portrait_key = portrait_key
        logger.info("Portrait prepared for MuseTalk in %.2fs (bbox=%s, frame=%s)",
                    time.time() - t0, bbox, frame_bgr.shape)

    # -- blending ----------------------------------------------------------
    def _blend(self, res_frame: np.ndarray) -> np.ndarray:
        """
        Paste a generated 256x256 face crop back into the portrait frame at the
        detected face bbox. Uses MuseTalk's masked blending when available,
        otherwise a direct paste (matches offline inference.py behaviour).
        """
        x1, y1, x2, y2 = self._face_bbox
        import cv2
        res = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1),
                         interpolation=cv2.INTER_LANCZOS4)
        out = self._base_frame_bgr.copy()
        try:
            from musetalk.utils.blending import get_image
            out = get_image(out, res, [x1, y1, x2, y2], mode="jaw")
        except Exception:
            out[y1:y2, x1:x2] = res
        return out

    # -- streaming inference ----------------------------------------------
    def stream_frames(self, pcm_f32: np.ndarray, frame_queue: "queue.Queue",
                      cancel_event: threading.Event, stats: dict) -> None:
        """
        Consume 16 kHz float32 PCM, emit RGB uint8 frames at TARGET_FPS into
        frame_queue as (frame_index, np.ndarray). Frame index 0 corresponds to
        PCM sample 0, so the WebRTC layer aligns A/V on one timeline.
        """
        import torch
        from musetalk.utils.utils import datagen

        tmp_wav = _write_temp_wav(pcm_f32, AUDIO_SAMPLE_RATE)
        try:
            whisper_input_features, _librosa_length = self.audio_processor.audio2feat(tmp_wav)
        finally:
            try:
                os.unlink(tmp_wav)
            except OSError:
                pass

        whisper_chunks = self.audio_processor.feature2chunks(
            feature_array=whisper_input_features, fps=TARGET_FPS
        )
        n_chunks = len(whisper_chunks)
        n_audio_frames = int(np.ceil(len(pcm_f32) / SAMPLES_PER_FRAME))
        stats["total_frames"] = n_audio_frames
        if n_chunks == 0:
            logger.warning("MuseTalk produced 0 whisper chunks — emitting portrait only")
            base_rgb = self._base_frame_bgr[:, :, ::-1].copy()
            for i in range(n_audio_frames):
                if cancel_event.is_set():
                    break
                frame_queue.put((i, base_rgb))
            return

        t_start = time.time()
        first_frame_logged = False
        frame_idx = 0
        weight_dtype = self.weight_dtype
        device = self.device

        gen = datagen(whisper_chunks, self._latents_cycle, batch_size=BATCH_SIZE)
        with torch.inference_mode():
            for whisper_batch, latent_batch in gen:
                if cancel_event.is_set():
                    stats["cancelled_at_frame"] = frame_idx
                    break
                audio_feature_batch = torch.from_numpy(whisper_batch).to(
                    device=device, dtype=weight_dtype)
                audio_feature_batch = self.pe(audio_feature_batch)
                latent_batch = latent_batch.to(device=device, dtype=weight_dtype)

                pred_latents = self.unet.model(
                    latent_batch, self.timesteps,
                    encoder_hidden_states=audio_feature_batch
                ).sample
                recon = self.vae.decode_latents(pred_latents)  # uint8 BGR (B,256,256,3)

                for res_frame in recon:
                    if cancel_event.is_set():
                        break
                    out_bgr = self._blend(res_frame)
                    out_rgb = out_bgr[:, :, ::-1].copy()
                    frame_queue.put((frame_idx, out_rgb))
                    frame_idx += 1
                    if not first_frame_logged:
                        stats["first_frame_latency_s"] = time.time() - t_start
                        first_frame_logged = True

        # If the audio has more frames than whisper chunks (tail padding),
        # hold the last generated frame for the remainder.
        if frame_idx < n_audio_frames and not cancel_event.is_set():
            last_rgb = self._base_frame_bgr[:, :, ::-1].copy() if frame_idx == 0 else out_rgb
            while frame_idx < n_audio_frames:
                if cancel_event.is_set():
                    break
                frame_queue.put((frame_idx, last_rgb))
                frame_idx += 1

        stats["render_time_s"] = time.time() - t_start
        if stats["render_time_s"] > 0:
            stats["fps"] = frame_idx / stats["render_time_s"]
        stats["frames_rendered"] = frame_idx
        if device.type == "cuda":
            stats["peak_vram_gb"] = torch.cuda.max_memory_allocated(device) / 1e9


def torch_inference():
    import torch
    return torch.inference_mode()


class DryRunEngine:
    """
    Development shim (AVATAR_ENGINE_MODE=dryrun): amplitude-driven mouth
    animation with NO MuseTalk dependency, used ONLY to test the streaming,
    synchronisation and fallback plumbing without a GPU. NOT phoneme-accurate
    and never enabled by default.
    """

    name = "DryRunEngine"

    def __init__(self):
        self.load_time_s = 0.0
        self._portrait_key = None
        self._base: Optional[np.ndarray] = None

    def load(self) -> None:
        logger.warning("DryRunEngine active — amplitude lip-sync shim, NOT MuseTalk.")

    def prepare_portrait(self, portrait: Image.Image, portrait_key: str) -> None:
        if self._portrait_key != portrait_key:
            self._base = np.array(portrait)
            self._portrait_key = portrait_key

    def stream_frames(self, pcm_f32: np.ndarray, frame_queue: "queue.Queue",
                      cancel_event: threading.Event, stats: dict) -> None:
        import cv2
        base = self._base
        h, w = base.shape[:2]
        mouth_y, mouth_x = int(h * 0.62), int(w * 0.5)
        n_frames = int(np.ceil(len(pcm_f32) / SAMPLES_PER_FRAME))
        stats["total_frames"] = n_frames
        t0 = time.time()
        first = False
        for i in range(n_frames):
            if cancel_event.is_set():
                stats["cancelled_at_frame"] = i
                break
            seg = pcm_f32[i * SAMPLES_PER_FRAME:(i + 1) * SAMPLES_PER_FRAME]
            amp = float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0
            open_px = int(min(1.0, amp * 8.0) * h * 0.05)
            frame = base.copy()
            if open_px > 1:
                y0, y1 = mouth_y, min(h, mouth_y + open_px)
                x0, x1 = max(0, mouth_x - int(w * 0.07)), min(w, mouth_x + int(w * 0.07))
                frame[y0:y1, x0:x1] = cv2.ellipse(
                    frame[y0:y1, x0:x1], ((x1 - x0) // 2, (y1 - y0) // 2),
                    ((x1 - x0) // 2 - 1, max(1, (y1 - y0) // 2 - 1)), 0, 0, 360, (35, 15, 20), -1)
            frame_queue.put((i, frame))
            if not first:
                stats["first_frame_latency_s"] = time.time() - t0
                first = True
        stats["render_time_s"] = time.time() - t0
        stats["fps"] = stats.get("cancelled_at_frame", n_frames) / max(1e-6, stats["render_time_s"])


# ---------------------------------------------------------------------------
# Persistent worker
# ---------------------------------------------------------------------------
class MuseTalkWorker:
    """Owns the engine, the bounded job queue and the single-GPU lock."""

    def __init__(self):
        self.engine = None
        self.available = False
        self.last_error: Optional[str] = None
        self.gpu_lock = threading.Lock()
        self.job_queue: "queue.Queue" = queue.Queue(maxsize=MAX_QUEUE)
        self._thread: Optional[threading.Thread] = None
        self._started = False

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        if self._started:
            return
        self._started = True
        try:
            engine = DryRunEngine() if ENGINE_MODE == "dryrun" else MuseTalkEngine()
            engine.load()
            self.engine = engine
            self.available = True
            logger.info("Avatar engine '%s' ready (mode=%s, device=%s, fp16=%s)",
                        engine.name, ENGINE_MODE, AVATAR_DEVICE, AVATAR_FP16)
        except GpuUnavailableError as e:
            self.last_error = str(e)
            self.available = False
            logger.error("Avatar engine unavailable: %s", e)
        except Exception as e:  # OOM etc.
            self.last_error = f"{type(e).__name__}: {e}"
            self.available = False
            logger.exception("Avatar engine failed to start: %s", e)

        self._thread = threading.Thread(target=self._job_loop, daemon=True, name="musetalk-worker")
        self._thread.start()

    def _job_loop(self) -> None:
        while True:
            item = self.job_queue.get()
            if item is None:
                return
            fn = item
            try:
                fn()
            except Exception as e:
                logger.exception("Avatar job failed: %s", e)
            finally:
                self.job_queue.task_done()

    # -- API used by the session layer ------------------------------------
    def submit_speak(self, portrait: Image.Image, portrait_key: str,
                     pcm_f32: np.ndarray, frame_queue: "queue.Queue",
                     cancel_event: threading.Event, stats: dict) -> bool:
        """
        Enqueue a speech job. Returns False immediately if the bounded queue is
        full (caller must prevent overlapping speech anyway).
        """
        if not self.available:
            return False

        def _run():
            with self.gpu_lock:  # only one GPU job at a time
                t0 = time.time()
                self.engine.prepare_portrait(portrait, portrait_key)
                stats["portrait_prepare_s"] = time.time() - t0
                self.engine.stream_frames(pcm_f32, frame_queue, cancel_event, stats)
            frame_queue.put(None)  # sentinel: speech finished
            logger.info(
                "Speech job done: first_frame=%.2fs fps=%.1f vram=%.2fGB frames=%s",
                stats.get("first_frame_latency_s", -1), stats.get("fps", -1),
                stats.get("peak_vram_gb", 0), stats.get("cancelled_at_frame", stats.get("total_frames")),
            )

        try:
            self.job_queue.put_nowait(_run)
            return True
        except queue.Full:
            logger.warning("Speech queue full (max=%d) — rejecting job", MAX_QUEUE)
            return False

    def status(self) -> dict:
        return {
            "available": self.available,
            "engine": self.engine.name if self.engine else None,
            "mode": ENGINE_MODE,
            "device": AVATAR_DEVICE,
            "fp16": AVATAR_FP16,
            "target_fps": TARGET_FPS,
            "batch_size": BATCH_SIZE,
            "bbox_shift": BBOX_SHIFT,
            "max_queue": MAX_QUEUE,
            "queue_depth": self.job_queue.qsize(),
            "model_load_time_s": self.engine.load_time_s if self.engine else None,
            "last_error": self.last_error,
        }


# Module-level singleton: the one persistent worker.
WORKER = MuseTalkWorker()
