#!/usr/bin/env python3
"""
Persistent MuseTalk 1.5 Worker
==============================
One process-wide worker that keeps MuseTalk 1.5 loaded in GPU memory and
turns 16 kHz mono PCM speech into lip-synced RGB frames in real time.

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
import queue
import logging
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
ENGINE_MODE = os.environ.get("AVATAR_ENGINE_MODE", "musetalk").lower().strip()

AUDIO_SAMPLE_RATE = 16000
SAMPLES_PER_FRAME = AUDIO_SAMPLE_RATE // TARGET_FPS  # 640 samples/frame @ 25 fps


class GpuUnavailableError(RuntimeError):
    pass


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
        self.whisper = None
        self.audio_processor = None
        self.timesteps = None
        self.weight_dtype = None
        self.load_time_s: Optional[float] = None
        # cached per-portrait avatar materials
        self._portrait_key: Optional[str] = None
        self._frame_list: Optional[List[np.ndarray]] = None
        self._coord_list = None
        self._latents = None

    # -- loading ----------------------------------------------------------
    def load(self) -> None:
        if not Path(MUSETALK_PATH).exists():
            raise GpuUnavailableError(
                f"MuseTalk repository not found at MUSETALK_PATH={MUSETALK_PATH}. "
                "Clone https://github.com/TMElyralab/MuseTalk and set MUSETALK_PATH."
            )
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise GpuUnavailableError(f"PyTorch is not installed: {e}")
        import torch
        if not torch.cuda.is_available() and AVATAR_DEVICE.startswith("cuda"):
            raise GpuUnavailableError("CUDA is not available to PyTorch on this machine.")

        t0 = time.time()
        if str(Path(MUSETALK_PATH).resolve()) not in sys.path:
            sys.path.insert(0, str(Path(MUSETALK_PATH).resolve()))

        try:
            from musetalk.utils.utils import load_all_model  # MuseTalk 1.5 API
            from musetalk.whisper.audio2feature import Audio2Feature
            from transformers import WhisperModel
        except ImportError as e:
            raise GpuUnavailableError(
                f"Could not import MuseTalk modules from {MUSETALK_PATH}: {e}. "
                "Ensure MuseTalk 1.5 requirements are installed."
            )

        self.device = torch.device(AVATAR_DEVICE)
        self.weight_dtype = torch.float16 if AVATAR_FP16 else torch.float32

        logger.info("Loading MuseTalk 1.5 checkpoints from %s ...", MUSETALK_MODEL_DIR)
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

        whisper_dir = str(Path(MUSETALK_MODEL_DIR) / "whisper")
        self.whisper = WhisperModel.from_pretrained(whisper_dir)
        self.whisper = self.whisper.to(device=self.device, dtype=self.weight_dtype)
        self.whisper.requires_grad_(False)

        self.timesteps = torch.tensor([0], device=self.device)
        if AVATAR_FP16:
            self.pe = self.pe.half()
            self.vae.model = self.vae.model.half()
            self.unet.model = self.unet.model.half()
        for m in (self.pe, self.vae.model, self.unet.model, self.whisper):
            m.requires_grad_(False)

        self.load_time_s = time.time() - t0
        vram = torch.cuda.max_memory_allocated(self.device) / 1e9 if self.device.type == "cuda" else 0.0
        logger.info("MuseTalk loaded in %.1fs (peak VRAM so far: %.2f GB)", self.load_time_s, vram)

    # -- portrait preparation (cached) ------------------------------------
    def prepare_portrait(self, portrait: Image.Image, portrait_key: str) -> None:
        """Extract landmarks / latents for the avatar source. Runs only when the portrait changes."""
        if self._portrait_key == portrait_key and self._latents is not None:
            return
        import torch
        from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
        from musetalk.utils.blending import get_image_prepare_material

        t0 = time.time()
        tmp_dir = PROJECT_ROOT / "web" / "avatar_cache"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_img = tmp_dir / "_musetalk_source.png"
        portrait.save(tmp_img)

        bbox_shift = 0  # MuseTalk default; tune for half-body avatars if needed
        coord_list, frame_list = get_landmark_and_bbox([str(tmp_img)], bbox_shift)
        if coord_list[0] == coord_placeholder:
            raise GpuUnavailableError(
                "MuseTalk could not detect a face in the prepared portrait. "
                "Re-upload a clearer front-facing portrait."
            )
        input_latent_list = []
        for idx, frame in enumerate(frame_list):
            coord = coord_list[idx]
            img_rgb, crop_box = get_image_prepare_material(frame, coord) if callable(get_image_prepare_material) else (frame, None)
            # MuseTalk 1.5 latents from the VAE
            resized = np.array(Image.fromarray(frame).resize((256, 256), Image.LANCZOS))
            tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            tensor = (tensor * 2 - 1).to(device=self.device, dtype=self.weight_dtype)
            with torch.inference_mode():
                latents = self.vae.get_latents_for_unet(tensor) if hasattr(self.vae, "get_latents_for_unet") \
                    else self.vae.encode(tensor).latent_dist.mode()
            input_latent_list.append(latents)

        self._frame_list = frame_list
        self._coord_list = coord_list
        self._latents = input_latent_list
        self._portrait_key = portrait_key
        logger.info("Portrait prepared for MuseTalk in %.2fs (%d source frames)", time.time() - t0, len(frame_list))

    # -- streaming inference ----------------------------------------------
    def stream_frames(self, pcm_f32: np.ndarray, frame_queue: "queue.Queue",
                      cancel_event: threading.Event, stats: dict) -> None:
        """
        Consume 16 kHz float32 PCM, emit RGB uint8 frames at TARGET_FPS into
        frame_queue as (frame_index, np.ndarray). Frame index 0 corresponds to
        PCM sample 0, so the WebRTC layer can align A/V on one timeline.
        """
        import torch
        from musetalk.utils.audio_processor import Audio2Feature  # noqa: F401  (kept for API clarity)

        n_frames = int(np.ceil(len(pcm_f32) / SAMPLES_PER_FRAME))
        stats["total_frames"] = n_frames
        first_frame_logged = False
        t_start = time.time()

        # Process in rolling windows (like MuseTalk realtime_inference.py):
        # whisper consumes the whole utterance's features; we slice per frame.
        whisper_chunks = self.audio_processor.audio2feat_from_pcm(pcm_f32, AUDIO_SAMPLE_RATE) \
            if hasattr(self.audio_processor, "audio2feat_from_pcm") else None

        if whisper_chunks is None:
            # Standard MuseTalk path: audio_processor.audio2feat expects a file.
            # Write PCM to a temp wav (valid RIFF, never raw mp3-in-wav) and run the standard API.
            import wave, tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tmp_wav = tf.name
            try:
                pcm_s16 = (np.clip(pcm_f32, -1, 1) * 32767).astype(np.int16)
                with wave.open(tmp_wav, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(AUDIO_SAMPLE_RATE)
                    wf.writeframes(pcm_s16.tobytes())
                whisper_chunks = self.audio_processor.audio2feat(tmp_wav)
            finally:
                try:
                    os.unlink(tmp_wav)
                except OSError:
                    pass

        latent = self._latents[0]
        ref_frame = self._frame_list[0]

        for i in range(n_frames):
            if cancel_event.is_set():
                stats["cancelled_at_frame"] = i
                break
            with torch.inference_mode():
                chunk = whisper_chunks[min(i, len(whisper_chunks) - 1)]
                audio_feature = torch.from_numpy(chunk).to(device=self.device, dtype=self.weight_dtype)
                audio_feature = audio_feature.unsqueeze(0)
                if audio_feature.dim() == 3:  # (1, T, D) -> (1, 2T, 384) window like MuseTalk
                    pass
                audio_feature = self.pe(audio_feature) if audio_feature.dim() == 3 else audio_feature
                pred_latents = self.unet.model(latent, self.timesteps, encoder_hidden_states=audio_feature).sample
                recon = self.vae.decode_latents(pred_latents) if hasattr(self.vae, "decode_latents") \
                    else self.vae.decode(pred_latents).sample
            frame = (recon[0].detach().float().cpu().numpy().transpose(1, 2, 0) * 255)
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            # Composite generated mouth region back onto the prepared portrait frame
            out = ref_frame.copy()
            fh, fw = frame.shape[:2]
            oh, ow = out.shape[:2]
            y0 = max(0, (oh - fh) // 2)
            x0 = max(0, (ow - fw) // 2)
            sub = out[y0:y0 + fh, x0:x0 + fw]
            if sub.shape[:2] == frame.shape[:2]:
                out[y0:y0 + fh, x0:x0 + fw] = frame
            else:
                out = np.array(Image.fromarray(frame).resize((ow, oh), Image.BILINEAR))
            frame_queue.put((i, out))
            if not first_frame_logged:
                stats["first_frame_latency_s"] = time.time() - t_start
                first_frame_logged = True

        stats["render_time_s"] = time.time() - t_start
        if stats.get("render_time_s") and stats.get("total_frames"):
            stats["fps"] = (stats.get("cancelled_at_frame", n_frames)) / stats["render_time_s"]
        if self.device.type == "cuda":
            stats["peak_vram_gb"] = torch.cuda.max_memory_allocated(self.device) / 1e9


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
            "max_queue": MAX_QUEUE,
            "queue_depth": self.job_queue.qsize(),
            "model_load_time_s": self.engine.load_time_s if self.engine else None,
            "last_error": self.last_error,
        }


# Module-level singleton: the one persistent worker.
WORKER = MuseTalkWorker()
