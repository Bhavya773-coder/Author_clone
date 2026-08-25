#!/usr/bin/env python3
"""
Automated tests for the real-time avatar pipeline.
====================================================
Run WITHOUT a GPU (uses AVATAR_ENGINE_MODE=dryrun and stubbed TTS):

    cd <repo root>
    AVATAR_ENGINE_MODE=dryrun python -X utf8 scripts/test_realtime_avatar.py

Covers:
1. Portrait validation & preprocessing (bad files rejected, crop correctness).
2. Single-generation TTS cache (TTS runs exactly once per answer).
3. FFmpeg audio decoding (16 kHz f32 + 48 kHz s16).
4. Idle loop generation.
5. Full HTTP + WebRTC integration: session, offer, idle stream, speak
   (synchronized A/V), stop latency, replay cache hit, session cleanup.
"""

import os
import sys
import io
import json
import time
import wave
import queue
import asyncio
import threading
import unittest
import urllib.request
from pathlib import Path

os.environ.setdefault("AVATAR_ENGINE_MODE", "dryrun")
os.environ.setdefault("REALTIME_AVATAR_PORT", "8019")
os.environ.setdefault("AVATAR_PORTRAIT_DIR", "/tmp/test_avatar_portraits")

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(Path(__file__).parent.resolve()))

import numpy as np

BASE = f"http://localhost:{os.environ['REALTIME_AVATAR_PORT']}"
PORTRAIT = PROJECT_ROOT / "web" / "avatar_character.jpg"


def make_wav_bytes(seconds=2.0, sr=44100) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        t = np.linspace(0, seconds, int(sr * seconds))
        wf.writeframes((np.sin(2 * np.pi * 220 * t) * 30000).astype(np.int16).tobytes())
    return buf.getvalue()


def http(path, method="GET", body=None, raw=False):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=90)
    return resp.read() if raw else json.loads(resp.read())


def http_upload(path, file_bytes, filename="test.jpg", params=""):
    boundary = "----testboundary"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"{filename}\"\r\nContent-Type: image/jpeg\r\n\r\n").encode() \
        + file_bytes + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(BASE + path + params, data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return json.loads(urllib.request.urlopen(req, timeout=90).read())


# ---------------------------------------------------------------------------
# 1. Portrait preprocessing unit tests
# ---------------------------------------------------------------------------
class TestPortraitPreprocessing(unittest.TestCase):
    def setUp(self):
        import portrait_preprocessing as pp
        self.pp = pp
        self.raw = PORTRAIT.read_bytes()

    def test_valid_portrait_processed(self):
        result = self.pp.preprocess_portrait(self.raw)
        self.assertEqual(result["image"].size, self.pp.OUTPUT_SIZE)
        x, y, w, h = result["face_box"]
        self.assertGreater(w, 50)
        bx0, by0, bx1, by1 = result["crop_box"]
        self.assertLessEqual(by0, y)          # padding above head/hat
        self.assertGreaterEqual(by1, y + h)   # chin not cropped

    def test_garbage_rejected(self):
        with self.assertRaises(self.pp.PortraitError):
            self.pp.preprocess_portrait(b"this is not an image")

    def test_empty_rejected(self):
        with self.assertRaises(self.pp.PortraitError):
            self.pp.preprocess_portrait(b"")

    def test_oversize_rejected(self):
        with self.assertRaises(self.pp.PortraitError):
            self.pp.preprocess_portrait(b"\xff\xd8\xff" + b"0" * (self.pp.MAX_UPLOAD_BYTES + 1))

    def test_faceless_image_rejected(self):
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (400, 400), (10, 30, 80)).save(buf, "PNG")
        with self.assertRaises(self.pp.PortraitError):
            self.pp.preprocess_portrait(buf.getvalue())


# ---------------------------------------------------------------------------
# 2. Single-TTS cache
# ---------------------------------------------------------------------------
class TestSingleTTS(unittest.TestCase):
    def test_tts_generated_exactly_once(self):
        import audio_pipeline as ap
        ap.clear_audio_cache()
        calls = {"n": 0}

        def fake_tts(text, voice, provider=None):
            calls["n"] += 1
            return make_wav_bytes(0.5), "audio/wav"

        orig = ap.generate_tts_audio
        ap.generate_tts_audio = fake_tts
        try:
            ap.synthesize_speech_once("hello world", "voice1")
            ap.synthesize_speech_once("hello world", "voice1")
            ap.synthesize_speech_once("hello world", "voice1")
            self.assertEqual(calls["n"], 1, "TTS must run exactly once for identical (text, voice)")
            ap.synthesize_speech_once("different text", "voice1")
            self.assertEqual(calls["n"], 2)
        finally:
            ap.generate_tts_audio = orig
            ap.clear_audio_cache()


# ---------------------------------------------------------------------------
# 3. Audio decoding
# ---------------------------------------------------------------------------
class TestAudioDecode(unittest.TestCase):
    def test_decode_formats(self):
        import audio_pipeline as ap
        wav = make_wav_bytes(1.0, sr=44100)
        pcm16, sr = ap.decode_to_musetalk_pcm(wav)
        self.assertEqual(sr, 16000)
        self.assertEqual(pcm16.dtype, np.float32)
        self.assertAlmostEqual(len(pcm16) / sr, 1.0, delta=0.05)
        pcm48, sr = ap.decode_to_webrtc_pcm(wav)
        self.assertEqual(sr, 48000)
        self.assertEqual(pcm48.dtype, np.int16)
        self.assertAlmostEqual(len(pcm48) / sr, 1.0, delta=0.05)


# ---------------------------------------------------------------------------
# 4. Idle loop
# ---------------------------------------------------------------------------
class TestIdleLoop(unittest.TestCase):
    def test_idle_loop_frames(self):
        import idle_animator
        from PIL import Image
        img = Image.open(PORTRAIT).resize((512, 640))
        loop = idle_animator.build_idle_loop(img, None, fps=25, seconds=4)
        self.assertEqual(len(loop), 100)
        self.assertEqual(loop[0].shape, (640, 512, 3))
        # loop must be seamless: last frame close to first
        self.assertLess(np.abs(loop[-1].astype(int) - loop[0].astype(int)).mean(), 2.0)
        # but not static
        self.assertGreater(np.abs(loop[25].astype(int) - loop[0].astype(int)).mean(), 0.5)


# ---------------------------------------------------------------------------
# 5. HTTP + WebRTC integration (dry-run engine, stubbed TTS)
# ---------------------------------------------------------------------------
class TestRealtimeServer(unittest.TestCase):
    server_thread = None

    @classmethod
    def setUpClass(cls):
        import audio_pipeline as ap
        ap.generate_tts_audio = lambda text, voice, provider=None: (make_wav_bytes(1.5), "audio/wav")

        import realtime_avatar_server as ras
        import uvicorn
        config = uvicorn.Config(ras.app, host="127.0.0.1",
                                port=int(os.environ["REALTIME_AVATAR_PORT"]), log_level="error")
        cls.uvicorn_server = uvicorn.Server(config)
        cls.server_thread = threading.Thread(target=cls.uvicorn_server.run, daemon=True)
        cls.server_thread.start()
        for _ in range(60):
            try:
                http("/api/avatar/health")
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("server did not start")

    @classmethod
    def tearDownClass(cls):
        if cls.uvicorn_server:
            cls.uvicorn_server.should_exit = True
            time.sleep(1)

    def test_01_health_and_capabilities(self):
        h = http("/api/avatar/health")
        self.assertTrue(h["engine"]["available"])
        c = http("/api/avatar/capabilities")
        self.assertTrue(c["webrtc"])
        self.assertEqual(c["fallback"], "css")

    def test_02_portrait_upload_and_preview(self):
        info = http_upload("/api/avatar/portrait", PORTRAIT.read_bytes())
        self.assertIn("portrait_id", info)
        self.assertTrue(info["preview_url"].startswith("/api/avatar/portrait/preview/"))
        preview = http(info["preview_url"], raw=True)
        self.assertGreater(len(preview), 10000)
        # manual crop adjustment must also work
        info2 = http_upload("/api/avatar/portrait", PORTRAIT.read_bytes(),
                            params="?offset_x=0.2&zoom=1.3")
        self.assertIn("portrait_id", info2)

    def test_03_bad_upload_rejected(self):
        try:
            http_upload("/api/avatar/portrait", b"garbage", filename="evil.jpg")
            self.fail("should have been rejected")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 422)

    def test_04_webrtc_speak_stop_flow(self):
        asyncio.run(self._webrtc_flow())

    async def _webrtc_flow(self):
        from aiortc import RTCPeerConnection, RTCSessionDescription

        sid = http("/api/avatar/session", method="POST", body={})["session_id"]
        pc = RTCPeerConnection()
        pc.addTransceiver("video", direction="recvonly")
        pc.addTransceiver("audio", direction="recvonly")
        frames = {"video": [], "audio": []}

        @pc.on("track")
        def on_track(track):
            async def consume():
                while True:
                    try:
                        f = await track.recv()
                    except Exception:
                        break
                    frames[track.kind].append((f.pts, time.time()))
            asyncio.ensure_future(consume())

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        ans = http(f"/api/avatar/session/{sid}/offer", method="POST",
                   body={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
        await pc.setRemoteDescription(RTCSessionDescription(sdp=ans["sdp"], type=ans["type"]))

        await asyncio.sleep(2)
        idle_v, idle_a = len(frames["video"]), len(frames["audio"])
        self.assertGreater(idle_v, 30, "idle video should stream ~25 fps")
        self.assertGreater(idle_a, 60, "idle audio should stream 20 ms packets")

        # speak
        v0, a0 = idle_v, idle_a
        resp = http(f"/api/avatar/session/{sid}/speak", method="POST",
                    body={"text": "Test utterance for synchronization.", "voice": "v"})
        self.assertEqual(resp["status"], "speaking")
        await asyncio.sleep(resp["speech_duration_s"] + 2)
        nv = len(frames["video"]) - v0
        na = len(frames["audio"]) - a0
        self.assertGreater(nv, 25, "speech video frames must flow")
        self.assertGreater(na, 50, "speech audio packets must flow")
        # A/V drift over the utterance
        v_span = frames["video"][-1][1] - frames["video"][v0][1]
        a_span = frames["audio"][-1][1] - frames["audio"][a0][1]
        self.assertLess(abs(v_span - a_span), 0.5, f"A/V drift too large: {abs(v_span-a_span)*1000:.0f} ms")

        # replay must be a cache hit (instant TTS)
        t0 = time.time()
        resp2 = http(f"/api/avatar/session/{sid}/speak", method="POST",
                     body={"text": "Test utterance for synchronization.", "voice": "v"})
        self.assertLess(time.time() - t0, 3.0, "replay should reuse cached TTS audio")
        self.assertTrue(resp2["audio_cached"])

        # stop latency
        http(f"/api/avatar/session/{sid}/speak", method="POST",
             body={"text": "Another sentence to interrupt.", "voice": "v"})
        await asyncio.sleep(0.3)
        t0 = time.time()
        http(f"/api/avatar/session/{sid}/stop", method="POST")
        self.assertLess(time.time() - t0, 0.5, "stop must respond within ~500 ms")

        # session delete
        http(f"/api/avatar/session/{sid}", method="DELETE")
        await pc.close()
        try:
            http(f"/api/avatar/session/{sid}/stats")
            self.fail("deleted session should be gone")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
