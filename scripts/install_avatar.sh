#!/usr/bin/env bash
# =============================================================================
# Real-Time Avatar Installation Script
# =============================================================================
# Installs everything the MuseTalk + WebRTC real-time avatar needs:
#   1. CUDA-compatible PyTorch (cu121 by default — override CUDA_TAG env)
#   2. Python dependencies (requirements-avatar.txt)
#   3. MuseTalk 1.5 repository + its own requirements
#   4. MuseTalk checkpoints (musetalkV15, sd-vae-ft-mse, whisper, dwpose, face-parse)
#   5. FFmpeg (system package)
#
# Usage:
#   bash scripts/install_avatar.sh
#   CUDA_TAG=cu124 bash scripts/install_avatar.sh
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CUDA_TAG="${CUDA_TAG:-cu121}"
MUSETALK_PATH="${MUSETALK_PATH:-$PROJECT_ROOT/MuseTalk}"
MUSETALK_MODEL_DIR="${MUSETALK_MODEL_DIR:-$MUSETALK_PATH/models}"

echo "======================================================================"
echo " Real-Time Avatar Installer"
echo "   Project:      $PROJECT_ROOT"
echo "   MuseTalk:     $MUSETALK_PATH"
echo "   Model dir:    $MUSETALK_MODEL_DIR"
echo "   CUDA wheels:  $CUDA_TAG"
echo "======================================================================"

# ---- 1. CUDA-compatible PyTorch -------------------------------------------
# NOTE: `pip install -r requirements.txt` does NOT install CUDA PyTorch.
echo ">>> [1/6] Installing CUDA-compatible PyTorch ($CUDA_TAG)..."
pip install torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA_TAG"

# ---- 2. Avatar service dependencies ---------------------------------------
echo ">>> [2/6] Installing avatar service dependencies..."
pip install -r requirements-avatar.txt

# ---- 3. FFmpeg -------------------------------------------------------------
echo ">>> [3/6] Installing FFmpeg..."
if ! command -v ffmpeg >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update && sudo apt-get install -y ffmpeg
  elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y ffmpeg
  else
    echo "!! Install FFmpeg manually and/or set FFMPEG_PATH."
  fi
else
  echo "   FFmpeg already present: $(ffmpeg -version | head -1)"
fi

# ---- 4. MuseTalk 1.5 repository -------------------------------------------
echo ">>> [4/6] Cloning MuseTalk 1.5..."
if [ ! -d "$MUSETALK_PATH" ]; then
  git clone https://github.com/TMElyralab/MuseTalk.git "$MUSETALK_PATH"
else
  echo "   MuseTalk already cloned."
fi
pip install -r "$MUSETALK_PATH/requirements.txt"

# ---- 5. MuseTalk checkpoints ----------------------------------------------
echo ">>> [5/6] Downloading MuseTalk checkpoints..."
mkdir -p "$MUSETALK_MODEL_DIR"/{musetalkV15,sd-vae-ft-mse,whisper,dwpose,face-parse-bisent}

HF="https://huggingface.co"
# MuseTalk V1.5 UNet + config
curl -L --create-dirs -o "$MUSETALK_MODEL_DIR/musetalkV15/unet.pth"      "$HF/TMElyralab/MuseTalk/resolve/main/musetalkV15/unet.pth"
curl -L --create-dirs -o "$MUSETALK_MODEL_DIR/musetalkV15/musetalk.json" "$HF/TMElyralab/MuseTalk/resolve/main/musetalkV15/musetalk.json"
# sd-vae-ft-mse (download via huggingface-cli for the whole repo folder)
pip install "huggingface_hub[cli]" >/dev/null
huggingface-cli download stabilityai/sd-vae-ft-mse --local-dir "$MUSETALK_MODEL_DIR/sd-vae-ft-mse"
# Whisper tiny (audio feature extractor)
huggingface-cli download openai/whisper-tiny --local-dir "$MUSETALK_MODEL_DIR/whisper"
# DWPose + face parsing (used by MuseTalk preprocessing)
huggingface-cli download yzd-v/DWPose --local-dir "$MUSETALK_MODEL_DIR/dwpose" --include "dw-ll_ucoco_384.pth"
huggingface-cli download jonathandinu/face-parsing --local-dir "$MUSETALK_MODEL_DIR/face-parse-bisent" --include "79999_iter.pth" || true

# ---- 6. Smoke check ---------------------------------------------------------
echo ">>> [6/6] Verifying installation..."
python - <<'PY'
import importlib
missing = []
for mod in ("torch", "fastapi", "uvicorn", "aiortc", "av", "cv2", "PIL", "numpy", "edge_tts"):
    try:
        importlib.import_module(mod)
    except ImportError:
        missing.append(mod)
import torch
print("PyTorch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), "| VRAM:",
          round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
print("Missing modules:", missing or "none")
PY

cat <<EOF

======================================================================
 Installation complete. Set these environment variables (or export
 them in your shell / .env):

   REALTIME_AVATAR_ENABLED=1
   MUSETALK_PATH=$MUSETALK_PATH
   MUSETALK_MODEL_DIR=$MUSETALK_MODEL_DIR
   AVATAR_DEVICE=cuda:0
   AVATAR_FP16=1
   AVATAR_TARGET_FPS=25
   AVATAR_MAX_QUEUE=3
   AVATAR_PORTRAIT_DIR=$PROJECT_ROOT/web/avatar_portraits
   FFMPEG_PATH=ffmpeg

 Then launch:
   Terminal 1:  ollama serve
   Terminal 2:  python -X utf8 scripts/server.py --port 8000
   Terminal 3:  python -X utf8 scripts/realtime_avatar_server.py

 Test:
   python -X utf8 scripts/test_realtime_avatar.py     (no-GPU pipeline test)
======================================================================
EOF
