#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# ILGAN — One-shot setup and training on H100 (NVIDIA NIM / Lambda / any cloud)
# ──────────────────────────────────────────────────────────────────────────────
# Usage:
#   chmod +x setup_and_train.sh
#   ./setup_and_train.sh
#
# What this does:
#   1. Installs system packages + CUDA-compatible PyTorch
#   2. Clones the ILGAN repo (or pulls latest if already cloned)
#   3. Installs Python dependencies
#   4. Downloads + extracts Pascal VOC 2012 to local SSD (one-time ~2 min)
#   5. Launches training with the full model (64ch, 128×128, batch=64)
#
# Config
# ──────────────────────────────────────────────────────────────────────────────
# You can override these by exporting them before running:
#
#   BATCH_SIZE=64     EPOCHS=1500     IMAGE_SIZE=128
#   LR=0.0002         N_CRITIC=5      VOC_DIR=./data/voc
#
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────────
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-1500}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
LR="${LR:-0.0002}"
N_CRITIC="${N_CRITIC:-5}"
VOC_DIR="${VOC_DIR:-./data/voc}"
NUM_CLASSES=20
MAX_BOXES=10
LATENT_DIM=256
GEN_CHANNELS=64
DISC_CHANNELS=64

GIT_REPO="https://github.com/akshaylakkur/GAN.git"
PROJECT_DIR="${HOME}/ilgan"

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[✗]${NC} $1"; }
info() { echo -e "${BLUE}[i]${NC} $1"; }
header() { echo -e "${CYAN}${BOLD}═══ $1 ═══${NC}"; }

# ──────────────────────────────────────────────────────────────────────────────
# 1. System dependencies
# ──────────────────────────────────────────────────────────────────────────────
header "System Setup"

# Detect if we're on Ubuntu/Debian
if command -v apt-get &>/dev/null; then
    info "Installing system packages..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        build-essential \
        python3-dev \
        python3-pip \
        python3-venv \
        wget curl git \
        tmux htop \
        libgl1-mesa-glx libglib2.0-0 \
        > /dev/null 2>&1
    log "System packages installed"
else
    warn "apt-get not found — skipping system package install. Assuming packages are present."
fi

# ──────────────────────────────────────────────────────────────────────────────
# 2. Clone / pull repo
# ──────────────────────────────────────────────────────────────────────────────
header "Repository Setup"

if [ -d "$PROJECT_DIR" ]; then
    info "Repository exists — pulling latest..."
    cd "$PROJECT_DIR"
    git pull --rebase 2>/dev/null || warn "git pull failed (network?), continuing with local copy."
else
    info "Cloning repository..."
    git clone "$GIT_REPO" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi
log "Repository ready at $PROJECT_DIR"

# ──────────────────────────────────────────────────────────────────────────────
# 3. Python virtual environment + dependencies
# ──────────────────────────────────────────────────────────────────────────────
header "Python Environment Setup"

# Use existing venv or create one
if [ ! -d ".venv" ]; then
    info "Creating Python virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel -q

# Install PyTorch with CUDA 12.1 (H100 compatible)
# This detects the current PyTorch version but forces CUDA build
info "Installing PyTorch with CUDA support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q

# Install ILGAN with all extras
info "Installing ILGAN and dependencies..."
pip install -e ".[all]" -q

# Verify CUDA is detected
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
device_name = torch.cuda.get_device_name(0)
total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f'  GPU: {device_name} ({total_mem:.0f} GB VRAM)')
print(f'  CUDA version: {torch.version.cuda}')
print(f'  PyTorch version: {torch.__version__}')
"

log "Python environment ready"

# ──────────────────────────────────────────────────────────────────────────────
# 4. Download Pascal VOC 2012 dataset
# ──────────────────────────────────────────────────────────────────────────────
header "Dataset Download"

mkdir -p "$VOC_DIR"

# The download_voc.py script handles:
#   - Resumable downloads (safe to Ctrl+C and re-run)
#   - SHA-256 verification of the tar archive
#   - Atomic extraction (temp dir, then rename on success)
#   - Idempotency (safe to re-run — skips already-done work)
#   - Progress bars via tqdm
#
# It downloads the ~2GB tar, extracts it, converts VOC XML annotations
# to YOLO-format .txt files, and creates train.txt / val.txt split files.
#
# Result: $VOC_DIR/images/ (11,540 JPEGs), $VOC_DIR/labels/ (11,540 .txt files)

info "Downloading and preparing VOC 2012 (one-time cost, ~2 min)..."
python -m ilgan.scripts.download_voc \
    --output-dir "$VOC_DIR" \
    --download-dir "$VOC_DIR" \
    --resume

# Verify
NUM_IMAGES=$(ls "$VOC_DIR/images/" 2>/dev/null | wc -l)
NUM_LABELS=$(ls "$VOC_DIR/labels/" 2>/dev/null | wc -l)
log "Dataset ready: $NUM_IMAGES images, $NUM_LABELS labels"

# ──────────────────────────────────────────────────────────────────────────────
# 5. Launch training
# ──────────────────────────────────────────────────────────────────────────────
header "Training"

# Create output directories
mkdir -p checkpoints logs samples

# Detect multi-GPU
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")

if [ "$NUM_GPUS" -gt 1 ]; then
    info "Multi-GPU detected ($NUM_GPUS GPUs) — using torchrun"
    TRAIN_CMD="torchrun --nproc_per_node=$NUM_GPUS --standalone -m ilgan.scripts.cli train"
else
    info "Single GPU detected — using direct launch"
    TRAIN_CMD="python -m ilgan.scripts.cli train"
fi

info "Training configuration:"
echo ""
echo "  Image size:      ${IMAGE_SIZE}×${IMAGE_SIZE}"
echo "  Batch size:      ${BATCH_SIZE}"
echo "  Epochs:          ${EPOCHS}"
echo "  Learning rate:   ${LR}"
echo "  Latent dim:      ${LATENT_DIM}"
echo "  Gen channels:    ${GEN_CHANNELS}"
echo "  Disc channels:   ${DISC_CHANNELS}"
echo "  n_critic:        ${N_CRITIC}"
echo "  Mixed precision: BF16 (H100 tensor cores)"
echo "  Gradient ckpt:   ON"
echo "  GPUs:            ${NUM_GPUS}"
echo "  Data:            ${VOC_DIR} (${NUM_IMAGES} images)"
echo "  Checkpoints:     ./checkpoints/"
echo "  Logs:            ./logs/"
echo "  Samples:         ./logs/sample_grid_boxes_*.png"
echo ""

log "Starting training..."

# ── Run ──────────────────────────────────────────────────────────────────────
$TRAIN_CMD \
    --data-root "$VOC_DIR" \
    --image-size "$IMAGE_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --latent-dim "$LATENT_DIM" \
    --num-classes "$NUM_CLASSES" \
    --max-boxes "$MAX_BOXES" \
    --n-critic "$N_CRITIC" \
    --num-workers 8 \
    --mixed-precision \
    --grad-checkpoint \
    --save-interval 50 \
    --eval-interval 25 \
    --log-interval 10 \
    --seed 42 \
    2>&1 | tee logs/training.log

# ──────────────────────────────────────────────────────────────────────────────
# 6. Done
# ──────────────────────────────────────────────────────────────────────────────
header "Training Complete"

echo ""
echo "  Final checkpoints:   ls -lh ./checkpoints/"
echo "  Training log:        tail -f ./logs/training.log"
echo "  Sample grids:        ls -lh ./logs/sample_grid_boxes_*.png"
echo "  Loss curves:         check ./logs/ for loss_curves.png"
echo ""
log "All done!"