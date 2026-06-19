#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# ILGAN Cloud Launcher
# ──────────────────────────────────────────────────────────────────────────────
# Usage:
#   chmod +x launch_cloud.sh
#   ./launch_cloud.sh <INSTANCE_IP> <PEM_PATH> [GITHUB_TOKEN]
#
# Example:
#   ./launch_cloud.sh 123.456.789.0 ~/my-key.pem ghp_xxxxxxxxxxxxxxxxxxxx
#
# What this does:
#   1. SSHs into the GPU instance
#   2. Installs system deps (CUDA drivers, Python, etc.)
#   3. Clones the ILGAN repo
#   4. Installs Python dependencies
#   5. Launches training in a nohup'd tmux session
#   6. Sets up auto-logging and checkpointing
#
# The training runs in a tmux session called "ilgan" so you can
# reattach later with: ssh -i <PEM> ubuntu@<IP> tmux attach -t ilgan
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────────
INSTANCE_IP="${1:?Usage: $0 <INSTANCE_IP> <PEM_PATH> [GITHUB_TOKEN]}"
PEM_PATH="${2:?Usage: $0 <INSTANCE_IP> <PEM_PATH> [GITHUB_TOKEN]}"
GITHUB_TOKEN="${3:-}"
REPO_URL="https://github.com/akshaylakkur/GAN.git"
SSH_USER="ubuntu"  # Change to 'ec2-user' for AWS AMI, 'root' for GCP

# Training hyperparameters (tweak these before launching)
IMAGE_SIZE=128
BATCH_SIZE=32
EPOCHS=2000
LR=0.0001
NUM_CLASSES=20
MAX_BOXES=10
N_CRITIC=3
BOX_WEIGHT=10.0
REPULSION_WEIGHT=2.0
REPULSION_THRESHOLD=0.3
CONSISTENCY_WEIGHT=1.0
SAVE_INTERVAL=50        # Save checkpoint every N epochs
EVAL_INTERVAL=25        # Run validation every N epochs
LOG_INTERVAL=10         # Log metrics every N steps
SAMPLE_INTERVAL=100     # Save sample images every N steps

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[✗]${NC} $1"; }
info() { echo -e "${BLUE}[i]${NC} $1"; }

# ── Pre-flight checks ──────────────────────────────────────────────────────
if [ ! -f "$PEM_PATH" ]; then
    err "PEM file not found: $PEM_PATH"
    exit 1
fi

# Test SSH connectivity
info "Testing SSH connection to $INSTANCE_IP..."
if ! ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "$PEM_PATH" "$SSH_USER@$INSTANCE_IP" "echo OK" 2>/dev/null; then
    err "Cannot SSH to $INSTANCE_IP. Check the IP and PEM file."
    exit 1
fi
log "SSH connection OK"

# ── 1. System setup ─────────────────────────────────────────────────────────
info "Installing system dependencies..."
ssh -t -o StrictHostKeyChecking=no -i "$PEM_PATH" "$SSH_USER@$INSTANCE_IP" << 'SYSDEPS'
    set -e
    export DEBIAN_FRONTEND=noninteractive

    # Update package list
    sudo apt-get update -qq

    # Install essential tools
    sudo apt-get install -y -qq \
        build-essential \
        curl \
        wget \
        git \
        tmux \
        htop \
        nvtop \
        python3 \
        python3-pip \
        python3-venv \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        > /dev/null 2>&1

    # Check CUDA
    if command -v nvidia-smi &> /dev/null; then
        echo "CUDA OK: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
        nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1
    else
        echo "WARNING: nvidia-smi not found. GPU may not be configured."
    fi

    # Create project directory
    mkdir -p ~/ilgan
SYSDEPS
log "System dependencies installed"

# ── 2. Clone repo ──────────────────────────────────────────────────────────
info "Cloning ILGAN repository..."
if [ -n "$GITHUB_TOKEN" ]; then
    AUTH_REPO_URL="https://${GITHUB_TOKEN}@github.com/akshaylakkur/GAN.git"
else
    AUTH_REPO_URL="$REPO_URL"
fi

ssh -t -o StrictHostKeyChecking=no -i "$PEM_PATH" "$SSH_USER@$INSTANCE_IP" \
    "cd ~/ilgan && git clone $AUTH_REPO_URL . 2>&1 || (cd ~/ilgan && git pull origin main)"
log "Repository cloned"

# ── 3. Install Python dependencies ─────────────────────────────────────────
info "Installing Python dependencies..."
ssh -t -o StrictHostKeyChecking=no -i "$PEM_PATH" "$SSH_USER@$INSTANCE_IP" << 'PYDEPS'
    set -e
    cd ~/ilgan

    # Create virtual environment
    python3 -m venv .venv
    source .venv/bin/activate

    # Upgrade pip
    pip install --upgrade pip setuptools wheel -q

    # Install PyTorch with CUDA
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 -q

    # Install ILGAN with all extras
    pip install -e ".[all]" -q

    echo "Python dependencies installed"
PYDEPS
log "Python dependencies installed"

# ── 4. Create training script ──────────────────────────────────────────────
info "Creating training launch script..."
ssh -t -o StrictHostKeyChecking=no -i "$PEM_PATH" "$SSH_USER@$INSTANCE_IP" \
    "cat > ~/ilgan/run_training.sh" << 'TRAINSCRIPT'
#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# ILGAN Training Script — launched by launch_cloud.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd ~/ilgan
source .venv/bin/activate

# Create output directories
mkdir -p checkpoints logs samples

# ── Detect GPU count for distributed training ────────────────────────────────
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
echo "Detected $NUM_GPUS GPU(s)"

# ── Build the training command ───────────────────────────────────────────────
# We use the streaming VOC dataset which downloads on-the-fly.
# The --data-root is ignored when streaming is enabled; we pass it
# via a config override.

TRAIN_CMD="ilgan train"

# If multiple GPUs, use torchrun for distributed training
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Using distributed training with $NUM_GPUS GPUs"
    TRAIN_CMD="torchrun --nproc_per_node=$NUM_GPUS --standalone \
        -m ilgan.scripts.cli train"
fi

# ── Launch training ──────────────────────────────────────────────────────────
# We use a custom config that enables streaming VOC and sets VOC-specific params
cat > /tmp/voc_config.yaml << 'VOCCFG'
# Streaming VOC 2012 configuration for ILGAN
data:
  image_size: 128
  batch_size: 32
  num_workers: 8
  use_streaming: true
  streaming_dataset: "voc"
  streaming_cache_size: 512

model:
  latent_dim: 256
  gen_base_channels: 64
  disc_base_channels: 64
  num_attention_heads: 8
  max_boxes: 10
  num_classes: 20

loss:
  adv_weight: 1.0
  box_weight: 10.0
  class_weight: 1.0
  confidence_weight: 1.0
  diversity_weight: 0.1
  consistency_weight: 1.0
  entropy_weight: 0.1
  repulsion_weight: 2.0
  repulsion_threshold: 0.3
  gp_weight: 10.0
  w_global: 0.5

training:
  epochs: 2000
  learning_rate: 0.0001
  beta1: 0.0
  beta2: 0.9
  n_critic: 3
  gradient_accumulation_steps: 1
  use_mixed_precision: true
  grad_checkpoint: true
  clip_grad_norm: 1.0

logging:
  log_interval: 10
  save_interval: 50
  eval_interval: 25
  use_wandb: false
  sample_interval: 100
  sample_dir: "./samples"

paths:
  checkpoint_dir: "./checkpoints"
  log_dir: "./logs"
VOCCFG

# Run training
echo "=========================================="
echo "  ILGAN Training — VOC 2012 (streaming)"
echo "  GPUs: $NUM_GPUS"
echo "  Image size: 128"
echo "  Batch size: 32"
echo "  Epochs: 2000"
echo "=========================================="

$TRAIN_CMD \
    --config /tmp/voc_config.yaml \
    --image-size 128 \
    --batch-size 32 \
    --num-classes 20 \
    --max-boxes 10 \
    --epochs 2000 \
    --lr 0.0001 \
    --box-weight 10.0 \
    --repulsion-weight 2.0 \
    --repulsion-threshold 0.3 \
    --consistency-weight 1.0 \
    --n-critic 3 \
    --mixed-precision \
    --grad-checkpoint \
    --save-interval 50 \
    --eval-interval 25 \
    --log-interval 10 \
    --seed 42 \
    2>&1 | tee logs/training.log

echo "Training complete."
TRAINSCRIPT

chmod +x ~/ilgan/run_training.sh
log "Training script created"

# ── 5. Launch training in tmux ─────────────────────────────────────────────
info "Launching training in tmux session 'ilgan'..."
ssh -t -o StrictHostKeyChecking=no -i "$PEM_PATH" "$SSH_USER@$INSTANCE_IP" << 'LAUNCH'
    set -e
    cd ~/ilgan

    # Kill existing tmux session if any
    tmux kill-session -t ilgan 2>/dev/null || true

    # Create new tmux session running the training script
    tmux new-session -d -s ilgan -n training 'bash run_training.sh'

    # Create a second window for monitoring
    tmux new-window -t ilgan -n monitor 'htop'

    # Create a third window for GPU monitoring
    tmux new-window -t ilgan -n gpu 'nvtop || watch -n 1 nvidia-smi'

    echo "Training launched in tmux session 'ilgan'"
    echo ""
    echo "Commands to monitor:"
    echo "  Attach to session:  tmux attach -t ilgan"
    echo "  View training log:  tail -f ~/ilgan/logs/training.log"
    echo "  Check GPU:          watch -n 1 nvidia-smi"
    echo "  List checkpoints:   ls -lh ~/ilgan/checkpoints/"
    echo "  View samples:       ls ~/ilgan/samples/ | tail -20"
LAUNCH
log "Training launched in tmux session"

# ── 6. Print summary ───────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║              ILGAN Training — Launched Successfully             ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Instance:    $INSTANCE_IP"
echo "║  Session:     tmux attach -t ilgan"
echo "║  Log file:    tail -f ~/ilgan/logs/training.log"
echo "║  Checkpoints: ~/ilgan/checkpoints/"
echo "║  Samples:     ~/ilgan/samples/"
echo "║                                                                    ║"
echo "║  Quick monitoring:                                                ║"
echo "║    ssh -i $PEM_PATH $SSH_USER@$INSTANCE_IP 'tail -f ~/ilgan/logs/training.log'"
echo "║                                                                    ║"
echo "║  Download latest samples:                                          ║"
echo "║    scp -i $PEM_PATH $SSH_USER@$INSTANCE_IP:~/ilgan/samples/latest_* ./"
echo "║                                                                    ║"
echo "║  Stop training:                                                    ║"
echo "║    ssh -i $PEM_PATH $SSH_USER@$INSTANCE_IP 'tmux kill-session -t ilgan'"
echo "╚══════════════════════════════════════════════════════════════════╝"
