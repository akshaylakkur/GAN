#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# ILGAN Local Test — runs a few iterations on MPS (Apple Silicon) or CPU
# to verify the full training pipeline works end-to-end before cloud deploy.
# ──────────────────────────────────────────────────────────────────────────────
# Usage:
#   chmod +x test_local.sh
#   ./test_local.sh                    # 5 steps, streaming VOC
#   ./test_local.sh --steps 10        # custom step count
#   ./test_local.sh --no-stream       # use local data dir instead
#   ./test_local.sh --device cpu      # force CPU
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────────
STEPS=5
USE_STREAMING=true
DEVICE="auto"  # auto, mps, cpu, cuda

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps) STEPS="$2"; shift 2 ;;
        --no-stream) USE_STREAMING=false; shift ;;
        --device) DEVICE="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--steps N] [--no-stream] [--device auto|mps|cpu|cuda]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

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

# ── Detect device ──────────────────────────────────────────────────────────
header "Device Detection"

if [ "$DEVICE" = "auto" ]; then
    DEVICE=$(python3 -c "
import torch
if torch.cuda.is_available():
    print('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('mps')
else:
    print('cpu')
")
fi

log "Using device: ${BOLD}$DEVICE${NC}"

# Show device details
python3 -c "
from ilgan.utils.device import get_device_info
info = get_device_info()
print(f'  CUDA available: {info[\"cuda_available\"]}')
print(f'  MPS available:  {info[\"mps_available\"]}')
print(f'  CPU cores:      {info[\"cpu_count\"]}')
print(f'  Device:         {info[\"device\"]}')
"

# ── Set environment variable for device override ──────────────────────────
export ILGAN_DEVICE="$DEVICE"
log "ILGAN_DEVICE=$DEVICE"

# ── Create output directories ─────────────────────────────────────────────
mkdir -p test_output/checkpoints test_output/logs test_output/samples

# ── Build config ──────────────────────────────────────────────────────────
header "Configuration"

if [ "$USE_STREAMING" = true ]; then
    info "Using streaming VOC dataset (no local data needed)"
    STREAM_CFG="use_streaming: true"
    DATA_ROOT=""
else
    warn "Using local data directory: ./data"
    STREAM_CFG="use_streaming: false"
    DATA_ROOT="--data-root ./data"
fi

cat > /tmp/test_config.yaml << YAMLCFG
data:
  image_size: 64
  batch_size: 2
  num_workers: 0
  ${STREAM_CFG}
  streaming_cache_size: 64

model:
  latent_dim: 64
  gen_base_channels: 16
  disc_base_channels: 16
  num_attention_heads: 2
  max_boxes: 5
  num_classes: 20

loss:
  adv_weight: 1.0
  box_weight: 5.0
  class_weight: 1.0
  confidence_weight: 1.0
  diversity_weight: 0.1
  consistency_weight: 0.5
  entropy_weight: 0.1
  repulsion_weight: 1.0
  repulsion_threshold: 0.2
  gp_weight: 10.0
  w_global: 0.5

training:
  epochs: 1
  learning_rate: 0.0002
  beta1: 0.0
  beta2: 0.9
  n_critic: 1
  gradient_accumulation_steps: 1
  use_mixed_precision: false
  grad_checkpoint: false
  clip_grad_norm: 1.0

logging:
  log_interval: 1
  save_interval: 1
  eval_interval: 1
  use_wandb: false

paths:
  checkpoint_dir: "./test_output/checkpoints"
  log_dir: "./test_output/logs"
YAMLCFG

log "Config written to /tmp/test_config.yaml"
echo ""
cat /tmp/test_config.yaml
echo ""

# ── Run training test ───────────────────────────────────────────────────
header "Running Training Test (${STEPS} steps)"

python3 -c "
import sys, os, time
os.environ['ILGAN_DEVICE'] = '$DEVICE'

from ilgan.utils.config import Config
from ilgan.utils.logger import Logger
from ilgan.utils.device import get_device, get_device_name
from ilgan.training import build_trainer

print(f'Device: {get_device_name()}')
print()

# Build config
cfg = Config(user_config='/tmp/test_config.yaml')
print('Config loaded:')
print(f'  image_size={cfg.data.image_size}')
print(f'  batch_size={cfg.data.batch_size}')
print(f'  latent_dim={cfg.model.latent_dim}')
print(f'  num_classes={cfg.model.num_classes}')
print(f'  max_boxes={cfg.model.max_boxes}')
print(f'  use_streaming={getattr(cfg.data, \"use_streaming\", False)}')
print()

# Create logger
logger = Logger(name='ilgan_test', log_dir='./test_output/logs', level='DEBUG')

# Build trainer
print('Building trainer...')
t0 = time.time()
trainer = build_trainer(config=cfg, logger=logger)
build_time = time.time() - t0
print(f'Trainer built in {build_time:.2f}s')
print(f'  Generator params: {sum(p.numel() for p in trainer.generator.parameters()):,}')
print(f'  Discriminator params: {sum(p.numel() for p in trainer.discriminator.parameters()):,}')
print(f'  Total params: {sum(p.numel() for p in trainer.generator.parameters()) + sum(p.numel() for p in trainer.discriminator.parameters()):,}')
print()

# Create dataloaders
print('Creating dataloaders...')
use_streaming = getattr(cfg.data, 'use_streaming', False)
if use_streaming:
    from ilgan.data.streaming_voc import get_streaming_loaders
    train_loader, val_loader = get_streaming_loaders(
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
        max_boxes=cfg.model.max_boxes,
        num_workers=0,
        cache_size=64,
    )
else:
    from ilgan.data.dataloader import get_train_val_loaders
    train_loader, val_loader = get_train_val_loaders(
        root_dir=cfg.paths.data_root,
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
        num_workers=0,
        val_split=0.2,
        augment=False,
        global_max_boxes=cfg.model.max_boxes,
        train_max_boxes=cfg.model.max_boxes,
        val_max_boxes=cfg.model.max_boxes,
    )

print(f'  Train samples: {len(train_loader.dataset)}')
print(f'  Val samples:   {len(val_loader.dataset)}')
print()

# ── Run a few training steps ─────────────────────────────────────────────
print(f'Running $STEPS training steps...')
print()

trainer.generator.train()
trainer.discriminator.train()
trainer.image_encoder.train()
trainer.box_encoder.train()

data_iter = iter(train_loader)
step_times = []
losses_history = []

for step in range($STEPS):
    step_start = time.time()
    
    # Get batch
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        batch = next(data_iter)
    
    batch = batch.to(trainer.device)
    
    # Sample latents
    z = torch.randn(batch.batch_size, trainer.latent_dim, device=trainer.device)
    
    # ── Generator forward ──────────────────────────────────────────────
    gen_outputs = trainer.generator(z)
    
    # ── Discriminator forward + loss ──────────────────────────────────
    d_loss = trainer.loss_aggregator.discriminator_loss(
        generator_outputs=gen_outputs,
        batch={'images': batch.images, 'boxes': batch.boxes,
               'labels': batch.labels, 'valid_mask': batch.valid_mask},
        discriminator=trainer.discriminator,
    )
    
    trainer.d_optimizer.zero_grad()
    d_loss.backward()
    trainer.d_optimizer.step()
    
    # ── Generator loss ─────────────────────────────────────────────────
    g_loss = trainer.loss_aggregator.generator_loss(
        generator_outputs=gen_outputs,
        batch={'images': batch.images, 'boxes': batch.boxes,
               'labels': batch.labels, 'valid_mask': batch.valid_mask},
        discriminator=trainer.discriminator,
        image_encoder=trainer.image_encoder,
        box_encoder=trainer.box_encoder,
        z_batch=z,
    )
    
    trainer.g_optimizer.zero_grad()
    g_loss.backward()
    trainer.g_optimizer.step()
    
    step_time = time.time() - step_start
    step_times.append(step_time)
    losses_history.append({'g_loss': g_loss.item(), 'd_loss': d_loss.item()})
    
    print(f'  Step {step+1:>2d}/$STEPS | '
          f'G_loss: {g_loss.item():>8.4f} | '
          f'D_loss: {d_loss.item():>8.4f} | '
          f'Time: {step_time:.2f}s')

# ── Summary ──────────────────────────────────────────────────────────────
avg_time = sum(step_times) / len(step_times)
print()
print('═' * 60)
print('  Training Test Complete')
print('═' * 60)
print(f'  Steps completed:  $STEPS')
print(f'  Avg step time:   {avg_time:.3f}s')
print(f'  Total time:      {sum(step_times):.2f}s')
print(f'  Device:          {get_device_name()}')
print(f'  Final G loss:    {losses_history[-1][\"g_loss\"]:.4f}')
print(f'  Final D loss:    {losses_history[-1][\"d_loss\"]:.4f}')
print()

# ── Run validation ───────────────────────────────────────────────────────
print('Running validation...')
trainer.generator.eval()
trainer.discriminator.eval()

from ilgan.metrics import MetricsTracker
tracker = MetricsTracker(
    num_classes=cfg.model.num_classes,
    device=trainer.device,
)

val_iter = iter(val_loader)
val_batches = min(2, len(val_loader))

for val_step in range(val_batches):
    try:
        batch = next(val_iter)
    except StopIteration:
        break
    
    batch = batch.to(trainer.device)
    z = torch.randn(batch.batch_size, trainer.latent_dim, device=trainer.device)
    
    with torch.no_grad():
        gen_outputs = trainer.generator(z)
    
    # Update metrics
    tracker.update_image_metrics(batch.images, gen_outputs['image'])
    tracker.update_box_metrics(
        pred_boxes=gen_outputs['boxes'],
        pred_scores=gen_outputs['confidences'].squeeze(-1),
        pred_labels=gen_outputs['class_logits'].argmax(dim=-1),
        target_boxes=batch.boxes,
        target_labels=batch.labels,
        valid_mask=batch.valid_mask,
    )

# Compute and print all metrics
all_metrics = tracker.compute_all()
print()
print('═' * 60)
print('  Validation Metrics')
print('═' * 60)

for key, value in sorted(all_metrics.items()):
    if isinstance(value, float):
        print(f'  {key:40s}: {value:.6f}')
    else:
        print(f'  {key:40s}: {value}')

print()
print('═' * 60)
print('  ✅  All tests passed!')
print('═' * 60)
" 2>&1

# ── Check output files ───────────────────────────────────────────────────
header "Output Files"
echo ""
echo "Checkpoints:"
ls -lh test_output/checkpoints/ 2>/dev/null || echo "  (none)"
echo ""
echo "Logs:"
ls -lh test_output/logs/ 2>/dev/null || echo "  (none)"
echo ""

# ── Cleanup ─────────────────────────────────────────────────────────────
info "Test output saved to ./test_output/"
info "Run 'rm -rf test_output' to clean up"
echo ""
log "Local test complete!"
