# ILGAN — One-shot Image & Bounding Box Generation via GANs

**ILGAN** (Image-Label Generative Adversarial Network) is a novel dual-output GAN that **simultaneously generates hyper-realistic images and their corresponding bounding box labels** from a single latent vector. It is the first architecture to unify unconditional image generation with object detection in a single forward pass.

> **Status**: Alpha — research-grade implementation with full training pipeline, CLI, metrics, and mathematical foundations documented.

---

## ✨ Key Idea

Most GANs generate images. Object detectors localise objects in images. **ILGAN does both at once** — from a random noise vector `z`, it produces:

- A **full RGB image** (e.g., 128×128)
- A set of **bounding boxes** with class labels and confidence scores

The generator learns that the image content and the spatial layout of objects are two views of the same underlying scene, enforced by a **cross-modal consistency constraint** that aligns image and box representations in a shared feature space.

---

## 🧠 How It Works — High-Level Architecture

```
                    ┌──────────────────────────────┐
    z ~ N(0, I) ──▶ │       ILGANGenerator         │
                    │  ┌─────────────────────────┐  │
                    │  │  ContentDecoder          │  │──▶ Image (128×128 RGB)
                    │  │  (progressive upsampler)  │  │──▶ Skip features (multi-res)
                    │  └──────────┬──────────────┘  │
                    │             ▼                  │
                    │  ┌─────────────────────────┐  │
                    │  │  SpatialHead             │  │──▶ Bounding boxes (cx,cy,w,h)
                    │  │  (coarse-to-fine SCCA)  │  │──▶ Class logits (80 classes)
                    │  └─────────────────────────┘  │  │──▶ Confidence scores
                    └──────────────────────────────┘  │
                                                     ▼
                                          ┌──────────────────────┐
                                          │  ImageDiscriminator  │
                                          │  (PatchGAN + global) │──▶ Realism scores
                                          └──────────────────────┘
```

### The Generator

The **ContentDecoder** is a progressive-growing decoder that transforms `z` into an image and multi-resolution skip features. The **SpatialHead** consumes these skip features through **Spatial-Content Cross-Attention (SCCA)** modules — a coarse-to-fine cascade that refines bounding box proposals from low-res to high-res features.

### The Discriminator

A **PatchGAN-style** discriminator that produces both a spatial grid of local realism scores (4×4 grid) and a single global score per image. It only sees images — never bounding boxes — forcing the generator to learn that realistic images naturally imply correct object layouts.

### Loss Functions (6 complementary terms)

| Loss | What it does | Weight |
|------|-------------|--------|
| **WGAN-GP Adversarial** | Wasserstein distance + gradient penalty for stable GAN training | 1.0 |
| **GIoU + Smooth L1** | Bounding box regression (spatial overlap + coordinate accuracy) | 5.0 |
| **Cross-Entropy** | Class label prediction for each box | 1.0 |
| **BCE Confidence** | Objectness score — learn which slots contain objects | 1.0 |
| **Collapse Prevention** | 4 sub-losses: attention entropy, slot repulsion, feature diversity, latent diversity | 0.1–1.0 |
| **Cross-Modal Consistency** | Aligns image and box features in a shared embedding space | 0.5 |

### Collapse Prevention (the key innovation)

ILGAN uses **four mathematical mechanisms** to prevent both image mode collapse and bounding box collapse:

1. **Slot Repulsion** — penalises slots whose attention centres are too close (< 0.2 normalised distance), forcing them to spread out spatially
2. **Attention Entropy** — encourages each slot to focus on a compact region (low entropy = precise localisation)
3. **Feature Diversity** — penalises spatial feature similarity in the decoder, preventing uniform image generation
4. **Latent Diversity** — pushes latent vectors apart in a batch, preventing all outputs from converging

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-org/ilgan.git
cd ilgan

# Install with pip (editable mode for development)
pip install -e .

# With all optional dependencies
pip install -e ".[all]"
```

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0 (CUDA recommended)
- Click, NumPy, PyYAML, Pillow, tqdm, SciPy

---

## 🚀 Quick Start

### 1. Prepare your dataset

Organise your dataset with images and YOLO-format labels:

```
data/
├── images/
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
└── labels/
    ├── 000000.txt
    ├── 000001.txt
    └── ...
```

Each label file contains one line per object:
```
<class_id> <cx> <cy> <w> <h>
```
where `(cx, cy, w, h)` are normalised to [0, 1].

### 2. Train a model

```bash
# Train with defaults (COCO-style: 80 classes, 128×128 images)
ilgan train --data-root ./data

# Train with custom configuration
ilgan train --data-root ./data \
    --config my_config.yaml \
    --image-size 256 \
    --batch-size 32 \
    --epochs 1000 \
    --lr 0.0001 \
    --use-wandb

# Resume from checkpoint
ilgan train --data-root ./data \
    --resume ./checkpoints/checkpoint_epoch_100_step_5000.pt
```

### 3. Evaluate

```bash
ilgan evaluate \
    --checkpoint ./checkpoints/best_checkpoint.pt \
    --data-root ./data \
    --num-samples 1000
```

### 4. Generate samples

```bash
ilgan generate \
    --checkpoint ./checkpoints/best_checkpoint.pt \
    --num-samples 64 \
    --output-dir ./generated
```

### 5. Analyse and debug

```bash
# Check GPU availability
ilgan list-devices

# Diagnose training issues (vanishing/exploding gradients)
ilgan analyze-losses --data-root ./data --steps 15

# Profile GPU memory usage
ilgan profile-memory --batch-size 4 --image-size 128

# Understand your dataset
ilgan compute-statistics --data-root ./data
```

---

## 📊 Metrics

ILGAN tracks a comprehensive set of metrics during training and evaluation:

| Metric | Description |
|--------|-------------|
| **FID** | Fréchet Inception Distance — image quality and diversity |
| **Inception Score** | Image quality and classifiability |
| **mAP@0.5** | Mean Average Precision for bounding box detection |
| **GIoU** | Generalized IoU between predicted and ground-truth boxes |
| **Joint Score** | Combined image quality + detection accuracy (custom) |
| **Cosine Similarity** | Cross-modal alignment between image and box features |
| **Gradient Norms** | Per-module gradient statistics for debugging |

---

## ⚙️ Configuration

All hyperparameters are configured via YAML. The default config is at `ilgan/configs/default_config.yaml`. Key sections:

```yaml
data:
  image_size: 128
  batch_size: 16
  num_workers: 4

model:
  latent_dim: 256
  gen_base_channels: 64
  disc_base_channels: 64
  num_attention_heads: 8
  max_boxes: 20
  num_classes: 80

loss:
  adv_weight: 1.0
  box_weight: 5.0
  gp_weight: 10.0
  consistency_weight: 0.5
  diversity_weight: 0.1
  repulsion_weight: 1.0

training:
  epochs: 500
  learning_rate: 0.0002
  beta1: 0.0
  beta2: 0.9
  n_critic: 5
  use_mixed_precision: true
  grad_checkpoint: true
  clip_grad_norm: 1.0
```

CLI options override config values, which override defaults.

---

## 📁 Project Structure

```
ilgan/
├── __init__.py              # Package metadata
├── __main__.py              # python -m ilgan entry point
├── main.py                  # Console script entry point
├── configs/
│   └── default_config.yaml  # Default hyperparameters
├── data/
│   ├── dataset.py           # Dataset loading & YOLO parsing
│   ├── dataloader.py        # Train/val dataloader factory
│   ├── structures.py        # Batch data structures
│   ├── augmentation.py     # Standard augmentations
│   └── advanced_augmentation.py  # Mosaic, MixUp, CutOut
├── docs/
│   └── mathematical_foundations.md  # Full mathematical specification
├── losses/
│   ├── adversarial.py       # WGAN-GP with local+global scores
│   ├── box_regression.py    # GIoU, Smooth L1, class, confidence
│   ├── collapse_prevention.py  # Entropy, repulsion, diversity losses
│   ├── consistency.py       # Cross-modal image-box alignment
│   └── __init__.py          # LossAggregator — orchestrates all losses
├── metrics/
│   ├── image_metrics.py     # FID, Inception Score
│   ├── box_metrics.py       # mAP, GIoU metrics
│   └── joint_metrics.py     # Combined image+box evaluation
├── models/
│   ├── generator.py         # ContentDecoder + SpatialHead + ILGANGenerator
│   ├── discriminator.py     # PatchGAN discriminator with minibatch stddev
│   └── attention.py         # Spatial-Content Cross-Attention (SCCA)
├── scripts/
│   ├── cli.py               # Click-based CLI (all subcommands)
│   ├── adaptive_optim.py    # Adaptive LR scheduler
│   ├── analyze_data.py      # Dataset analysis utilities
│   ├── checkpoint_strategy.py  # Checkpoint management
│   ├── noise_schedule.py    # Noise annealing schedule
│   ├── proofs.py            # Mathematical proofs
│   └── spectral_regularization.py  # Spectral norm utilities
├── tests/
│   ├── test_attention.py    # SCCA module tests
│   ├── test_generator_*.py  # Generator component tests
│   ├── test_discriminator_image.py  # Discriminator tests
│   ├── test_losses.py       # Loss function tests
│   ├── test_metrics.py      # Metrics tests
│   ├── test_dataset.py      # Dataset tests
│   ├── test_training_pipeline.py  # End-to-end training test
│   └── test_cli.py          # CLI command tests
├── training/
│   ├── trainer.py           # ILGANTrainer — top-level orchestrator
│   ├── train_epoch.py       # Single training epoch loop
│   ├── val_epoch.py         # Validation loop
│   ├── checkpoint.py        # Save/load/resume checkpoints
│   ├── optimizers.py        # Adam optimizers + schedulers
│   ├── mixed_precision.py   # AMP scaler + autocast
│   └── gradient_utils.py   # Gradient clipping, logging, NaN detection
└── utils/
    ├── config.py            # YAML config with validation
    ├── logger.py            # Console + file logger
    ├── visualization.py     # Sample grid generation
    └── wandb_logger.py      # Weights & Biases integration
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ilgan

# Run specific test file
pytest ilgan/tests/test_attention.py -v
```

---

## 📖 Mathematical Foundations

A complete mathematical specification is available at:

[`ilgan/docs/mathematical_foundations.md`](ilgan/docs/mathematical_foundations.md)

This document includes:

- Formal definitions of all loss functions
- **Representation Collapse Prevention Theorem** with proof
- **Cross-Modal Consistency Theorem** with proof
- **Adaptive Diversity Scheduler** optimality proof
- **Gradient Penalty** Lipschitz constraint guarantee
- Complete training objective with all weightings
- Optimisation dynamics and convergence analysis

---

## 🛠️ Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Format code
black ilgan/

# Lint
ruff check ilgan/

# Type check
mypy ilgan/
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 📚 Citation

If you use ILGAN in your research, please cite:

```bibtex
@software{ilgan2024,
  title = {ILGAN: One-shot Image and Bounding Box Generation via Generative Adversarial Networks},
  author = {ILGAN Research Team},
  year = {2024},
  url = {https://github.com/your-org/ilgan}
}
```

---

## 🙏 Acknowledgements

This work builds on foundational research in generative adversarial networks, object detection, and representation learning:

- **WGAN-GP**: Gulrajani et al., 2017
- **Spectral Normalisation**: Miyato et al., 2018
- **Self-Attention GAN**: Zhang et al., 2019
- **StyleGAN**: Karras et al., 2019
- **PatchGAN / Pix2Pix**: Isola et al., 2017
- **GIoU Loss**: Rezatofighi et al., 2019
