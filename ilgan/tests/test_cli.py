"""
Tests for the ILGAN Click CLI (``ilgan/scripts/cli.py``).

This test suite verifies:

1. **Help output**: Every CLI command (``train``, ``evaluate``, ``generate``,
   ``list-devices``, ``analyze-losses``, ``profile-memory``,
   ``compute-statistics``) prints a non-empty help message when invoked with
   ``--help``.

2. **Train command**: Running ``train`` with a minimal configuration
   (``--epochs 1 --batch-size 2 --image-size 32``) on a tiny synthetic
   dataset completes without errors and produces checkpoint files.

3. **Generate command**: Running ``generate`` with a freshly trained model
   produces output image and label files in the specified output directory.

4. **Evaluate command**: Running ``evaluate`` with a trained model computes
   metrics and saves them to a JSON file.

All tests use Click's ``CliRunner`` to invoke commands in isolation without
spawning subprocesses.  A temporary directory with a synthetic YOLO-format
dataset is created in ``setUpClass`` and cleaned up in ``tearDownClass``.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import torch
from click.testing import CliRunner

from ilgan.scripts.cli import cli
from ilgan.utils.config import Config
from ilgan.utils.logger import Logger


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: synthetic dataset creation
# ──────────────────────────────────────────────────────────────────────────────


def _create_synthetic_dataset(
    root_dir: str,
    num_images: int = 10,
    image_size: int = 32,
    num_classes: int = 2,
    max_boxes_per_image: int = 3,
    seed: int = 42,
) -> None:
    """Create a tiny synthetic YOLO-format dataset for testing.

    Parameters
    ----------
    root_dir : str
        Root directory where ``images/`` and ``labels/`` subdirectories
        will be created.
    num_images : int
        Number of synthetic images to generate.
    image_size : int
        Spatial size of generated images (square).
    num_classes : int
        Number of distinct class labels.
    max_boxes_per_image : int
        Maximum number of bounding boxes per image.
    seed : int
        Random seed for reproducibility.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    images_dir = os.path.join(root_dir, "images")
    labels_dir = os.path.join(root_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    stems: List[str] = []
    for i in range(num_images):
        stem = f"synth_{i:04d}"
        stems.append(stem)

        # Generate random image
        img_tensor = torch.rand(3, image_size, image_size, generator=rng) * 255
        img_tensor = img_tensor.to(torch.uint8)
        from torchvision.utils import save_image

        save_image(
            img_tensor.float() / 255.0,
            os.path.join(images_dir, f"{stem}.png"),
        )

        # Generate random bounding boxes (1 to max_boxes_per_image)
        num_boxes = torch.randint(
            1, max_boxes_per_image + 1, (1,), generator=rng
        ).item()
        label_lines: List[str] = []
        for _ in range(num_boxes):
            cls_id = torch.randint(0, num_classes, (1,), generator=rng).item()
            cx = torch.rand(1, generator=rng).item() * 0.8 + 0.1
            cy = torch.rand(1, generator=rng).item() * 0.8 + 0.1
            w = torch.rand(1, generator=rng).item() * 0.3 + 0.05
            h = torch.rand(1, generator=rng).item() * 0.3 + 0.05
            label_lines.append(
                f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
            )

        with open(os.path.join(labels_dir, f"{stem}.txt"), "w") as f:
            f.write("\n".join(label_lines) + "\n")

    # Write split files (8 train / 2 val)
    train_stems = stems[:8]
    val_stems = stems[8:]

    with open(os.path.join(root_dir, "train.txt"), "w") as f:
        f.write("\n".join(train_stems) + "\n")

    with open(os.path.join(root_dir, "val.txt"), "w") as f:
        f.write("\n".join(val_stems) + "\n")


def _create_dummy_checkpoint(
    checkpoint_path: str,
    config: Config,
    device: torch.device,
) -> str:
    """Create a minimal dummy checkpoint file for testing generate/evaluate.

    This function builds a tiny ILGAN model, runs one forward pass, and
    saves the state dicts to *checkpoint_path*.  The checkpoint contains
    all keys expected by ``CheckpointManager.load``.

    Parameters
    ----------
    checkpoint_path : str
        Path where the checkpoint will be saved.
    config : Config
        ILGAN configuration (used to determine model architecture).
    device : torch.device
        Device to create the model on.

    Returns
    -------
    str
        The path to the saved checkpoint (same as *checkpoint_path*).
    """
    from ilgan.losses.consistency import BoxFeatureEncoder, ImageFeatureEncoder
    from ilgan.models import ILGANGenerator, ImageDiscriminator
    from ilgan.training.checkpoint import CheckpointManager
    from ilgan.training.optimizers import build_optimizers

    # Build models with tiny architecture
    generator = ILGANGenerator(config).to(device)
    discriminator = ImageDiscriminator(
        disc_base_channels=config.model.disc_base_channels,
        image_size=config.data.image_size,
    ).to(device)
    image_encoder = ImageFeatureEncoder(proj_dim=128).to(device)
    box_encoder = BoxFeatureEncoder(proj_dim=128).to(device)

    # Build optimizers
    g_optimizer, d_optimizer = build_optimizers(
        generator=generator,
        discriminator=discriminator,
        image_encoder=image_encoder,
        box_encoder=box_encoder,
        config=config,
    )

    # Run one forward pass to initialise all parameters
    z = torch.randn(2, config.model.latent_dim, device=device)
    _ = generator(z)

    # Create checkpoint manager and save
    ckpt_dir = os.path.dirname(checkpoint_path)
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_manager = CheckpointManager(
        checkpoint_dir=ckpt_dir,
        config=config,
        max_checkpoints=2,
    )

    saved_path = ckpt_manager.save(
        epoch=0,
        global_step=0,
        generator=generator,
        discriminator=discriminator,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        image_encoder=image_encoder,
        box_encoder=box_encoder,
        metrics={"joint_score": 0.0},
    )

    # Copy to the requested path if different
    if saved_path != checkpoint_path:
        shutil.copy2(saved_path, checkpoint_path)

    return checkpoint_path


# ──────────────────────────────────────────────────────────────────────────────
# TestCase: Help output for all commands
# ──────────────────────────────────────────────────────────────────────────────


class TestCLIHelp(unittest.TestCase):
    """Verify that every CLI command prints a non-empty help message."""

    def setUp(self):
        self.runner = CliRunner()

    def _assert_help_output(self, command: List[str]) -> None:
        """Invoke a command with ``--help`` and verify non-empty output."""
        result = self.runner.invoke(cli, command + ["--help"])
        self.assertEqual(
            result.exit_code,
            0,
            msg=(
                f"Command '{' '.join(command)} --help' failed "
                f"with exit code {result.exit_code}.\n"
                f"Output: {result.output}\n"
                f"Exception: {result.exception}"
            ),
        )
        self.assertGreater(
            len(result.output.strip()),
            0,
            msg=f"Command '{' '.join(command)} --help' produced empty output.",
        )

    def test_cli_help(self):
        """Top-level ``ilgan --help`` prints usage information."""
        self._assert_help_output([])

    def test_train_help(self):
        """``ilgan train --help`` prints training usage."""
        self._assert_help_output(["train"])

    def test_evaluate_help(self):
        """``ilgan evaluate --help`` prints evaluation usage."""
        self._assert_help_output(["evaluate"])

    def test_generate_help(self):
        """``ilgan generate --help`` prints generation usage."""
        self._assert_help_output(["generate"])

    def test_list_devices_help(self):
        """``ilgan list-devices --help`` prints device listing usage."""
        self._assert_help_output(["list-devices"])

    def test_analyze_losses_help(self):
        """``ilgan analyze-losses --help`` prints loss analysis usage."""
        self._assert_help_output(["analyze-losses"])

    def test_profile_memory_help(self):
        """``ilgan profile-memory --help`` prints memory profiling usage."""
        self._assert_help_output(["profile-memory"])

    def test_compute_statistics_help(self):
        """``ilgan compute-statistics --help`` prints statistics usage."""
        self._assert_help_output(["compute-statistics"])


# ──────────────────────────────────────────────────────────────────────────────
# TestCase: Train command
# ──────────────────────────────────────────────────────────────────────────────


class TestCLITrain(unittest.TestCase):
    """Verify the ``train`` command runs with a minimal configuration."""

    @classmethod
    def setUpClass(cls):
        """Create a temporary directory with a synthetic dataset."""
        cls._tmpdir = tempfile.mkdtemp(prefix="ilgan_test_cli_train_")
        cls._data_dir = os.path.join(cls._tmpdir, "data")
        cls._ckpt_dir = os.path.join(cls._tmpdir, "checkpoints")
        cls._log_dir = os.path.join(cls._tmpdir, "logs")
        os.makedirs(cls._ckpt_dir, exist_ok=True)
        os.makedirs(cls._log_dir, exist_ok=True)

        # Create synthetic dataset
        _create_synthetic_dataset(
            root_dir=cls._data_dir,
            num_images=10,
            image_size=32,
            num_classes=2,
            max_boxes_per_image=3,
            seed=42,
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def setUp(self):
        self.runner = CliRunner()

    def test_train_minimal(self):
        """``train`` with minimal config completes without errors.

        Uses:
        - ``--epochs 1`` (single epoch)
        - ``--batch-size 2`` (tiny batch)
        - ``--image-size 32`` (small images)
        - ``--latent-dim 32`` (small latent)
        - ``--gen-base-channels 8`` (tiny generator)
        - ``--disc-base-channels 8`` (tiny discriminator)
        - ``--num-attention-heads 2`` (minimal attention)
        - ``--max-boxes 5`` (few boxes)
        - ``--num-classes 2`` (few classes)
        - ``--num-workers 0`` (no subprocess workers)
        - ``--no-mixed-precision`` (avoid AMP issues in test)
        - ``--no-grad-checkpoint`` (avoid checkpointing overhead)
        - ``--save-interval 1`` (save every epoch)
        - ``--eval-interval 1`` (evaluate every epoch)
        - ``--log-interval 1`` (log every step)
        """
        result = self.runner.invoke(
            cli,
            [
                "train",
                "--data-root", self._data_dir,
                "--epochs", "1",
                "--batch-size", "2",
                "--image-size", "32",
                "--latent-dim", "32",
                "--gen-base-channels", "8",
                "--disc-base-channels", "8",
                "--num-attention-heads", "2",
                "--max-boxes", "5",
                "--num-classes", "2",
                "--num-workers", "0",
                "--no-mixed-precision",
                "--no-grad-checkpoint",
                "--save-interval", "1",
                "--eval-interval", "1",
                "--log-interval", "1",
                "--checkpoint-dir", self._ckpt_dir,
                "--log-dir", self._log_dir,
                "--seed", "42",
            ],
        )

        # Check exit code
        self.assertEqual(
            result.exit_code,
            0,
            msg=(
                f"Train command failed with exit code {result.exit_code}.\n"
                f"Output:\n{result.output}\n"
                f"Exception: {result.exception}"
            ),
        )

        # Verify that checkpoint files were created
        ckpt_files = list(Path(self._ckpt_dir).glob("*.pt"))
        self.assertGreater(
            len(ckpt_files),
            0,
            msg=f"No checkpoint files found in {self._ckpt_dir}.",
        )

        # Verify that log files were created
        log_files = list(Path(self._log_dir).glob("*"))
        self.assertGreater(
            len(log_files),
            0,
            msg=f"No log files found in {self._log_dir}.",
        )

        # Verify output mentions training completion
        self.assertIn(
            "Training Complete",
            result.output,
            msg="Output does not contain 'Training Complete' summary.",
        )


# ──────────────────────────────────────────────────────────────────────────────
# TestCase: Generate command
# ──────────────────────────────────────────────────────────────────────────────


class TestCLIGenerate(unittest.TestCase):
    """Verify the ``generate`` command produces output files."""

    @classmethod
    def setUpClass(cls):
        """Create a temporary directory with a synthetic dataset and a
        dummy checkpoint."""
        cls._tmpdir = tempfile.mkdtemp(prefix="ilgan_test_cli_gen_")
        cls._data_dir = os.path.join(cls._tmpdir, "data")
        cls._ckpt_dir = os.path.join(cls._tmpdir, "checkpoints")
        cls._log_dir = os.path.join(cls._tmpdir, "logs")
        cls._output_dir = os.path.join(cls._tmpdir, "generated")
        os.makedirs(cls._ckpt_dir, exist_ok=True)
        os.makedirs(cls._log_dir, exist_ok=True)

        # Create synthetic dataset
        _create_synthetic_dataset(
            root_dir=cls._data_dir,
            num_images=10,
            image_size=32,
            num_classes=2,
            max_boxes_per_image=3,
            seed=42,
        )

        # Create a minimal config matching the tiny architecture
        cls.cfg = Config(
            overrides={
                "data.image_size": 32,
                "data.batch_size": 2,
                "data.num_workers": 0,
                "model.latent_dim": 32,
                "model.gen_base_channels": 8,
                "model.disc_base_channels": 8,
                "model.num_attention_heads": 2,
                "model.max_boxes": 5,
                "model.num_classes": 2,
                "training.epochs": 1,
                "training.learning_rate": 0.0002,
                "training.beta1": 0.0,
                "training.beta2": 0.9,
                "training.n_critic": 1,
                "training.gradient_accumulation_steps": 1,
                "training.use_mixed_precision": False,
                "training.grad_checkpoint": False,
                "training.clip_grad_norm": 1.0,
                "logging.log_interval": 1,
                "logging.save_interval": 1,
                "logging.eval_interval": 1,
                "logging.use_wandb": False,
                "paths.data_root": cls._data_dir,
                "paths.checkpoint_dir": cls._ckpt_dir,
                "paths.log_dir": cls._log_dir,
            },
        )

        # Create a dummy checkpoint
        cls._checkpoint_path = os.path.join(cls._ckpt_dir, "test_checkpoint.pt")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _create_dummy_checkpoint(cls._checkpoint_path, cls.cfg, device)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def setUp(self):
        self.runner = CliRunner()

    def test_generate_with_checkpoint(self):
        """``generate`` with a trained checkpoint produces output files."""
        result = self.runner.invoke(
            cli,
            [
                "generate",
                "--checkpoint", self._checkpoint_path,
                "--num-samples", "4",
                "--output-dir", self._output_dir,
                "--batch-size", "2",
                "--seed", "42",
                "--log-dir", self._log_dir,
                # Explicitly enable all save flags
                "--save-images",
                "--save-boxes",
                "--save-grid",
            ],
        )

        # Check exit code
        self.assertEqual(
            result.exit_code,
            0,
            msg=(
                f"Generate command failed with exit code {result.exit_code}.\n"
                f"Output:\n{result.output}\n"
                f"Exception: {result.exception}"
            ),
        )

        # Verify output directory structure
        images_dir = os.path.join(self._output_dir, "images")
        labels_dir = os.path.join(self._output_dir, "labels")
        grid_dir = os.path.join(self._output_dir, "grid")

        # Check that image files were created
        image_files = list(Path(images_dir).glob("*.png"))
        self.assertGreater(
            len(image_files),
            0,
            msg=f"No image files found in {images_dir}.",
        )

        # Check that label files were created
        label_files = list(Path(labels_dir).glob("*.txt"))
        self.assertGreater(
            len(label_files),
            0,
            msg=f"No label files found in {labels_dir}.",
        )

        # Check that grid file was created
        grid_files = list(Path(grid_dir).glob("*.png"))
        self.assertGreater(
            len(grid_files),
            0,
            msg=f"No grid files found in {grid_dir}.",
        )

        # Verify output mentions generation complete
        self.assertIn(
            "Generation Complete",
            result.output,
            msg="Output does not contain 'Generation Complete' summary.",
        )

    def test_generate_no_save_flags(self):
        """``generate`` with ``--no-save-images --no-save-boxes --no-save-grid``
        still completes without errors but produces no output files."""
        output_dir_no_save = os.path.join(self._tmpdir, "generated_no_save")
        result = self.runner.invoke(
            cli,
            [
                "generate",
                "--checkpoint", self._checkpoint_path,
                "--num-samples", "2",
                "--output-dir", output_dir_no_save,
                "--batch-size", "2",
                "--seed", "42",
                "--log-dir", self._log_dir,
                "--no-save-images",
                "--no-save-boxes",
                "--no-save-grid",
            ],
        )

        self.assertEqual(
            result.exit_code,
            0,
            msg=(
                f"Generate command (no-save) failed with exit code "
                f"{result.exit_code}.\n"
                f"Output:\n{result.output}\n"
                f"Exception: {result.exception}"
            ),
        )

        # Verify that no image/label/grid directories were created
        # (the output dir itself should exist, but subdirs should not)
        self.assertTrue(os.path.isdir(output_dir_no_save))
        # The CLI creates the directories regardless, but they should be empty
        # or not contain any files
        for subdir in ["images", "labels", "grid"]:
            subdir_path = os.path.join(output_dir_no_save, subdir)
            if os.path.isdir(subdir_path):
                files = list(Path(subdir_path).iterdir())
                self.assertEqual(
                    len(files),
                    0,
                    msg=f"Expected empty {subdir} directory, found {len(files)} files.",
                )


# ──────────────────────────────────────────────────────────────────────────────
# TestCase: Evaluate command
# ──────────────────────────────────────────────────────────────────────────────


class TestCLIEvaluate(unittest.TestCase):
    """Verify the ``evaluate`` command computes metrics."""

    @classmethod
    def setUpClass(cls):
        """Create a temporary directory with a synthetic dataset and a
        dummy checkpoint."""
        cls._tmpdir = tempfile.mkdtemp(prefix="ilgan_test_cli_eval_")
        cls._data_dir = os.path.join(cls._tmpdir, "data")
        cls._ckpt_dir = os.path.join(cls._tmpdir, "checkpoints")
        cls._log_dir = os.path.join(cls._tmpdir, "logs")
        cls._output_dir = os.path.join(cls._tmpdir, "evaluation")
        os.makedirs(cls._ckpt_dir, exist_ok=True)
        os.makedirs(cls._log_dir, exist_ok=True)

        # Create synthetic dataset
        _create_synthetic_dataset(
            root_dir=cls._data_dir,
            num_images=10,
            image_size=32,
            num_classes=2,
            max_boxes_per_image=3,
            seed=42,
        )

        # Create a minimal config matching the tiny architecture
        cls.cfg = Config(
            overrides={
                "data.image_size": 32,
                "data.batch_size": 2,
                "data.num_workers": 0,
                "model.latent_dim": 32,
                "model.gen_base_channels": 8,
                "model.disc_base_channels": 8,
                "model.num_attention_heads": 2,
                "model.max_boxes": 5,
                "model.num_classes": 2,
                "training.epochs": 1,
                "training.learning_rate": 0.0002,
                "training.beta1": 0.0,
                "training.beta2": 0.9,
                "training.n_critic": 1,
                "training.gradient_accumulation_steps": 1,
                "training.use_mixed_precision": False,
                "training.grad_checkpoint": False,
                "training.clip_grad_norm": 1.0,
                "logging.log_interval": 1,
                "logging.save_interval": 1,
                "logging.eval_interval": 1,
                "logging.use_wandb": False,
                "paths.data_root": cls._data_dir,
                "paths.checkpoint_dir": cls._ckpt_dir,
                "paths.log_dir": cls._log_dir,
            },
        )

        # Create a dummy checkpoint
        cls._checkpoint_path = os.path.join(cls._ckpt_dir, "test_checkpoint.pt")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _create_dummy_checkpoint(cls._checkpoint_path, cls.cfg, device)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def setUp(self):
        self.runner = CliRunner()

    def test_evaluate_with_checkpoint(self):
        """``evaluate`` with a trained checkpoint computes metrics."""
        result = self.runner.invoke(
            cli,
            [
                "evaluate",
                "--checkpoint", self._checkpoint_path,
                "--data-root", self._data_dir,
                "--batch-size", "2",
                "--num-samples", "4",
                "--output-dir", self._output_dir,
                "--num-workers", "0",
                "--seed", "42",
                "--log-dir", self._log_dir,
            ],
        )

        # Check exit code
        self.assertEqual(
            result.exit_code,
            0,
            msg=(
                f"Evaluate command failed with exit code {result.exit_code}.\n"
                f"Output:\n{result.output}\n"
                f"Exception: {result.exception}"
            ),
        )

        # Verify that metrics JSON was saved
        metrics_path = os.path.join(self._output_dir, "evaluation_metrics.json")
        self.assertTrue(
            os.path.isfile(metrics_path),
            msg=f"Metrics file not found at {metrics_path}.",
        )

        # Verify that the metrics file contains valid JSON
        with open(metrics_path, "r") as f:
            metrics: Dict[str, Any] = json.load(f)

        self.assertIsInstance(metrics, dict)
        self.assertGreater(
            len(metrics),
            0,
            msg="Metrics dictionary is empty.",
        )

        # Verify output mentions evaluation results
        self.assertIn(
            "Evaluation Results",
            result.output,
            msg="Output does not contain 'Evaluation Results' section.",
        )

    def test_evaluate_missing_checkpoint(self):
        """``evaluate`` with a non-existent checkpoint exits with error."""
        fake_checkpoint = os.path.join(self._tmpdir, "nonexistent.pt")
        result = self.runner.invoke(
            cli,
            [
                "evaluate",
                "--checkpoint", fake_checkpoint,
                "--data-root", self._data_dir,
                "--batch-size", "2",
                "--num-samples", "2",
                "--output-dir", os.path.join(self._tmpdir, "eval_fail"),
                "--num-workers", "0",
                "--seed", "42",
                "--log-dir", self._log_dir,
            ],
        )

        # Should fail because the checkpoint doesn't exist
        self.assertNotEqual(
            result.exit_code,
            0,
            msg=(
                f"Evaluate with non-existent checkpoint should have failed "
                f"but got exit code 0.\n"
                f"Output:\n{result.output}"
            ),
        )


# ──────────────────────────────────────────────────────────────────────────────
# TestCase: List Devices command
# ──────────────────────────────────────────────────────────────────────────────


class TestCLIListDevices(unittest.TestCase):
    """Verify the ``list-devices`` command runs without errors."""

    def setUp(self):
        self.runner = CliRunner()

    def test_list_devices(self):
        """``list-devices`` prints device information without errors."""
        result = self.runner.invoke(cli, ["list-devices"])
        self.assertEqual(
            result.exit_code,
            0,
            msg=(
                f"list-devices command failed with exit code {result.exit_code}.\n"
                f"Output:\n{result.output}\n"
                f"Exception: {result.exception}"
            ),
        )
        self.assertGreater(
            len(result.output.strip()),
            0,
            msg="list-devices produced empty output.",
        )


# ──────────────────────────────────────────────────────────────────────────────
# TestCase: Analyze Losses command
# ──────────────────────────────────────────────────────────────────────────────


class TestCLIAnalyzeLosses(unittest.TestCase):
    """Verify the ``analyze-losses`` command runs with a minimal config."""

    @classmethod
    def setUpClass(cls):
        """Create a temporary directory with a synthetic dataset."""
        cls._tmpdir = tempfile.mkdtemp(prefix="ilgan_test_cli_analyze_")
        cls._data_dir = os.path.join(cls._tmpdir, "data")
        cls._log_dir = os.path.join(cls._tmpdir, "logs")
        os.makedirs(cls._log_dir, exist_ok=True)

        _create_synthetic_dataset(
            root_dir=cls._data_dir,
            num_images=10,
            image_size=32,
            num_classes=2,
            max_boxes_per_image=3,
            seed=42,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def setUp(self):
        self.runner = CliRunner()

    def test_analyze_losses_minimal(self):
        """``analyze-losses`` with minimal config completes without errors."""
        result = self.runner.invoke(
            cli,
            [
                "analyze-losses",
                "--data-root", self._data_dir,
                "--steps", "2",
                "--batch-size", "2",
                "--image-size", "32",
                "--latent-dim", "32",
                "--num-classes", "2",
                "--max-boxes", "5",
                "--seed", "42",
                "--log-dir", self._log_dir,
            ],
        )

        self.assertEqual(
            result.exit_code,
            0,
            msg=(
                f"analyze-losses command failed with exit code "
                f"{result.exit_code}.\n"
                f"Output:\n{result.output}\n"
                f"Exception: {result.exception}"
            ),
        )

        # Verify output mentions analysis completion
        self.assertIn(
            "Analysis Complete",
            result.output,
            msg="Output does not contain 'Analysis Complete' summary.",
        )


# ──────────────────────────────────────────────────────────────────────────────
# TestCase: Profile Memory command
# ──────────────────────────────────────────────────────────────────────────────


class TestCLIProfileMemory(unittest.TestCase):
    """Verify the ``profile-memory`` command runs without errors."""

    def setUp(self):
        self.runner = CliRunner()

    def test_profile_memory_minimal(self):
        """``profile-memory`` with minimal config completes without errors."""
        result = self.runner.invoke(
            cli,
            [
                "profile-memory",
                "--batch-size", "2",
                "--image-size", "32",
                "--latent-dim", "32",
                "--num-classes", "2",
                "--max-boxes", "5",
                "--gen-base-channels", "8",
                "--disc-base-channels", "8",
                "--num-attention-heads", "2",
                "--seed", "42",
            ],
        )

        self.assertEqual(
            result.exit_code,
            0,
            msg=(
                f"profile-memory command failed with exit code "
                f"{result.exit_code}.\n"
                f"Output:\n{result.output}\n"
                f"Exception: {result.exception}"
            ),
        )

        # Verify output mentions profile completion
        self.assertIn(
            "Memory Profile",
            result.output,
            msg="Output does not contain 'Memory Profile' header.",
        )


# ──────────────────────────────────────────────────────────────────────────────
# TestCase: Compute Statistics command
# ──────────────────────────────────────────────────────────────────────────────


class TestCLIComputeStatistics(unittest.TestCase):
    """Verify the ``compute-statistics`` command runs on a dataset."""

    @classmethod
    def setUpClass(cls):
        """Create a temporary directory with a synthetic dataset."""
        cls._tmpdir = tempfile.mkdtemp(prefix="ilgan_test_cli_stats_")
        cls._data_dir = os.path.join(cls._tmpdir, "data")
        os.makedirs(cls._data_dir, exist_ok=True)

        _create_synthetic_dataset(
            root_dir=cls._data_dir,
            num_images=10,
            image_size=32,
            num_classes=2,
            max_boxes_per_image=3,
            seed=42,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def setUp(self):
        self.runner = CliRunner()

    def test_compute_statistics(self):
        """``compute-statistics`` on a dataset prints statistics."""
        result = self.runner.invoke(
            cli,
            [
                "compute-statistics",
                "--data-root", self._data_dir,
                "--max-samples", "5",
                "--verbose",
            ],
        )

        self.assertEqual(
            result.exit_code,
            0,
            msg=(
                f"compute-statistics command failed with exit code "
                f"{result.exit_code}.\n"
                f"Output:\n{result.output}\n"
                f"Exception: {result.exception}"
            ),
        )

        # Verify output contains dataset statistics
        self.assertIn(
            "Dataset Statistics",
            result.output,
            msg="Output does not contain 'Dataset Statistics' header.",
        )

    def test_compute_statistics_with_output(self):
        """``compute-statistics`` with ``--output`` saves JSON."""
        output_path = os.path.join(self._tmpdir, "stats_output.json")
        result = self.runner.invoke(
            cli,
            [
                "compute-statistics",
                "--data-root", self._data_dir,
                "--max-samples", "5",
                "--output", output_path,
            ],
        )

        self.assertEqual(
            result.exit_code,
            0,
            msg=(
                f"compute-statistics (with output) failed with exit code "
                f"{result.exit_code}.\n"
                f"Output:\n{result.output}\n"
                f"Exception: {result.exception}"
            ),
        )

        # Verify JSON file was created
        self.assertTrue(
            os.path.isfile(output_path),
            msg=f"Statistics output file not found at {output_path}.",
        )

        # Verify it contains valid JSON
        with open(output_path, "r") as f:
            stats: Dict[str, Any] = json.load(f)
        self.assertIsInstance(stats, dict)
        self.assertGreater(len(stats), 0)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main()
