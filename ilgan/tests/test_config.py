"""
Tests for the ILGAN Config class.

Verifies:
- Default config loads and all sections exist
- Dictionary / attribute access works
- Override via dict applies correctly
- Invalid paths raise FileNotFoundError
- Type validation catches type mismatches
- Constraint validation catches out-of-range values
"""

import os
import tempfile
import shutil
import unittest

import yaml

from ilgan.utils.config import Config


class TestConfigDefaults(unittest.TestCase):
    """Verify the default configuration loads correctly."""

    def setUp(self):
        # Create a temporary data directory so path validation passes
        self._tmpdir = tempfile.mkdtemp()
        self._data_dir = os.path.join(self._tmpdir, "data")
        os.makedirs(self._data_dir, exist_ok=True)

        # Write a minimal dummy file so the dir is "ready"
        dummy_file = os.path.join(self._data_dir, ".ilgan_placeholder")
        with open(dummy_file, "w") as f:
            f.write("")

        # Build config with an override pointing at our temp data dir
        self.cfg = Config(
            overrides={
                "paths.data_root": self._data_dir,
                "paths.checkpoint_dir": os.path.join(self._tmpdir, "checkpoints"),
                "paths.log_dir": os.path.join(self._tmpdir, "logs"),
            }
        )

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_all_sections_exist(self):
        """All five top-level sections are present."""
        for section in ("data", "model", "loss", "training", "logging", "paths"):
            self.assertIn(section, self.cfg,
                          f"Section '{section}' missing from config")

    def test_data_section_has_all_keys(self):
        """data section contains all expected keys."""
        data_keys = ("image_size", "batch_size", "num_workers",
                     "augment_prob", "yolo_format")
        for k in data_keys:
            self.assertIn(f"data.{k}", self.cfg,
                          f"Key 'data.{k}' missing from config")

    def test_model_section_has_all_keys(self):
        model_keys = ("latent_dim", "gen_base_channels", "disc_base_channels",
                      "num_attention_heads", "max_boxes", "num_classes")
        for k in model_keys:
            self.assertIn(f"model.{k}", self.cfg,
                          f"Key 'model.{k}' missing from config")

    def test_loss_section_has_all_keys(self):
        loss_keys = ("adv_weight", "box_weight", "diversity_weight",
                     "consistency_weight", "gp_weight")
        for k in loss_keys:
            self.assertIn(f"loss.{k}", self.cfg,
                          f"Key 'loss.{k}' missing from config")

    def test_training_section_has_all_keys(self):
        train_keys = ("epochs", "learning_rate", "beta1", "beta2",
                      "n_critic", "gradient_accumulation_steps",
                      "use_mixed_precision", "grad_checkpoint", "clip_grad_norm")
        for k in train_keys:
            self.assertIn(f"training.{k}", self.cfg,
                          f"Key 'training.{k}' missing from config")

    def test_logging_section_has_all_keys(self):
        log_keys = ("log_interval", "save_interval", "eval_interval", "use_wandb")
        for k in log_keys:
            self.assertIn(f"logging.{k}", self.cfg,
                          f"Key 'logging.{k}' missing from config")

    def test_paths_section_has_all_keys(self):
        path_keys = ("data_root", "checkpoint_dir", "log_dir")
        for k in path_keys:
            self.assertIn(f"paths.{k}", self.cfg,
                          f"Key 'paths.{k}' missing from config")

    def test_default_values_data(self):
        """Verify default data values."""
        self.assertEqual(self.cfg.data.image_size, 128)
        self.assertEqual(self.cfg.data.batch_size, 16)
        self.assertEqual(self.cfg.data.num_workers, 4)
        self.assertEqual(self.cfg.data.augment_prob, 0.5)
        self.assertTrue(self.cfg.data.yolo_format)

    def test_default_values_model(self):
        self.assertEqual(self.cfg.model.latent_dim, 256)
        self.assertEqual(self.cfg.model.gen_base_channels, 64)
        self.assertEqual(self.cfg.model.disc_base_channels, 64)
        self.assertEqual(self.cfg.model.num_attention_heads, 8)
        self.assertEqual(self.cfg.model.max_boxes, 20)
        self.assertEqual(self.cfg.model.num_classes, 80)

    def test_default_values_training(self):
        self.assertEqual(self.cfg.training.epochs, 500)
        self.assertAlmostEqual(self.cfg.training.learning_rate, 0.0002)
        self.assertAlmostEqual(self.cfg.training.beta1, 0.0)
        self.assertAlmostEqual(self.cfg.training.beta2, 0.9)
        self.assertEqual(self.cfg.training.n_critic, 5)
        self.assertEqual(self.cfg.training.gradient_accumulation_steps, 1)
        self.assertTrue(self.cfg.training.use_mixed_precision)
        self.assertTrue(self.cfg.training.grad_checkpoint)
        self.assertAlmostEqual(self.cfg.training.clip_grad_norm, 1.0)

    def test_default_values_loss(self):
        self.assertAlmostEqual(self.cfg.loss.adv_weight, 1.0)
        self.assertAlmostEqual(self.cfg.loss.box_weight, 5.0)
        self.assertAlmostEqual(self.cfg.loss.diversity_weight, 0.1)
        self.assertAlmostEqual(self.cfg.loss.consistency_weight, 0.5)
        self.assertAlmostEqual(self.cfg.loss.gp_weight, 10.0)

    def test_default_values_logging(self):
        self.assertEqual(self.cfg.logging.log_interval, 10)
        self.assertEqual(self.cfg.logging.save_interval, 50)
        self.assertEqual(self.cfg.logging.eval_interval, 100)
        self.assertFalse(self.cfg.logging.use_wandb)


class TestConfigOverrides(unittest.TestCase):
    """Verify override functionality."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._data_dir = os.path.join(self._tmpdir, "data")
        os.makedirs(self._data_dir, exist_ok=True)
        dummy = os.path.join(self._data_dir, ".placeholder")
        with open(dummy, "w") as f:
            f.write("")

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_override_via_dict(self):
        """Overriding a value via the overrides dict applies correctly."""
        cfg = Config(
            overrides={
                "training.epochs": 100,
                "data.batch_size": 32,
                "paths.data_root": self._data_dir,
                "paths.checkpoint_dir": os.path.join(self._tmpdir, "ckpt"),
                "paths.log_dir": os.path.join(self._tmpdir, "logs"),
            }
        )
        self.assertEqual(cfg.training.epochs, 100)
        self.assertEqual(cfg.data.batch_size, 32)

    def test_override_via_setitem(self):
        """Setting a value via cfg['key'] = value works."""
        cfg = Config(
            overrides={
                "paths.data_root": self._data_dir,
                "paths.checkpoint_dir": os.path.join(self._tmpdir, "ckpt"),
                "paths.log_dir": os.path.join(self._tmpdir, "logs"),
            }
        )
        cfg["training.epochs"] = 250
        self.assertEqual(cfg.training.epochs, 250)

    def test_to_dict_returns_copy(self):
        """to_dict() returns a deep copy, mutations don't affect original."""
        cfg = Config(
            overrides={
                "paths.data_root": self._data_dir,
                "paths.checkpoint_dir": os.path.join(self._tmpdir, "ckpt"),
                "paths.log_dir": os.path.join(self._tmpdir, "logs"),
            }
        )
        d = cfg.to_dict()
        d["training"]["epochs"] = 9999
        self.assertEqual(cfg.training.epochs, 500)

    def test_flatten(self):
        """flatten() returns dotted-key representation."""
        cfg = Config(
            overrides={
                "paths.data_root": self._data_dir,
                "paths.checkpoint_dir": os.path.join(self._tmpdir, "ckpt"),
                "paths.log_dir": os.path.join(self._tmpdir, "logs"),
            }
        )
        flat = cfg.flatten()
        self.assertIn("data.image_size", flat)
        self.assertIn("training.epochs", flat)
        self.assertEqual(flat["data.image_size"], 128)


class TestConfigInvalidPaths(unittest.TestCase):
    """Verify invalid paths raise errors."""

    def test_missing_data_root_raises(self):
        """A non-existent data_root should raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            Config(
                overrides={
                    "paths.data_root": "/tmp/nonexistent_ilgan_data_xyz",
                }
            )

    def test_missing_user_config_raises(self):
        """A non-existent user config path should raise FileNotFoundError."""
        from ilgan.utils.config import Config
        with self.assertRaises(FileNotFoundError):
            Config(
                user_config="/tmp/nonexistent_ilgan_config.yaml",
                overrides={
                    "paths.data_root": "/tmp",
                }
            )


class TestConfigTypeValidation(unittest.TestCase):
    """Verify type and constraint validation."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._data_dir = os.path.join(self._tmpdir, "data")
        os.makedirs(self._data_dir, exist_ok=True)
        dummy = os.path.join(self._data_dir, ".placeholder")
        with open(dummy, "w") as f:
            f.write("")

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_image_size_must_be_int(self):
        """Providing a float for image_size should raise TypeError."""
        with self.assertRaises(TypeError):
            Config(
                overrides={
                    "data.image_size": 128.0,
                    "paths.data_root": self._data_dir,
                    "paths.checkpoint_dir": os.path.join(self._tmpdir, "ckpt"),
                    "paths.log_dir": os.path.join(self._tmpdir, "logs"),
                }
            )

    def test_batch_size_must_be_positive(self):
        """batch_size must be > 0."""
        with self.assertRaises(ValueError):
            Config(
                overrides={
                    "data.batch_size": 0,
                    "paths.data_root": self._data_dir,
                    "paths.checkpoint_dir": os.path.join(self._tmpdir, "ckpt"),
                    "paths.log_dir": os.path.join(self._tmpdir, "logs"),
                }
            )

    def test_learning_rate_must_be_positive(self):
        """learning_rate must be > 0."""
        with self.assertRaises(ValueError):
            Config(
                overrides={
                    "training.learning_rate": 0.0,
                    "paths.data_root": self._data_dir,
                    "paths.checkpoint_dir": os.path.join(self._tmpdir, "ckpt"),
                    "paths.log_dir": os.path.join(self._tmpdir, "logs"),
                }
            )

    def test_augment_prob_range(self):
        """augment_prob must be in [0, 1]."""
        with self.assertRaises(ValueError):
            Config(
                overrides={
                    "data.augment_prob": 1.5,
                    "paths.data_root": self._data_dir,
                    "paths.checkpoint_dir": os.path.join(self._tmpdir, "ckpt"),
                    "paths.log_dir": os.path.join(self._tmpdir, "logs"),
                }
            )

    def test_beta1_range(self):
        """beta1 must be in [0.0, 1.0)."""
        with self.assertRaises(ValueError):
            Config(
                overrides={
                    "training.beta1": 1.0,
                    "paths.data_root": self._data_dir,
                    "paths.checkpoint_dir": os.path.join(self._tmpdir, "ckpt"),
                    "paths.log_dir": os.path.join(self._tmpdir, "logs"),
                }
            )


class TestConfigUserFile(unittest.TestCase):
    """Verify loading from a user-provided YAML file."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._data_dir = os.path.join(self._tmpdir, "data")
        os.makedirs(self._data_dir, exist_ok=True)
        dummy = os.path.join(self._data_dir, ".placeholder")
        with open(dummy, "w") as f:
            f.write("")

        # Write a user config file
        self._user_cfg_path = os.path.join(self._tmpdir, "user_config.yaml")
        user_cfg = {
            "training": {"epochs": 10, "learning_rate": 0.001},
            "data": {"batch_size": 8},
        }
        with open(self._user_cfg_path, "w") as f:
            yaml.dump(user_cfg, f)

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_user_config_overrides_defaults(self):
        """Values from user config file override defaults."""
        cfg = Config(
            user_config=self._user_cfg_path,
            overrides={
                "paths.data_root": self._data_dir,
                "paths.checkpoint_dir": os.path.join(self._tmpdir, "ckpt"),
                "paths.log_dir": os.path.join(self._tmpdir, "logs"),
            },
        )
        self.assertEqual(cfg.training.epochs, 10)
        self.assertAlmostEqual(cfg.training.learning_rate, 0.001)
        self.assertEqual(cfg.data.batch_size, 8)
        # Unchanged default
        self.assertEqual(cfg.data.image_size, 128)


if __name__ == "__main__":
    unittest.main()