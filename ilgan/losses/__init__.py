"""
ILGAN loss functions — master aggregator and re-exports.

The ``ilgan.losses`` package contains all loss functions used for training
the ILGAN dual-output GAN.  This module re-exports every public function
and class from the four sub-modules and provides the central
:class:`LossAggregator` that orchestrates all losses into a single forward
pass.

Modules
-------
adversarial
    WGAN-GP losses adapted for the dual local/global discriminator.
box_regression
    Bounding box regression losses (GIoU, smooth L1, class, confidence).
collapse_prevention
    Novel collapse-prevention losses: attention entropy, slot repulsion,
    feature diversity, and latent diversity.  These mathematically prevent
    both image mode collapse and bounding box collapse.
consistency
    Cross-modal consistency loss that aligns image and bounding box
    representations in a shared feature space via cosine embedding loss.

LossAggregator
--------------
The :class:`LossAggregator` is the central orchestrator.  It:

1. Reads all loss weights from a :class:`ilgan.utils.config.Config` object.
2. Provides a ``__call__`` method that computes **all** losses (adversarial,
   box regression, collapse prevention, consistency) in a single forward
   pass, returning a flat dictionary of every individual loss term plus
   ``"total_g_loss"`` and ``"total_d_loss"``.
3. Provides ``generator_loss()`` and ``discriminator_loss()`` convenience
   methods that return only the scalar loss needed for the respective
   optimiser step.

Usage
-----
::

    from ilgan.losses import LossAggregator

    agg = LossAggregator(cfg)

    # Full loss dict for logging
    losses = agg(generator_outputs, batch, discriminator,
                 image_encoder, box_encoder, z_batch)

    # Scalar losses for optimiser steps
    g_loss = agg.generator_loss(generator_outputs, batch, discriminator,
                                 image_encoder, box_encoder, z_batch)
    d_loss = agg.discriminator_loss(generator_outputs, batch, discriminator)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ilgan.utils.config import Config

# ──────────────────────────────────────────────────────────────────────────────
# Re-export everything from sub-modules
# ──────────────────────────────────────────────────────────────────────────────

# Adversarial losses
from ilgan.losses.adversarial import (
    compute_adversarial_losses,
    gradient_penalty,
    wgan_discriminator_loss,
    wgan_generator_loss,
)

# Box regression losses
from ilgan.losses.box_regression import (
    class_loss,
    compute_box_losses,
    confidence_loss,
    giou_loss,
    l1_box_loss,
)

# Collapse prevention losses
from ilgan.losses.collapse_prevention import (
    attention_entropy_loss,
    compute_collapse_losses,
    feature_diversity_loss,
    latent_diversity_loss,
    repulsion_loss,
)

# Consistency losses
from ilgan.losses.consistency import (
    BoxFeatureEncoder,
    ImageFeatureEncoder,
    compute_consistency_loss,
    consistency_loss,
    cosine_similarity,
)

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Adversarial
    "wgan_discriminator_loss",
    "wgan_generator_loss",
    "gradient_penalty",
    "compute_adversarial_losses",
    # Box regression
    "giou_loss",
    "l1_box_loss",
    "compute_box_losses",
    "class_loss",
    "confidence_loss",
    # Collapse prevention
    "attention_entropy_loss",
    "repulsion_loss",
    "feature_diversity_loss",
    "latent_diversity_loss",
    "compute_collapse_losses",
    # Consistency
    "ImageFeatureEncoder",
    "BoxFeatureEncoder",
    "consistency_loss",
    "cosine_similarity",
    "compute_consistency_loss",
    # Aggregator
    "LossAggregator",
]


# ──────────────────────────────────────────────────────────────────────────────
# LossAggregator — master orchestrator
# ──────────────────────────────────────────────────────────────────────────────


class LossAggregator:
    r"""Central loss aggregator for the ILGAN dual-output GAN.

    The :class:`LossAggregator` reads all loss weights from a
    :class:`~ilgan.utils.config.Config` object and provides a unified
    interface for computing every loss term in the ILGAN objective.

    The total generator loss is the weighted sum of:

    .. math::

        \mathcal{L}_G = w_{adv} \cdot \mathcal{L}_{adv}
            + w_{box} \cdot (\mathcal{L}_{GIoU} + \mathcal{L}_{L1})
            + w_{cls} \cdot \mathcal{L}_{cls}
            + w_{conf} \cdot \mathcal{L}_{conf}
            + w_{entropy} \cdot \mathcal{L}_{entropy}
            + w_{repulsion} \cdot \mathcal{L}_{repulsion}
            + w_{diversity} \cdot \mathcal{L}_{diversity}
            + w_{latent} \cdot \mathcal{L}_{latent}
            + w_{consistency} \cdot \mathcal{L}_{consistency}

    The total discriminator loss is:

    .. math::

        \mathcal{L}_D = \mathcal{L}_{WGAN\_D}
            + \lambda_{gp} \cdot \mathcal{L}_{GP}

    where :math:`\mathcal{L}_{WGAN\_D}` is the WGAN discriminator loss and
    :math:`\mathcal{L}_{GP}` is the gradient penalty.

    Parameters
    ----------
    cfg : Config
        The ILGAN configuration object.  The following keys are read from
        ``cfg.loss``:

        - ``adv_weight`` (float, default ``1.0``): weight for the generator
          adversarial loss.
        - ``box_weight`` (float, default ``5.0``): weight for the combined
          box regression loss (GIoU + L1).
        - ``class_weight`` (float, default ``1.0``): weight for the class
          cross-entropy loss.
        - ``confidence_weight`` (float, default ``1.0``): weight for the
          confidence (objectness) BCE loss.
        - ``entropy_weight`` (float, default ``0.1``): weight for the
          attention entropy loss.
        - ``repulsion_weight`` (float, default ``1.0``): weight for the
          slot repulsion loss.
        - ``diversity_weight`` (float, default ``0.1``): weight for the
          feature diversity loss.
        - ``latent_diversity_weight`` (float, default ``0.01``): weight for
          the latent diversity loss.
        - ``consistency_weight`` (float, default ``0.5``): weight for the
          cross-modal consistency loss.
        - ``gp_weight`` (float, default ``10.0``): gradient penalty
          coefficient (``lambda_gp``).
        - ``w_global`` (float, default ``0.5``): weight for the global
          discriminator score relative to the local score.

        Additionally, ``cfg.loss.repulsion_threshold`` (float, default
        ``0.2``) controls the minimum distance between slot centres of mass
        before repulsion is incurred.

    Raises
    ------
    TypeError
        If ``cfg`` is not a :class:`~ilgan.utils.config.Config` instance.
    """

    def __init__(self, cfg: Config) -> None:
        if not isinstance(cfg, Config):
            raise TypeError(
                f"Expected a Config object, got {type(cfg).__name__}."
            )

        # ── Store the full config for reference ──────────────────────────
        self.cfg = cfg

        # ── Read loss weights from config with sensible defaults ────────
        loss_cfg = cfg.loss

        # Adversarial
        self.adv_weight: float = getattr(loss_cfg, "adv_weight", 1.0)
        self.gp_weight: float = getattr(loss_cfg, "gp_weight", 10.0)
        self.w_global: float = getattr(loss_cfg, "w_global", 0.5)

        # Box regression
        self.box_weight: float = getattr(loss_cfg, "box_weight", 5.0)
        self.class_weight: float = getattr(loss_cfg, "class_weight", 1.0)
        self.confidence_weight: float = getattr(loss_cfg, "confidence_weight", 1.0)

        # Collapse prevention
        self.entropy_weight: float = getattr(loss_cfg, "entropy_weight", 0.1)
        self.repulsion_weight: float = getattr(loss_cfg, "repulsion_weight", 1.0)
        self.diversity_weight: float = getattr(loss_cfg, "diversity_weight", 0.1)
        self.latent_diversity_weight: float = getattr(
            loss_cfg, "latent_diversity_weight", 0.01
        )
        self.repulsion_threshold: float = getattr(
            loss_cfg, "repulsion_threshold", 0.2
        )

        # Consistency
        self.consistency_weight: float = getattr(
            loss_cfg, "consistency_weight", 0.5
        )

        # ── Validate all weights are non-negative ────────────────────────
        self._validate_weights()

    # ──────────────────────────────────────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────────────────────────────────────

    def _validate_weights(self) -> None:
        """Assert that all loss weights are non-negative.

        Raises
        ------
        ValueError
            If any weight is negative.
        """
        weight_attrs = [
            "adv_weight",
            "gp_weight",
            "w_global",
            "box_weight",
            "class_weight",
            "confidence_weight",
            "entropy_weight",
            "repulsion_weight",
            "diversity_weight",
            "latent_diversity_weight",
            "consistency_weight",
        ]
        for attr in weight_attrs:
            value = getattr(self, attr)
            if value < 0.0:
                raise ValueError(
                    f"Loss weight '{attr}' must be non-negative, got {value}. "
                    f"Check your config's loss section."
                )

    # ──────────────────────────────────────────────────────────────────────────
    # Main call — compute all losses
    # ──────────────────────────────────────────────────────────────────────────

    def __call__(
        self,
        generator_outputs: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
        discriminator: nn.Module,
        image_encoder: nn.Module,
        box_encoder: nn.Module,
        z_batch: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        r"""Compute all losses for both generator and discriminator.

        This is the primary entry point.  It unpacks the generator outputs
        and the batch, runs the discriminator on real and fake images, and
        computes every loss term defined in the ILGAN objective.

        Parameters
        ----------
        generator_outputs : dict
            Dictionary produced by ``ILGANGenerator.forward()``.  Must
            contain the following keys:

            - ``"image"``: ``torch.Tensor``, shape ``[B, 3, H, W]``,
              generated image in ``[-1, 1]``.
            - ``"boxes"``: ``torch.Tensor``, shape ``[B, N, 4]``, predicted
              bounding boxes in ``(cx, cy, w, h)`` format, normalised to
              ``[0, 1]``.
            - ``"class_logits"``: ``torch.Tensor``, shape ``[B, N, C]``,
              predicted class logits.
            - ``"confidences"``: ``torch.Tensor``, shape ``[B, N, 1]``,
              objectness scores in ``[0, 1]``.
            - ``"aux"``: ``dict``, auxiliary outputs containing:

                - ``"attention_maps"``: ``torch.Tensor``, shape
                  ``[B, N, H_attn, W_attn]``, slot attention distributions.
                - ``"skip_features"``: ``list[torch.Tensor]``, multi-resolution
                  feature maps from the content decoder.

        batch : dict
            Dictionary from the data loader.  Must contain the following
            keys:

            - ``"images"``: ``torch.Tensor``, shape ``[B, 3, H, W]``, real
              images in ``[-1, 1]``.
            - ``"boxes"``: ``torch.Tensor``, shape ``[B, N, 4]``, ground-truth
              bounding boxes in ``(cx, cy, w, h)`` format.
            - ``"labels"``: ``torch.Tensor``, shape ``[B, N]``, ground-truth
              class labels (integers).
            - ``"valid_mask"``: ``torch.Tensor``, shape ``[B, N]``, boolean
              mask indicating which boxes are valid (non-padded).

        discriminator : nn.Module
            The ``ImageDiscriminator`` module.  Its ``forward`` must return
            a tuple ``(local_scores, global_score)`` where ``local_scores``
            has shape ``[B, 1, H_grid, W_grid]`` and ``global_score`` has
            shape ``[B, 1]``.

        image_encoder : nn.Module
            The ``ImageFeatureEncoder`` module that maps images to the
            shared feature space.  Must accept a ``[B, 3, H, W]`` tensor
            and return a ``[B, proj_dim]`` tensor.

        box_encoder : nn.Module
            The ``BoxFeatureEncoder`` module that maps bounding boxes to
            the shared feature space.  Must accept ``(boxes, confidences,
            valid_mask)`` and return a ``[B, proj_dim]`` tensor.

        z_batch : torch.Tensor
            Batch of latent vectors, shape ``[B, latent_dim]``.  Used for
            the latent diversity loss.

        Returns
        -------
        dict of str -> torch.Tensor
            A flat dictionary containing **all** individual loss terms plus
            the two aggregate keys:

            - ``"total_g_loss"``: scalar — the sum of all generator losses
              (adversarial + box + class + confidence + collapse + consistency).
              This is the loss to minimise for the generator optimiser.
            - ``"total_d_loss"``: scalar — the sum of the discriminator
              WGAN loss and the gradient penalty.  This is the loss to
              minimise for the discriminator optimiser.

            Individual loss terms (all scalar):

            - ``"g_loss_adv"``: generator adversarial loss (weighted).
            - ``"d_loss_adv"``: discriminator WGAN loss (unweighted).
            - ``"gp_loss"``: gradient penalty (weighted by ``gp_weight``).
            - ``"gp_value"``: unweighted gradient penalty (for logging).
            - ``"giou_loss"``: GIoU loss (unweighted).
            - ``"l1_loss"``: smooth L1 loss (unweighted).
            - ``"box_loss"``: weighted box regression loss.
            - ``"class_loss"``: class cross-entropy loss (weighted).
            - ``"confidence_loss"``: confidence BCE loss (weighted).
            - ``"entropy"``: attention entropy loss (unweighted).
            - ``"repulsion"``: slot repulsion loss (unweighted).
            - ``"feature_diversity"``: feature diversity loss (unweighted).
            - ``"latent_diversity"``: latent diversity loss (unweighted).
            - ``"collapse_loss"``: weighted sum of all collapse losses.
            - ``"consistency_loss"``: weighted consistency loss.
            - ``"cosine_similarity"``: mean cosine similarity (for logging).

        Raises
        ------
        KeyError
            If ``generator_outputs`` or ``batch`` are missing required keys.
        RuntimeError
            If the discriminator does not return a tuple of two tensors.
        """
        # ──────────────────────────────────────────────────────────────────
        # 1. Unpack generator outputs
        # ──────────────────────────────────────────────────────────────────
        try:
            fake_images = generator_outputs["image"]
            pred_boxes = generator_outputs["boxes"]
            class_logits = generator_outputs["class_logits"]
            confidences = generator_outputs["confidences"]
            aux = generator_outputs["aux"]
            attention_maps = aux["attention_maps"]
            skip_features = aux["skip_features"]
        except KeyError as e:
            raise KeyError(
                f"generator_outputs is missing required key: {e}. "
                f"Expected keys: 'image', 'boxes', 'class_logits', "
                f"'confidences', 'aux' (with 'attention_maps' and "
                f"'skip_features')."
            ) from e

        # ──────────────────────────────────────────────────────────────────
        # 2. Unpack batch
        # ──────────────────────────────────────────────────────────────────
        try:
            real_images = batch["images"]
            target_boxes = batch["boxes"]
            target_labels = batch["labels"]
            valid_mask = batch["valid_mask"]
        except KeyError as e:
            raise KeyError(
                f"batch is missing required key: {e}. "
                f"Expected keys: 'images', 'boxes', 'labels', 'valid_mask'."
            ) from e

        # ──────────────────────────────────────────────────────────────────
        # 3. Compute adversarial losses
        # ──────────────────────────────────────────────────────────────────
        # Run discriminator on both real and fake images.
        # The discriminator returns (local_scores, global_score).
        real_scores_local, real_scores_global = discriminator(real_images)
        fake_scores_local, fake_scores_global = discriminator(fake_images)

        adv_losses = compute_adversarial_losses(
            discriminator=discriminator,
            real_images=real_images,
            fake_images=fake_images,
            real_scores_local=real_scores_local,
            real_scores_global=real_scores_global,
            fake_scores_local=fake_scores_local,
            fake_scores_global=fake_scores_global,
            lambda_gp=self.gp_weight,
            adv_weight=self.adv_weight,
            w_global=self.w_global,
        )

        # ──────────────────────────────────────────────────────────────────
        # 4. Compute box regression losses
        # ──────────────────────────────────────────────────────────────────
        box_losses = compute_box_losses(
            pred_boxes=pred_boxes,
            target_boxes=target_boxes,
            valid_mask=valid_mask,
            box_weight=self.box_weight,
        )

        cls_losses = class_loss(
            class_logits=class_logits,
            target_labels=target_labels,
            valid_mask=valid_mask,
        )

        conf_losses = confidence_loss(
            confidences=confidences.squeeze(-1),  # [B, N, 1] -> [B, N]
            target_confidence=valid_mask.to(confidences.dtype),
            valid_mask=valid_mask,
        )

        # ──────────────────────────────────────────────────────────────────
        # 5. Compute collapse prevention losses
        # ──────────────────────────────────────────────────────────────────
        collapse_losses = compute_collapse_losses(
            attention_maps=attention_maps,
            skip_features=skip_features,
            z_batch=z_batch,
            diversity_weight=self.diversity_weight,
            entropy_weight=self.entropy_weight,
            repulsion_weight=self.repulsion_weight,
            latent_diversity_weight=self.latent_diversity_weight,
            repulsion_threshold=self.repulsion_threshold,
        )

        # ──────────────────────────────────────────────────────────────────
        # 6. Compute consistency loss
        # ──────────────────────────────────────────────────────────────────
        # The confidence loss expects [B, N] but box_encoder expects [B, N, 1]
        confidences_for_encoder = confidences  # already [B, N, 1]

        consistency_losses = compute_consistency_loss(
            generated_images=fake_images,
            predicted_boxes=pred_boxes,
            confidences=confidences_for_encoder,
            valid_mask=valid_mask,
            image_encoder=image_encoder,
            box_encoder=box_encoder,
            consistency_weight=self.consistency_weight,
        )

        # ──────────────────────────────────────────────────────────────────
        # 7. Assemble the full loss dictionary
        # ──────────────────────────────────────────────────────────────────
        losses: Dict[str, torch.Tensor] = {}

        # Adversarial
        losses["g_loss_adv"] = adv_losses["g_loss"]
        losses["d_loss_adv"] = adv_losses["d_loss"]
        losses["gp_loss"] = adv_losses["gp_loss"]
        losses["gp_value"] = adv_losses["gp_value"]

        # Box regression
        losses["giou_loss"] = box_losses["giou_loss"]
        losses["l1_loss"] = box_losses["l1_loss"]
        losses["box_loss"] = box_losses["box_loss"]

        # Class and confidence
        losses["class_loss"] = self.class_weight * cls_losses["class_loss"]
        losses["confidence_loss"] = self.confidence_weight * conf_losses["confidence_loss"]

        # Collapse prevention
        losses["entropy"] = collapse_losses["entropy"]
        losses["repulsion"] = collapse_losses["repulsion"]
        losses["feature_diversity"] = collapse_losses["feature_diversity"]
        losses["latent_diversity"] = collapse_losses["latent_diversity"]
        losses["collapse_loss"] = collapse_losses["collapse_loss"]

        # Consistency
        losses["consistency_loss"] = consistency_losses["consistency_loss"]
        losses["cosine_similarity"] = consistency_losses["cosine_similarity"]

        # ──────────────────────────────────────────────────────────────────
        # 8. Compute total losses
        # ──────────────────────────────────────────────────────────────────
        # Generator total: adversarial + box + class + confidence + collapse + consistency
        total_g_loss = (
            losses["g_loss_adv"]
            + losses["box_loss"]
            + losses["class_loss"]
            + losses["confidence_loss"]
            + losses["collapse_loss"]
            + losses["consistency_loss"]
        )
        losses["total_g_loss"] = total_g_loss

        # Discriminator total: WGAN D loss + gradient penalty
        total_d_loss = adv_losses["d_loss"]
        losses["total_d_loss"] = total_d_loss

        return losses

    # ──────────────────────────────────────────────────────────────────────────
    # Generator loss — convenience method
    # ──────────────────────────────────────────────────────────────────────────

    def generator_loss(
        self,
        generator_outputs: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
        discriminator: nn.Module,
        image_encoder: nn.Module,
        box_encoder: nn.Module,
        z_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Compute only the generator's total loss.

        This is a convenience wrapper around ``__call__`` that returns just
        the scalar ``"total_g_loss"``, suitable for the generator optimiser
        step.

        Parameters
        ----------
        generator_outputs : dict
            Generator outputs (see :meth:`__call__` for details).
        batch : dict
            Batch from the data loader (see :meth:`__call__` for details).
        discriminator : nn.Module
            The ``ImageDiscriminator`` module.
        image_encoder : nn.Module
            The ``ImageFeatureEncoder`` module.
        box_encoder : nn.Module
            The ``BoxFeatureEncoder`` module.
        z_batch : torch.Tensor
            Batch of latent vectors, shape ``[B, latent_dim]``.

        Returns
        -------
        torch.Tensor
            Scalar generator loss (0-dimensional).  This is the loss to
            backpropagate through the generator.

        Example
        -------
        >>> agg = LossAggregator(cfg)
        >>> g_loss = agg.generator_loss(gen_out, batch, disc, img_enc, box_enc, z)
        >>> g_loss.backward()  # generator backward pass
        """
        losses = self.__call__(
            generator_outputs=generator_outputs,
            batch=batch,
            discriminator=discriminator,
            image_encoder=image_encoder,
            box_encoder=box_encoder,
            z_batch=z_batch,
        )
        return losses["total_g_loss"]

    # ──────────────────────────────────────────────────────────────────────────
    # Discriminator loss — convenience method
    # ──────────────────────────────────────────────────────────────────────────

    def discriminator_loss(
        self,
        generator_outputs: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
        discriminator: nn.Module,
    ) -> torch.Tensor:
        """Compute only the discriminator's total loss.

        This is a convenience wrapper that computes the discriminator loss
        **without** needing the encoders or latent vectors (which are not
        required for the discriminator step).  It runs the discriminator on
        real and fake images and returns the WGAN discriminator loss plus
        gradient penalty.

        Parameters
        ----------
        generator_outputs : dict
            Generator outputs.  Only ``generator_outputs["image"]`` is used
            (the fake images).  Other keys are ignored.
        batch : dict
            Batch from the data loader.  Only ``batch["images"]`` is used
            (the real images).  Other keys are ignored.
        discriminator : nn.Module
            The ``ImageDiscriminator`` module.

        Returns
        -------
        torch.Tensor
            Scalar discriminator loss (0-dimensional).  This is the loss to
            backpropagate through the discriminator.

        Example
        -------
        >>> agg = LossAggregator(cfg)
        >>> d_loss = agg.discriminator_loss(gen_out, batch, disc)
        >>> d_loss.backward()  # discriminator backward pass
        """
        # Unpack only what we need
        # Detach fake_images so the discriminator backward does not
        # backpropagate into the generator.  The generator loss is
        # computed separately via generator_loss().
        fake_images = generator_outputs["image"].detach()
        real_images = batch["images"]

        # Run discriminator
        real_scores_local, real_scores_global = discriminator(real_images)
        fake_scores_local, fake_scores_global = discriminator(fake_images)

        # Compute adversarial losses (we only need the D-side)
        adv_losses = compute_adversarial_losses(
            discriminator=discriminator,
            real_images=real_images,
            fake_images=fake_images,
            real_scores_local=real_scores_local,
            real_scores_global=real_scores_global,
            fake_scores_local=fake_scores_local,
            fake_scores_global=fake_scores_global,
            lambda_gp=self.gp_weight,
            adv_weight=self.adv_weight,
            w_global=self.w_global,
        )

        return adv_losses["d_loss"]

    # ──────────────────────────────────────────────────────────────────────────
    # Representation
    # ──────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"LossAggregator(\n"
            f"  adv_weight={self.adv_weight},\n"
            f"  gp_weight={self.gp_weight},\n"
            f"  w_global={self.w_global},\n"
            f"  box_weight={self.box_weight},\n"
            f"  class_weight={self.class_weight},\n"
            f"  confidence_weight={self.confidence_weight},\n"
            f"  entropy_weight={self.entropy_weight},\n"
            f"  repulsion_weight={self.repulsion_weight},\n"
            f"  diversity_weight={self.diversity_weight},\n"
            f"  latent_diversity_weight={self.latent_diversity_weight},\n"
            f"  consistency_weight={self.consistency_weight},\n"
            f"  repulsion_threshold={self.repulsion_threshold},\n"
            f")"
        )
