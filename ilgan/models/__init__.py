"""
ILGAN model modules.

The ``ilgan.models`` package contains the neural network building blocks
for the ILGAN dual-output GAN.  The centrepiece is the
``SpatialContentCrossAttention`` (SCCA) module that binds image generation
to bounding box prediction, and the ``ContentDecoder`` that transforms a
latent vector into a full-resolution image with multi-resolution skip features.
The ``SpatialHead`` consumes these skip features to produce bounding box
proposals through coarse-to-fine cross-attention refinement.  The
``ILGANGenerator`` composes ContentDecoder and SpatialHead into a single
unified generator that produces both images and bounding boxes in one shot.

The ``ImageDiscriminator`` is a PatchGAN-style discriminator that evaluates
the realism of generated or real images, producing both a spatial grid of
local realism scores and a single global realism score per sample.
"""

from ilgan.models.attention import SpatialContentCrossAttention
from ilgan.models.discriminator import (
    DownBlock,
    FrozenBatchNorm2d,
    ImageDiscriminator,
    minibatch_stddev,
)
from ilgan.models.generator import (
    ContentDecoder,
    FeatureProjector,
    ILGANGenerator,
    SlotMLP,
    SpatialHead,
    SpectralNormConv2d,
    UpBlock,
)

__all__ = [
    "SpatialContentCrossAttention",
    "ContentDecoder",
    "FeatureProjector",
    "ILGANGenerator",
    "SlotMLP",
    "SpatialHead",
    "SpectralNormConv2d",
    "UpBlock",
    "FrozenBatchNorm2d",
    "DownBlock",
    "ImageDiscriminator",
    "minibatch_stddev",
]