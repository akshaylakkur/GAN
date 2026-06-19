"""
Differentiable-friendly augmentation transforms for ILGAN.

All transforms operate on (image, boxes, labels, valid_mask) tuples and
preserve the spatial correspondence between image content and bounding box
coordinates.  Every transform is deterministic per-sample given a seed
derived from (sample_index, epoch), enabling reproducible augmentation
across training runs.

Available transforms
--------------------
- RandomHorizontalFlip
- RandomColorJitter
- RandomAffine
- Cutout
- Compose
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance

from ilgan.data.structures import Sample

# ──────────────────────────────────────────────────────────────────────────────
# Type aliases
# ──────────────────────────────────────────────────────────────────────────────

AugReturn = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
"""Return type of an augmentation's ``__call__``::

    (image, boxes, labels, valid_mask)

- image:   ``[3, H, W]``  in ``[-1, 1]``
- boxes:   ``[N, 4]``     in YOLO format ``[xc, yc, w, h]`` normalised ``[0, 1]``
- labels:  ``[N]``        integer class IDs
- valid_mask: ``[N]``     bool, ``True`` for real boxes
"""


# ──────────────────────────────────────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────────────────────────────────────


class Augmentation(ABC):
    """Abstract base for all ILGAN augmentations.

    Subclasses must implement ``forward(image, boxes, labels, valid_mask)``.
    The ``__call__`` entry point handles optional RNG seeding logic.
    """

    def __init__(self, p: float = 1.0) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Probability p must be in [0, 1], got {p}.")
        self.p = p

    @abstractmethod
    def forward(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> AugReturn:
        """Apply the augmentation with *guaranteed* application.

        Subclasses implement this method assuming the transform *will* be
        applied (the stochasticity from ``p`` is handled by ``__call__``).
        """
        ...

    def __call__(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
        rng_seed: Optional[int] = None,
    ) -> AugReturn:
        """Apply the augmentation (probabilistically if ``p < 1``).

        Parameters
        ----------
        image : torch.Tensor
            ``[3, H, W]`` in ``[-1, 1]``.
        boxes : torch.Tensor
            ``[N, 4]`` YOLO format.
        labels : torch.Tensor
            ``[N]`` class IDs.
        valid_mask : torch.Tensor
            ``[N]`` bool mask.
        rng_seed : int, optional
            Deterministic seed for reproducibility.  If ``None``, uses
            a random state derived from ``torch.random``.

        Returns
        -------
        AugReturn
            Augmented (image, boxes, labels, valid_mask).
        """
        if rng_seed is not None:
            state = torch.random.get_rng_state()
            torch.manual_seed(rng_seed)
            result = self._apply_probabilistic(image, boxes, labels, valid_mask)
            torch.random.set_rng_state(state)
            return result

        return self._apply_probabilistic(image, boxes, labels, valid_mask)

    def _apply_probabilistic(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> AugReturn:
        """Apply with probability ``self.p``, otherwise identity."""
        if self.p >= 1.0:
            return self.forward(image, boxes, labels, valid_mask)
        if self.p <= 0.0:
            return image, boxes, labels, valid_mask
        if torch.rand(1).item() < self.p:
            return self.forward(image, boxes, labels, valid_mask)
        return image, boxes, labels, valid_mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


# ──────────────────────────────────────────────────────────────────────────────
# RandomHorizontalFlip
# ──────────────────────────────────────────────────────────────────────────────


class RandomHorizontalFlip(Augmentation):
    """Flip the image horizontally with probability ``p``.

    Box coordinates are transformed correspondingly:
    ``x_center -> 1.0 - x_center`` while ``y_center``, ``width``, and
    ``height`` remain unchanged.
    """

    def forward(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> AugReturn:
        flipped = image.flip(-1)  # [3, H, W] → flip last dim (width)

        flipped_boxes = boxes.clone()
        if boxes.size(0) > 0:
            flipped_boxes[:, 0] = 1.0 - boxes[:, 0]

        return flipped, flipped_boxes, labels, valid_mask

    def __repr__(self) -> str:
        return f"RandomHorizontalFlip(p={self.p})"


# ──────────────────────────────────────────────────────────────────────────────
# RandomColorJitter
# ──────────────────────────────────────────────────────────────────────────────


class RandomColorJitter(Augmentation):
    """Apply random colour jitter to the image only (boxes unchanged).

    Internally uses PIL's ``ImageEnhance`` for brightness, contrast,
    saturation, and hue adjustments.  The image is converted to PIL,
    enhanced, and converted back to a tensor.

    Parameters
    ----------
    brightness : float
        Maximum absolute brightness shift.  Factor sampled uniformly in
        ``[1 - brightness, 1 + brightness]``.
    contrast : float
        Maximum absolute contrast shift.
    saturation : float
        Maximum absolute saturation shift.
    hue : float
        Maximum absolute hue shift (range ``[0, 0.5]``).
    p : float
        Probability of applying the jitter.
    """

    def __init__(
        self,
        brightness: float = 0.1,
        contrast: float = 0.1,
        saturation: float = 0.1,
        hue: float = 0.05,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = min(hue, 0.5)
        self._jitter_fn = _ColorJitterFn(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def forward(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> AugReturn:
        jittered = self._jitter_fn(image)
        return jittered, boxes, labels, valid_mask

    def __repr__(self) -> str:
        return (
            f"RandomColorJitter("
            f"brightness={self.brightness}, "
            f"contrast={self.contrast}, "
            f"saturation={self.saturation}, "
            f"hue={self.hue}, "
            f"p={self.p})"
        )


class _ColorJitterFn:
    """Stateless colour jitter implementation using PIL.

    All random choices use ``torch.rand`` / ``torch.randint`` so that
    ``torch.manual_seed`` provides full deterministic control.
    """

    def __init__(
        self,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
    ) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        # ── sample random factors ───────────────────────────────────────
        b_factor, c_factor, s_factor, h_factor = self._get_params()

        # ── tensor -> PIL ───────────────────────────────────────────────
        pil = _tensor_to_pil(image)

        # ── build operations list ───────────────────────────────────────
        ops: List[Tuple[int, object]] = []
        if b_factor is not None:
            ops.append((0, ImageEnhance.Brightness(pil)))
        if c_factor is not None:
            ops.append((1, ImageEnhance.Contrast(pil)))
        if s_factor is not None:
            ops.append((2, ImageEnhance.Color(pil)))
        if h_factor is not None:
            ops.append((3, h_factor))

        # ── shuffle with torch (deterministic under torch.manual_seed) ──
        if len(ops) > 1:
            perm = torch.randperm(len(ops)).tolist()
            ops = [ops[i] for i in perm]

        for op_type, enhancer_or_factor in ops:
            if op_type == 3:
                pil = _apply_hue_shift(pil, h_factor)  # type: ignore[arg-type]
            else:
                factor = [b_factor, c_factor, s_factor][op_type]
                pil = enhancer_or_factor.enhance(factor)  # type: ignore[union-attr]

        # ── PIL -> tensor ───────────────────────────────────────────────
        return _pil_to_tensor(pil)

    def _get_params(
        self,
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Return (brightness_factor, contrast_factor, saturation_factor, hue_factor)."""
        b: Optional[float] = None
        c: Optional[float] = None
        s: Optional[float] = None
        h: Optional[float] = None
        if self.brightness > 0.0:
            b = 1.0 + torch.empty(1).uniform_(-self.brightness, self.brightness).item()
        if self.contrast > 0.0:
            c = 1.0 + torch.empty(1).uniform_(-self.contrast, self.contrast).item()
        if self.saturation > 0.0:
            s = 1.0 + torch.empty(1).uniform_(-self.saturation, self.saturation).item()
        if self.hue > 0.0:
            h = torch.empty(1).uniform_(-self.hue, self.hue).item()
        return b, c, s, h


def _apply_hue_shift(pil_image: Image.Image, factor: float) -> Image.Image:
    """Apply hue shift by converting to HSV, shifting hue, converting back."""
    pil_hsv = pil_image.convert("HSV")
    arr = np.array(pil_hsv, dtype=np.float32)  # [H, W, 3]
    # Shift hue (channel 0) — wrap around at 180 (PIL HSV range for 8-bit)
    arr[:, :, 0] = (arr[:, :, 0] + factor * 180.0) % 180.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    hsv_pil = Image.fromarray(arr, mode="HSV")
    return hsv_pil.convert("RGB")


# ──────────────────────────────────────────────────────────────────────────────
# RandomAffine
# ──────────────────────────────────────────────────────────────────────────────


class RandomAffine(Augmentation):
    """Apply a random affine transformation to both image and boxes.

    The same affine matrix is used to:
    - Warp the image via ``torch.nn.functional.grid_sample``
      (fully differentiable).
    - Transform bounding box corners and recompute the YOLO-format
      coordinates.

    All operations are centred on the image centre (``0.5, 0.5`` in
    normalised ``[0, 1]`` coordinates).

    Parameters
    ----------
    degrees : float
        Maximum absolute rotation angle (degrees). Sampled uniformly in
        ``[-degrees, degrees]``.
    translate : float
        Maximum absolute translation as a fraction of image size.
        Sampled uniformly in ``[-translate, translate]`` independently for
        x and y.
    scale : tuple of (float, float)
        Range ``(min_scale, max_scale)`` for the isotropic scale factor.
    p : float
        Probability of applying the affine transform.
    """

    def __init__(
        self,
        degrees: float = 5.0,
        translate: float = 0.05,
        scale: Tuple[float, float] = (0.9, 1.1),
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        if scale[0] <= 0.0 or scale[1] < scale[0]:
            raise ValueError(
                f"scale must be (min, max) with min > 0, got {scale}."
            )

    def forward(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> AugReturn:
        # ── sample parameters ──────────────────────────────────────────
        angle_rad = math.radians(
            torch.empty(1).uniform_(-self.degrees, self.degrees).item()
        )
        s = torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
        tx = torch.empty(1).uniform_(-self.translate, self.translate).item()
        ty = torch.empty(1).uniform_(-self.translate, self.translate).item()

        *_, H, W = image.shape
        device = image.device
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # ── Build the inverse affine matrix for grid_sample ────────────
        #
        # Forward transform in normalised [0,1] coords (centered at 0.5):
        #   x' = s * cos(a) * (x - 0.5) - s * sin(a) * (y - 0.5) + 0.5 + tx
        #   y' = s * sin(a) * (x - 0.5) + s * cos(a) * (y - 0.5) + 0.5 + ty
        #
        # Inverse transform (for grid_sample):
        #   x  = cos(a)/s * (x' - 0.5 - tx) + sin(a)/s * (y' - 0.5 - ty) + 0.5
        #   y  = -sin(a)/s * (x' - 0.5 - tx) + cos(a)/s * (y' - 0.5 - ty) + 0.5
        #
        # In matrix form:
        #   [x]   = [cos(a)/s   sin(a)/s] [x' - 0.5 - tx]   [0.5]
        #   [y]     [-sin(a)/s  cos(a)/s] [y' - 0.5 - ty] + [0.5]
        #
        #   x = cos(a)/s * x' + sin(a)/s * y' + [0.5 - cos(a)/s*(0.5+tx) - sin(a)/s*(0.5+ty)]
        #   y = -sin(a)/s * x' + cos(a)/s * y' + [0.5 + sin(a)/s*(0.5+tx) - cos(a)/s*(0.5+ty)]
        #
        # grid_sample with align_corners=False uses grid coords in [-1,1].
        # A normalised [0,1] coord u maps to grid coord: g = 2*u - 1.
        # So we need to define theta such that:
        #   g_in = theta_00 * g_out_x + theta_01 * g_out_y + theta_02
        #
        #   x_in   = a11 * x_out + a12 * y_out + b1
        #   g_in_x = 2*x_in - 1 = 2*a11*x_out + 2*a12*y_out + 2*b1 - 1
        #   but x_out = (g_out_x + 1)/2, so:
        #   g_in_x = 2*a11*(g_out_x+1)/2 + 2*a12*(g_out_y+1)/2 + 2*b1 - 1
        #          = a11*g_out_x + a12*g_out_y + (a11 + a12 + 2*b1 - 1)

        a11 = cos_a / s
        a12 = sin_a / s
        a21 = -sin_a / s
        a22 = cos_a / s

        b1 = 0.5 - a11 * (0.5 + tx) - a12 * (0.5 + ty)
        b2 = 0.5 - a21 * (0.5 + tx) - a22 * (0.5 + ty)

        theta_00 = a11
        theta_01 = a12
        theta_02 = a11 + a12 + 2.0 * b1 - 1.0

        theta_10 = a21
        theta_11 = a22
        theta_12 = a21 + a22 + 2.0 * b2 - 1.0

        theta = torch.tensor(
            [[theta_00, theta_01, theta_02],
             [theta_10, theta_11, theta_12]],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)  # [1, 2, 3]

        # ── warp image ──────────────────────────────────────────────────
        grid = F.affine_grid(theta, (1, 3, H, W), align_corners=False)
        warped = F.grid_sample(
            image.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(0)  # [3, H, W]

        # ── transform boxes ─────────────────────────────────────────────
        if boxes.size(0) > 0 and valid_mask.any():
            transformed_boxes = self._transform_boxes(
                boxes, valid_mask, angle_rad, s, tx, ty,
            )
        else:
            transformed_boxes = boxes.clone()

        return warped, transformed_boxes, labels, valid_mask

    @staticmethod
    def _transform_boxes(
        boxes: torch.Tensor,
        valid_mask: torch.Tensor,
        angle_rad: float,
        scale: float,
        tx: float,
        ty: float,
    ) -> torch.Tensor:
        """Apply the forward affine transform to YOLO boxes.

        All coordinates are in normalised ``[0, 1]`` space.  The forward
        transform is centred on ``(0.5, 0.5)``.

        Parameters
        ----------
        boxes : torch.Tensor
            ``[N, 4]`` YOLO format ``[xc, yc, w, h]`` in ``[0, 1]``.
        valid_mask : torch.Tensor
            ``[N]`` bool.
        angle_rad : float
            Rotation angle in radians.
        scale : float
            Isotropic scale factor.
        tx : float
            Translation x as fraction of image.
        ty : float
            Translation y as fraction of image.

        Returns
        -------
        torch.Tensor
            Transformed boxes, same shape, clamped to ``[0, 1]``.
            Degenerate boxes (flipped entirely out of view) are set to
            ``-1.0``.
        """
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        s = scale

        new_boxes = boxes.clone()
        valid_indices = torch.where(valid_mask)[0]

        for idx in valid_indices:
            xc, yc, w, h = boxes[idx].tolist()

            # Four corners in [0,1] coords
            x1 = xc - w / 2.0
            y1 = yc - h / 2.0
            x2 = xc + w / 2.0
            y2 = yc + h / 2.0

            corners = torch.tensor([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ], dtype=torch.float32)  # [4, 2]

            # Center at 0.5, apply forward transform, shift back
            corners_c = corners - 0.5  # [4, 2]

            x_new = s * cos_a * corners_c[:, 0] - s * sin_a * corners_c[:, 1] + 0.5 + tx
            y_new = s * sin_a * corners_c[:, 0] + s * cos_a * corners_c[:, 1] + 0.5 + ty

            # Compute axis-aligned bounding box and clamp to [0, 1]
            x1_new = x_new.min().clamp(0.0, 1.0).item()
            y1_new = y_new.min().clamp(0.0, 1.0).item()
            x2_new = x_new.max().clamp(0.0, 1.0).item()
            y2_new = y_new.max().clamp(0.0, 1.0).item()

            new_w = x2_new - x1_new
            new_h = y2_new - y1_new
            new_xc = (x1_new + x2_new) / 2.0
            new_yc = (y1_new + y2_new) / 2.0

            # Mark degenerate boxes (no visible area)
            if new_w <= 0.0 or new_h <= 0.0:
                new_boxes[idx] = torch.tensor([-1.0, -1.0, -1.0, -1.0])
            else:
                new_boxes[idx, 0] = new_xc
                new_boxes[idx, 1] = new_yc
                new_boxes[idx, 2] = new_w
                new_boxes[idx, 3] = new_h

        return new_boxes

    def __repr__(self) -> str:
        return (
            f"RandomAffine("
            f"degrees={self.degrees}, "
            f"translate={self.translate}, "
            f"scale={self.scale}, "
            f"p={self.p})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Cutout
# ──────────────────────────────────────────────────────────────────────────────


class Cutout(Augmentation):
    """Randomly mask out square regions of the image (boxes unchanged).

    The masked regions are filled with the ``fill_value`` (default 0.0,
    which corresponds to the mean of the ``[-1, 1]`` range).  The masked
    regions are recorded in the sample metadata for optional inpainting
    loss computation during training.

    Parameters
    ----------
    p : float
        Probability of applying cutout.
    max_holes : int
        Maximum number of square holes to cut.
    max_size : float
        Maximum side length of each hole as a fraction of image size.
    fill_value : float
        Value to fill the masked region with (default 0.0 = mean of ``[-1, 1]``).
    """

    def __init__(
        self,
        p: float = 0.5,
        max_holes: int = 3,
        max_size: float = 0.1,
        fill_value: float = 0.0,
    ) -> None:
        super().__init__(p=p)
        self.max_holes = max_holes
        self.max_size = max_size
        self.fill_value = fill_value

    def forward(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> AugReturn:
        *_, H, W = image.shape
        max_hole_px = int(self.max_size * min(H, W))
        if max_hole_px < 1:
            return image, boxes, labels, valid_mask

        num_holes = int(torch.randint(1, self.max_holes + 1, (1,)).item())
        output = image.clone()

        for _ in range(num_holes):
            hole_w = int(torch.randint(1, max_hole_px + 1, (1,)).item())
            hole_h = int(torch.randint(1, max_hole_px + 1, (1,)).item())

            hole_w = min(hole_w, W)
            hole_h = min(hole_h, H)

            x_start = int(torch.randint(0, W - hole_w + 1, (1,)).item())
            y_start = int(torch.randint(0, H - hole_h + 1, (1,)).item())

            output[:, y_start:y_start + hole_h, x_start:x_start + hole_w] = self.fill_value

        return output, boxes, labels, valid_mask

    def __repr__(self) -> str:
        return (
            f"Cutout("
            f"p={self.p}, "
            f"max_holes={self.max_holes}, "
            f"max_size={self.max_size})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Compose
# ──────────────────────────────────────────────────────────────────────────────


class Compose:
    """Chain multiple augmentations together, applied in sequence.

    Supports deterministic per-sample behaviour via ``rng_seed`` which
    is mixed with the transform index to produce unique sub-seeds.

    Parameters
    ----------
    transforms : list of Augmentation
        The augmentations to apply, in order.
    shuffle_order : bool
        If ``True``, the order of transforms is shuffled at each call
        (this consumes one sub-seed from the seed chain).
    """

    def __init__(
        self,
        transforms: Sequence[Augmentation],
        shuffle_order: bool = False,
    ) -> None:
        self.transforms = list(transforms)
        self._shuffle_order = shuffle_order

    def __call__(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
        rng_seed: Optional[int] = None,
    ) -> AugReturn:
        """Apply all transforms in sequence.

        A single ``rng_seed`` is used to derive per-transform seeds so that
        the entire pipeline is deterministic per sample.
        """
        transforms = self.transforms
        if self._shuffle_order and rng_seed is not None:
            # Deterministic shuffle based on seed
            rng_state = torch.random.get_rng_state()
            torch.manual_seed(rng_seed)
            perm = torch.randperm(len(transforms)).tolist()
            transforms = [self.transforms[i] for i in perm]
            torch.random.set_rng_state(rng_state)

        for t_idx, transform in enumerate(transforms):
            if rng_seed is not None:
                t_seed = _mix_seed(rng_seed, t_idx + 1)
            else:
                t_seed = None
            image, boxes, labels, valid_mask = transform(
                image, boxes, labels, valid_mask,
                rng_seed=t_seed,
            )
        return image, boxes, labels, valid_mask

    def apply_to_sample(
        self,
        sample: Sample,
        rng_seed: Optional[int] = None,
    ) -> Sample:
        """Apply the augmentation pipeline to a ``Sample`` directly."""
        image, boxes, labels, valid_mask = self(
            sample.image,
            sample.boxes,
            sample.labels,
            sample.valid_mask,
            rng_seed=rng_seed,
        )
        return Sample(
            image=image,
            boxes=boxes,
            labels=labels,
            valid_mask=valid_mask,
            image_path=sample.image_path,
            metadata={**sample.metadata, "augmented": True},
        )

    def __len__(self) -> int:
        return len(self.transforms)

    def __repr__(self) -> str:
        lines = [f"  {i}: {t!r}" for i, t in enumerate(self.transforms)]
        suffix = " (shuffled)" if self._shuffle_order else ""
        return f"Compose[{suffix}] (\n" + "\n".join(lines) + "\n)"

    def __getitem__(self, idx: int) -> Augmentation:
        return self.transforms[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Helper: tensor <-> PIL
# ──────────────────────────────────────────────────────────────────────────────


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a ``[3, H, W]`` tensor in ``[-1, 1]`` to a PIL RGB image."""
    arr = tensor.detach().cpu().numpy()          # [3, H, W]
    arr = ((arr + 1.0) * 127.5).clip(0, 255).astype("uint8")
    arr = arr.transpose(1, 2, 0)                 # [H, W, 3]
    return Image.fromarray(arr, mode="RGB")


def _pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """Convert a PIL RGB image back to a ``[3, H, W]`` tensor in ``[-1, 1]``."""
    arr = torch.tensor(
        np.array(pil_image, dtype=np.float32),
        dtype=torch.float32,                     # [H, W, 3]
    ).permute(2, 0, 1)                           # [3, H, W]
    return arr / 127.5 - 1.0                     # [0, 255] -> [-1, 1]


def _mix_seed(base_seed: int, index: int) -> int:
    """Derive a deterministic sub-seed from a base seed and an index.

    Uses a simple multiplicative hash to spread seeds across the 32-bit
    integer space, avoiding correlation between adjacent indices.
    """
    return (base_seed ^ (index * 0x9E3779B9)) & 0xFFFFFFFF


# ──────────────────────────────────────────────────────────────────────────────
# Factory: build default augmentation pipeline
# ──────────────────────────────────────────────────────────────────────────────


def build_default_augmentation_pipeline(
    hflip_prob: float = 0.5,
    color_jitter_prob: float = 0.8,
    affine_prob: float = 0.5,
    cutout_prob: float = 0.5,
    degrees: float = 5.0,
    translate: float = 0.05,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    brightness: float = 0.1,
    contrast: float = 0.1,
    saturation: float = 0.1,
    hue: float = 0.05,
    max_holes: int = 3,
    max_cutout_size: float = 0.1,
    shuffle_order: bool = True,
) -> Compose:
    """Build the default augmentation pipeline for ILGAN training.

    Transforms are applied in a randomly shuffled order per-sample when
    ``shuffle_order=True`` (default), increasing diversity.

    Returns
    -------
    Compose
        A composable augmentation pipeline ready to be called on
        ``(image, boxes, labels, valid_mask)`` tuples.
    """
    transforms: List[Augmentation] = [
        RandomHorizontalFlip(p=hflip_prob),
        RandomColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            p=color_jitter_prob,
        ),
        RandomAffine(
            degrees=degrees,
            translate=translate,
            scale=scale_range,
            p=affine_prob,
        ),
        Cutout(p=cutout_prob, max_holes=max_holes, max_size=max_cutout_size),
    ]

    return Compose(transforms, shuffle_order=shuffle_order)


__all__ = [
    "Augmentation",
    "RandomHorizontalFlip",
    "RandomColorJitter",
    "RandomAffine",
    "Cutout",
    "Compose",
    "build_default_augmentation_pipeline",
]