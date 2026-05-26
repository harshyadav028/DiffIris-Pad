"""Phase 1 — Corruption Functions.

Importable module of deterministic corruption functions used:
- During pair creation (prepare_bonafide_pairs.py)
- At inference time (iris_dataset.py)

All functions operate on numpy arrays (H, W, 3) uint8 [0-255].
Pipeline order: noise → blur → resolution degradation.
Determinism guaranteed via np.random.RandomState(seed).

Usage (import):
    from iris_bbdm_pad.data.corruption import apply_corruption_pipeline, filename_to_seed
"""

from __future__ import annotations

import hashlib
from typing import Dict, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Individual corruption functions
# ---------------------------------------------------------------------------

def add_gaussian_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    """Add Gaussian noise to an image.

    Args:
        image: Input image (H, W, 3) uint8 [0-255].
        sigma: Standard deviation of the Gaussian noise.

    Returns:
        Noisy image (H, W, 3) uint8 [0-255].
    """
    noise = np.random.normal(0.0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_gaussian_blur(image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    """Apply Gaussian blur to an image.

    Args:
        image: Input image (H, W, 3) uint8 [0-255].
        kernel_size: Kernel size (must be odd and positive).
        sigma: Standard deviation of the Gaussian kernel.

    Returns:
        Blurred image (H, W, 3) uint8 [0-255].
    """
    # Ensure kernel_size is odd
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def resolution_degradation(
    image: np.ndarray,
    scale_factor: float,
    target_size: int = 256,
) -> np.ndarray:
    """Degrade resolution by downscaling then upscaling back.

    Args:
        image: Input image (H, W, 3) uint8 [0-255].
        scale_factor: Downscale factor in (0, 1). Smaller = more degradation.
        target_size: Final output size (pixels, square).

    Returns:
        Degraded image (target_size, target_size, 3) uint8 [0-255].
    """
    h, w = image.shape[:2]
    small_h = max(4, int(h * scale_factor))
    small_w = max(4, int(w * scale_factor))

    # Downscale with area interpolation (like JPEG artifacts)
    downscaled = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)

    # Upscale back to target size with bilinear (blocky artifacts)
    upscaled = cv2.resize(downscaled, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    return upscaled


# ---------------------------------------------------------------------------
# Seed utility
# ---------------------------------------------------------------------------

def filename_to_seed(filename: str) -> int:
    """Convert a filename to a deterministic integer seed via MD5 hash.

    Args:
        filename: Any string (typically the image filename stem).

    Returns:
        Non-negative integer seed in [0, 2**31 - 1].
    """
    md5 = hashlib.md5(filename.encode("utf-8")).hexdigest()
    return int(md5[:8], 16) % (2**31 - 1)


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

def get_default_corruption_config() -> Dict:
    """Return default corruption parameter ranges as a JSON-serializable dict.

    Returns:
        Dict with keys: noise_sigma_range, blur_kernel_choices,
        blur_sigma_range, downscale_range, target_size.
    """
    return {
        "noise_sigma_range": [10.0, 40.0],
        "blur_kernel_choices": [3, 5, 7],
        "blur_sigma_range": [0.5, 2.0],
        "downscale_range": [0.25, 0.75],
        "target_size": 256,
    }


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------

def apply_corruption_pipeline(
    image: np.ndarray,
    seed: int,
    noise_sigma_range: Tuple[float, float] = (10.0, 40.0),
    blur_kernel_choices: Tuple[int, ...] = (3, 5, 7),
    blur_sigma_range: Tuple[float, float] = (0.5, 2.0),
    downscale_range: Tuple[float, float] = (0.25, 0.75),
    target_size: int = 256,
) -> np.ndarray:
    """Apply deterministic corruption pipeline: noise → blur → degradation.

    All randomness is seeded by ``seed`` so the same image always gets the
    same corruption applied.

    Args:
        image: Clean input image (H, W, 3) uint8 [0-255].
        seed: Deterministic random seed (use filename_to_seed()).
        noise_sigma_range: (min, max) range for Gaussian noise sigma.
        blur_kernel_choices: Tuple of odd kernel sizes to choose from.
        blur_sigma_range: (min, max) range for blur sigma.
        downscale_range: (min, max) downscale factor for resolution degradation.
        target_size: Output image spatial size (square).

    Returns:
        Corrupted image (target_size, target_size, 3) uint8 [0-255].
    """
    rng = np.random.RandomState(seed)

    # Sample corruption parameters
    noise_sigma = float(rng.uniform(noise_sigma_range[0], noise_sigma_range[1]))
    blur_kernel = int(rng.choice(list(blur_kernel_choices)))
    blur_sigma = float(rng.uniform(blur_sigma_range[0], blur_sigma_range[1]))
    scale_factor = float(rng.uniform(downscale_range[0], downscale_range[1]))

    # Ensure input is the right dtype
    img = image.astype(np.uint8)

    # Step 1: Gaussian noise
    img = add_gaussian_noise(img, noise_sigma)

    # Step 2: Gaussian blur
    img = add_gaussian_blur(img, blur_kernel, blur_sigma)

    # Step 3: Resolution degradation
    img = resolution_degradation(img, scale_factor, target_size=target_size)

    return img


def sample_corruption_params(seed: int, config: Dict) -> Dict:
    """Sample corruption parameters for a given seed (for inspection/logging).

    Args:
        seed: Deterministic random seed.
        config: Corruption config dict (from get_default_corruption_config()).

    Returns:
        Dict with sampled: noise_sigma, blur_kernel, blur_sigma, scale_factor.
    """
    rng = np.random.RandomState(seed)
    noise_sigma = float(rng.uniform(*config["noise_sigma_range"]))
    blur_kernel = int(rng.choice(config["blur_kernel_choices"]))
    blur_sigma = float(rng.uniform(*config["blur_sigma_range"]))
    scale_factor = float(rng.uniform(*config["downscale_range"]))
    return {
        "noise_sigma": noise_sigma,
        "blur_kernel": blur_kernel,
        "blur_sigma": blur_sigma,
        "scale_factor": scale_factor,
    }


# ---------------------------------------------------------------------------
# Intensity variants (for visualization only)
# ---------------------------------------------------------------------------

def apply_corruption_at_intensity(
    image: np.ndarray,
    intensity: float,
    seed: int,
    target_size: int = 256,
) -> np.ndarray:
    """Apply corruption at a specific intensity level (0.0=minimal, 1.0=maximal).

    Used for the corruption_intensity_spectrum visualization.

    Args:
        image: Clean input image (H, W, 3) uint8 [0-255].
        intensity: Value in [0.0, 1.0] mapping linearly across parameter ranges.
        seed: Deterministic seed (for blur kernel choice reproducibility).
        target_size: Output size.

    Returns:
        Corrupted image at the specified intensity.
    """
    rng = np.random.RandomState(seed)

    cfg = get_default_corruption_config()
    lo_n, hi_n = cfg["noise_sigma_range"]
    lo_b, hi_b = cfg["blur_sigma_range"]
    lo_d, hi_d = cfg["downscale_range"]

    noise_sigma = lo_n + intensity * (hi_n - lo_n)
    blur_sigma = lo_b + intensity * (hi_b - lo_b)
    scale_factor = hi_d - intensity * (hi_d - lo_d)  # higher intensity = smaller scale = more degradation
    blur_kernel = int(rng.choice(cfg["blur_kernel_choices"]))

    img = image.astype(np.uint8)
    img = add_gaussian_noise(img, noise_sigma)
    img = add_gaussian_blur(img, blur_kernel, blur_sigma)
    img = resolution_degradation(img, scale_factor, target_size=target_size)
    return img
