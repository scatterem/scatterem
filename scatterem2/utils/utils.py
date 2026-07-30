"""Module for various convenient utilities."""

from __future__ import annotations

import copy
import gc
import inspect
import os
from importlib.util import find_spec
from typing import Any, Optional, Tuple, TypeVar

import matplotlib.pyplot as plt
from scatterem2.vis.visualization import show_2d_array
import numpy as np
import pywt
import torch
from numpy.typing import NDArray

if find_spec("cv2") is not None:
    import cv2

T = TypeVar("T", float, int, bool)

def radial_average2(corner_centered_ctf,sampling):
    """ returns the radially-averaged CTF of a corner-centered 2D CTF array. """
    nx, ny = corner_centered_ctf.shape
    sx, sy = sampling
    
    kx = np.fft.fftfreq(nx,sx)
    ky = np.fft.fftfreq(ny,sy)
    k  = np.sqrt(kx[:,None]**2 + ky[None,:]**2).ravel()

    intensity = corner_centered_ctf.ravel()

    bin_size = kx[1]-kx[0]
    k_bins = np.arange(0, k.max() + bin_size, bin_size)

    inds = k / bin_size
    inds_f = np.floor(inds).astype("int")
    d_ind = inds - inds_f

    nf = np.bincount(inds_f, weights=(1 - d_ind), minlength=k_bins.shape[0])
    nc = np.bincount(inds_f + 1, weights=(d_ind), minlength=k_bins.shape[0])
    n = nf + nc

    I_bins0 = np.bincount(
        inds_f, weights=intensity * (1 - d_ind), minlength=k_bins.shape[0]
    )
    I_bins1 = np.bincount(
        inds_f + 1, weights=intensity * (d_ind), minlength=k_bins.shape[0]
    )

    I_bins = (I_bins0 + I_bins1) / n

    inds = k_bins <= np.abs(kx).max()

    return k_bins[inds], I_bins[inds]

def fuse_images_fourier_weighted(
    im1: torch.Tensor, 
    im2: torch.Tensor, 
    weight1: torch.Tensor,
    weight2: torch.Tensor
) -> torch.Tensor:
    """
    Fuse two images by fourier filtering im2 with weight2 and adding it to im1 with weight1.
    Args:
        im1: torch.Tensor, first image
        im2: torch.Tensor, second image
        weight2: torch.Tensor, weight for the second image
    Returns:
        fused: torch.Tensor, fused image
    """
    print(f"Max weight1: {weight1.max().item():.4f}, min weight1: {weight1.min().item():.4f}")
    print(f"Max weight2: {weight2.max().item():.4f}, min weight2: {weight2.min().item():.4f}")
    im1 -= im1.min()
    im1_max = im1.max()
    im1 /= im1_max

    im2 -= im2.min()
    im2_max = im2.max()
    im2 /= im2_max

    im1_fft = torch.fft.fft2(im1, dim=(0, 1), norm="ortho")
    im2_fft = torch.fft.fft2(im2, dim=(0, 1), norm="ortho")
    im1_fft_weighted = im1_fft * weight1
    im2_fft_weighted = im2_fft * weight2
    im_fused_fft = im1_fft_weighted + im2_fft_weighted

    im_fused = torch.fft.ifft2(im_fused_fft, dim=(0, 1), norm="ortho").real
    ptycho_filter = torch.fft.ifft2(im1_fft_weighted, dim=(0, 1), norm="ortho").real
    tcdf_filter = torch.fft.ifft2(im2_fft_weighted, dim=(0, 1), norm="ortho").real
    im_fused *= im1_max
    return im_fused, ptycho_filter, tcdf_filter


def fuse_images_dwt(im1: torch.Tensor, im2: torch.Tensor) -> torch.Tensor:
    """
    Fuse two images by wavelet transforming them and averaging the approximation coefficients.
    Args:
        im1: torch.Tensor, first image
        im2: torch.Tensor, second image
    Returns:
        fused: torch.Tensor, fused image
    """
    im1 -= im1.min()
    im1 /= im1.max()

    im2 -= im2.min()
    im2 /= im2.max()

    wavelet = "db1"  # Daubechies 1 (Haar-like)
    coeffs_phase = pywt.dwt2(im1.cpu().numpy(), wavelet)
    coeffs_df = pywt.dwt2(im2.cpu().numpy(), wavelet)

    cA_p, (cH_p, cV_p, cD_p) = coeffs_phase
    cA_d, (cH_d, cV_d, cD_d) = coeffs_df

    # show_2d_array(cA_p, title="Approximation Coefficients - Phase Image")
    # show_2d_array(cA_d, title="Approximation Coefficients - DF Image")

    # --- Fusion Rule: Average approximation, max detail coefficients ---
    fused_cA = (cA_p + cA_d) / 2
    fused_cH = np.maximum(cH_p, cH_d)
    fused_cV = np.maximum(cV_p, cV_d)
    fused_cD = np.maximum(cD_p, cD_d)

    # --- Reconstruct fused image ---
    fused = pywt.idwt2((fused_cA, (fused_cH, fused_cV, fused_cD)), wavelet)
    fused = np.clip(fused, 0, 1)

    return torch.from_numpy(fused), torch.from_numpy(cA_p), torch.from_numpy(cA_d)


def aggressive_cleanup():
    """Aggressive memory cleanup"""
    # Force garbage collection multiple times
    for _ in range(3):
        gc.collect()

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()

    # Force Python to release memory back to OS (Linux only)
    if hasattr(os, "sync"):
        os.sync()


def compose_affine_matrix(
    scale: float, asymmetry: float, rotation: float, shear: float, device: torch.device
) -> torch.Tensor:
    """
    asymmetry introduces unqual scaling in x and y direction. asymmetry > 0 streches x, asymmetry < 0 stretches y.
    rotation is in degrees, positive is counter-clockwise.
    shear is in degrees, positive is counter-clockwise.
    """
    rotation_rad = torch.as_tensor(rotation) * (torch.pi / 180.0)
    shear_rad = torch.as_tensor(shear) * (torch.pi / 180.0)

    A1 = torch.as_tensor([[scale, 0], [0, scale]])
    A2 = torch.as_tensor([[1 + asymmetry / 2, 0], [0, 1 - asymmetry / 2]])
    A3 = torch.as_tensor(
        [
            [torch.cos(rotation_rad), torch.sin(rotation_rad)],
            [-torch.sin(rotation_rad), torch.cos(rotation_rad)],
        ]
    )
    A4 = torch.as_tensor([[1, 0], [torch.tan(shear_rad), 1]])

    affine_mat = A1 @ A2 @ A3 @ A4

    return affine_mat.to(device)


def number_to_tuple(
    value: T | tuple[T, ...], dimension: Optional[int] = None
) -> tuple[T, ...]:
    """Normalise a scalar-or-sequence to a tuple of length dimension.

    A scalar is repeated; a sequence is passed through once its length is
    confirmed. dimension=None means "whatever length was given", so a scalar
    becomes a 1-tuple.

    Raises
    ------
    ValueError
        If a sequence is given whose length is not dimension. The previous
        implementation used a bare assert here, which vanishes under -O.
    """
    if isinstance(value, (int, float, bool)):
        return (value,) if dimension is None else (value,) * dimension

    values = tuple(value)
    if dimension is not None and len(values) != dimension:
        raise ValueError(
            f"expected {dimension} values, got {len(values)}: {values!r}"
        )
    return values


class CopyMixin:
    _exclude_from_copy: tuple = ()

    @staticmethod
    def _arg_keys(cls: type) -> tuple[str, ...]:
        parameters = inspect.signature(cls).parameters
        return tuple(
            key
            for key, value in parameters.items()
            if value.kind not in (value.VAR_POSITIONAL, value.VAR_KEYWORD)
        )

    def _copy_kwargs(
        self, exclude: tuple[str, ...] = (), cls: type | None = None
    ) -> dict[str, Any]:
        if cls is None:
            cls = self.__class__

        exclude = self._exclude_from_copy + exclude
        keys = [key for key in self._arg_keys(cls) if key not in exclude]
        kwargs = {key: copy.deepcopy(getattr(self, key)) for key in keys}
        return kwargs

    def copy(self) -> "CopyMixin":
        """Make a copy."""
        return copy.deepcopy(self)


def get_dtype(complex: bool = False) -> torch.dtype:
    """The working floating-point dtype, real or complex.

    Single precision throughout. The previous version dispatched over a
    ``precision`` config setting whose lookup was commented out, so only the
    float32 branches were ever reachable; this states that outright instead of
    keeping four dead branches and a docstring that said "numpy dtype" while
    returning a torch one.

    Parameters
    ----------
    complex : bool, optional
        Return the complex dtype of the same precision.
    """
    return torch.complex64 if complex else torch.float32

def detect_edges(mask: np.ndarray, method: str = "canny") -> np.ndarray:
    """Detect edges in binary mask using specified method."""
    if method == "canny":
        # Use Canny edge detection if cv2 is available, otherwise fallback to gradient
        if find_spec("cv2") is not None:
            import cv2

            edges = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
        else:
            # Fallback to gradient method when cv2 is not available
            gy, gx = np.gradient(mask.astype(np.float32))
            edges = np.sqrt(gx**2 + gy**2)
            edges = (edges > 0.1).astype(np.uint8)
    elif method == "sobel":
        # Use Sobel edge detection if cv2 is available, otherwise fallback to gradient
        if find_spec("cv2") is not None:
            import cv2

            sobelx = cv2.Sobel(mask.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(mask.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            edges = (edges > 0.1).astype(np.uint8)
        else:
            # Fallback to gradient method when cv2 is not available
            gy, gx = np.gradient(mask.astype(np.float32))
            edges = np.sqrt(gx**2 + gy**2)
            edges = (edges > 0.1).astype(np.uint8)
    elif method == "gradient":
        # Use simple gradient method
        gy, gx = np.gradient(mask.astype(np.float32))
        edges = np.sqrt(gx**2 + gy**2)
        edges = (edges > 0.1).astype(np.uint8)
    else:
        raise ValueError(f"Unknown edge method: {method}")

    return edges


def fit_circle_ransac(
    points: np.ndarray, iterations: int = 1000, threshold: float = 2.0
) -> Tuple[float, float, float]:
    """
    Fit circle to points using RANSAC for robustness to outliers.

    Args:
        points: Nx2 array of [y, x] coordinates
        iterations: Number of RANSAC iterations
        threshold: Distance threshold for inliers

    Returns:
        (cy, cx, r): Circle center and radius, or None if failed
    """
    if len(points) < 3:
        return None

    best_circle = None
    best_inliers = 0

    for _ in range(iterations):
        # Randomly sample 3 points
        sample_idx = np.random.choice(len(points), 3, replace=False)
        sample_points = points[sample_idx]

        # Fit circle to 3 points
        circle = fit_circle_3points(sample_points)
        if circle is None:
            continue

        cy, cx, r = circle

        # Count inliers
        distances = np.abs(
            np.sqrt((points[:, 0] - cy) ** 2 + (points[:, 1] - cx) ** 2) - r
        )
        inliers = distances < threshold
        num_inliers = np.sum(inliers)

        if num_inliers > best_inliers:
            best_inliers = num_inliers
            best_circle = circle

    return best_circle


def fit_circle_3points(points: np.ndarray) -> Tuple[float, float, float]:
    """Fit circle through 3 points using algebraic method."""
    if len(points) != 3:
        return None

    y1, x1 = points[0]
    y2, x2 = points[1]
    y3, x3 = points[2]

    # Check for collinear points
    det = (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2)
    if abs(det) < 1e-10:
        return None

    # Calculate circle center
    a = x1**2 + y1**2
    b = x2**2 + y2**2
    c = x3**2 + y3**2

    cx = (a * (y2 - y3) + b * (y3 - y1) + c * (y1 - y2)) / (2 * det)
    cy = (a * (x3 - x2) + b * (x1 - x3) + c * (x2 - x1)) / (2 * det)

    # Calculate radius
    r = np.sqrt((cx - x1) ** 2 + (cy - y1) ** 2)

    return cy, cx, r


def select_best_circle(candidates: list, DP: np.ndarray) -> Tuple[float, float, float]:
    """Select best circle from candidates based on multiple criteria."""
    if len(candidates) == 1:
        return candidates[0][:3]

    # Score each candidate
    scores = []
    for cy, cx, r, thresh, idx in candidates:
        # Score based on: radius consistency, center consistency, edge strength
        score = 0

        # Radius consistency (prefer radii that appear frequently)
        radii = [c[2] for c in candidates]
        radius_std = np.std(radii)
        if radius_std > 0:
            score += 1 / (1 + abs(r - np.median(radii)) / radius_std)

        # Center consistency
        centers = [(c[0], c[1]) for c in candidates]
        center_distances = [
            np.sqrt((cy - c[0]) ** 2 + (cx - c[1]) ** 2) for c in centers
        ]
        score += 1 / (1 + np.mean(center_distances))

        # Edge strength at circle boundary
        angles = np.linspace(0, 2 * np.pi, 100)
        boundary_y = cy + r * np.cos(angles)
        boundary_x = cx + r * np.sin(angles)

        # Check if boundary points are within image
        h, w = DP.shape
        valid_mask = (
            (boundary_y >= 0) & (boundary_y < h) & (boundary_x >= 0) & (boundary_x < w)
        )

        if np.sum(valid_mask) > 0:
            boundary_y = boundary_y[valid_mask]
            boundary_x = boundary_x[valid_mask]

            # Sample boundary intensities
            boundary_intensities = DP[boundary_y.astype(int), boundary_x.astype(int)]
            score += np.std(boundary_intensities)  # Higher variation = better edge

        scores.append(score)

    # Return candidate with highest score
    best_idx = np.argmax(scores)
    return candidates[best_idx][:3]


def refine_circle_fit(
    initial_circle: Tuple[float, float, float], DP: np.ndarray, edge_method: str
) -> Tuple[float, float, float]:
    """Refine circle fit using least-squares on edge points near initial circle."""
    cy, cx, r = initial_circle

    # Create mask around initial circle
    h, w = DP.shape
    y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    distances = np.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2)

    # Focus on region near the circle boundary
    ring_mask = (distances > r * 0.8) & (distances < r * 1.2)

    if not np.any(ring_mask):
        return cy, cx, r

    # Detect edges in the ring region
    local_region = DP * ring_mask
    edges = detect_edges((local_region > 0.1).astype(np.uint8), edge_method)

    edge_points = np.column_stack(np.where(edges))

    if len(edge_points) < 10:
        return cy, cx, r

    # Least-squares circle fitting
    refined_circle = fit_circle_least_squares(edge_points, initial_guess=(cy, cx, r))

    if refined_circle is not None:
        return refined_circle
    else:
        return cy, cx, r


def fit_circle_least_squares(
    points: np.ndarray, initial_guess: Tuple[float, float, float] = None
) -> Tuple[float, float, float]:
    """Fit circle using least-squares optimization."""
    from scipy.optimize import least_squares

    def residuals(params, points):
        cy, cx, r = params
        distances = np.sqrt((points[:, 0] - cy) ** 2 + (points[:, 1] - cx) ** 2)
        return distances - r

    if initial_guess is None:
        # Use centroid as initial guess
        cy_init = np.mean(points[:, 0])
        cx_init = np.mean(points[:, 1])
        r_init = np.mean(
            np.sqrt((points[:, 0] - cy_init) ** 2 + (points[:, 1] - cx_init) ** 2)
        )
        initial_guess = (cy_init, cx_init, r_init)

    result = least_squares(residuals, initial_guess, args=(points,))
    if result.success:
        return tuple(result.x)

    return None


