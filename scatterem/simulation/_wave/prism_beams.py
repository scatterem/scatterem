"""Small pure helpers shared by the PRISM-EELS scattering matrices.

The interpolation-factor pair, the |z| of a complex tensor, the parent-beam
selection and the natural-neighbor partition weights: four functions with no
state, used by ``prism_eels`` (both S1 matrices), ``prism_eels_image`` and
``_detector_exit``. Split out of ``prism_eels.py`` unchanged.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

# Memoizes the natural-neighbor weights per (parents, children, device) key:
# the Delaunay triangulation behind them is pure geometry, so it is computed
# once per partition rather than once per slice.
_PARTITION_WEIGHT_CACHE: dict = {}


def _factor_pair(f) -> tuple:
    """Normalise an interpolation factor to a ``(fy, fx)`` int pair."""
    if isinstance(f, int):
        return (f, f)
    return (int(f[0]), int(f[1]))


def _abs_complex(z: Tensor) -> Tensor:
    """``z.abs()`` for a complex tensor, without ATen's full-size complex temporary.

    ATen routes complex->float unary ops through ``unary_op_impl_with_complex_to_float``:
    it runs the kernel **complex -> complex** into a freshly allocated full-size complex
    temporary and then copies that temporary's real part into the float result.  That is
    28 bytes moved per element (read 8, write 8, read 8, write 4) where the answer needs
    12 (read 8, write 4).  ``torch.hypot(z.real, z.imag)`` is **bit-identical** -- the
    complex ``abs_kernel_cuda`` is ``thrust::abs``, i.e. ``hypotf``, the same function
    ``hypot`` reaches -- and moves only the 12.  That argument is CUDA's: on the CPU
    complex128 ``abs`` and ``hypot`` disagree by one ulp on ~0.6 % of inputs, so the
    CPU keeps ``.abs()``.

    Falls back to ``.abs()`` whenever the result would be recorded by autograd: the
    drop-in replaces ``AbsBackward0`` (``grad * self.sgn()``) with ``HypotBackward0`` on
    two real views, which is different arithmetic.  Every caller below runs forward-only.
    """
    if (
        not z.is_complex()
        or z.device.type != "cuda"
        or (torch.is_grad_enabled() and z.requires_grad)
    ):
        return z.abs()
    return torch.hypot(z.real, z.imag)


def _select_parent_beams(
    all_signed: np.ndarray, n_radial: int, n_angular: int
) -> np.ndarray:
    """Hex-ring parent-beam selection (mirrors ``SMeta.make_beamlet_meta``).

    ``all_signed`` are the aperture beams in SIGNED spatial frequency (pixels).
    Samples the aperture on the DC beam plus ``n_radial`` concentric hex rings,
    snaps each sample to the nearest aperture beam, and deduplicates.  Returns
    indices into ``all_signed``.
    """
    radius = float(np.linalg.norm(all_signed, axis=1).max())
    a_off = np.pi / n_angular
    samples = [[0.0, 0.0]]
    for i, r in enumerate(np.linspace(0.0, radius, n_radial + 1)[1:]):
        n_ang = n_angular * (1 + i)
        for a in np.linspace(-np.pi, np.pi, n_ang, endpoint=False):
            samples.append([r * np.sin(a + a_off * i), r * np.cos(a + a_off * i)])
    samples = np.asarray(samples, dtype=np.float64)
    d = np.linalg.norm(samples[:, None, :] - all_signed[None, :, :], axis=2)
    return np.unique(np.argmin(d, axis=1))


def _partition_weights(
    beams_np: np.ndarray, all_signed: np.ndarray, n_radial: int, n_angular: int
):
    """Parent indices + natural-neighbor weights for an aperture geometry.

    Cached on the aperture beam set + ``(n_radial, n_angular)``: the geometry (and
    hence the parents and the NNW weights) is independent of the probe phase
    (defocus), the transmissions, and the sample thickness, so the expensive
    Delaunay/NNW computation runs once per aperture and is reused across every
    subsequent :class:`PartitionedScatteringMatrix` with the same aperture.
    Returns ``(parent_indices, w)`` with ``w`` of shape ``(B, Bp)``.
    """
    from scatterem.utils.stem import natural_neighbor_weights

    key = (beams_np.shape, beams_np.tobytes(), int(n_radial), int(n_angular))
    cached = _PARTITION_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached
    pidx = _select_parent_beams(all_signed, int(n_radial), int(n_angular))
    # Default is the fast vectorised Delaunay-barycentric weights ("linear").
    w = natural_neighbor_weights(all_signed[pidx], all_signed)  # (B, Bp)
    _PARTITION_WEIGHT_CACHE[key] = (pidx, w)
    return pidx, w


def _checked_factor(interpolation_factor, ny: int, nx: int) -> tuple[int, int]:
    """``(f_y, f_x)`` validated against the grid it will subsample.

    Both PRISM scattering matrices ran this check inline with the same two
    conditions and slightly different wording; it lives here once so they cannot
    drift apart.
    """
    fy, fx = int(interpolation_factor[0]), int(interpolation_factor[1])
    if fy < 1 or fx < 1:
        raise ValueError(f"interpolation_factor must be >= 1, got ({fy}, {fx})")
    if ny % fy or nx % fx:
        raise ValueError(
            f"interpolation_factor {(fy, fx)} must divide the grid "
            f"{(ny, nx)} (subsampling needs a sublattice)"
        )
    return fy, fx
