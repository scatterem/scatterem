"""Removal of the continuous ptychographic gauges from a mixed-state probe.

Ptychographic data ``|F{P O}|^2`` cannot distinguish (i) a common translation
of probe and object and (ii) a linear phase ramp ``e^{2 pi i q.r}`` moved from
the object to the probe. Before two reconstructions are compared these gauges
must be fixed; the functions here fix them by centring the real-space and the
aperture-plane intensity centroids.

Grid conventions: real-space modes are centred on the fftshift origin
``(H // 2, W // 2)``, and the aperture plane is
``A = fftshift(fft2(ifftshift(psi)))``
(:func:`scatterem.diagnostics.metrics.oam.to_aperture_plane`).
All leading dimensions of a stack are modes of one state. NumPy and torch
inputs are accepted; outputs follow the input.
"""

from typing import cast

import numpy as np
import torch as t

from scatterem.diagnostics.metrics.oam import to_aperture_plane
from scatterem.physics.fourier import fourier_shift
from scatterem.utils._array_compat import ArrayLike, as_complex, as_tensor, restore

__all__ = [
    "remove_phase_ramp_gauge",
    "remove_probe_gauges",
    "remove_translation_gauge",
]


def _coords(n: int, like: t.Tensor) -> t.Tensor:
    return t.arange(n, dtype=like.real.dtype, device=like.device)


def _centroid_offset(inten: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
    """Intensity centroid minus the fftshift origin ``(H // 2, W // 2)``, pixels."""
    h, w = inten.shape
    tot = inten.sum()
    cy = (_coords(h, inten)[:, None] * inten).sum() / tot
    cx = (_coords(w, inten)[None, :] * inten).sum() / tot
    return cy - h // 2, cx - w // 2


def remove_translation_gauge(fields: ArrayLike) -> ArrayLike:
    """Shift all modes together so the total-intensity centroid is at the centre.

    The centroid of ``sum_m |psi_m|^2`` (over all leading dims) is moved to the
    fftshift origin ``(H // 2, W // 2)`` by one common subpixel Fourier shift,
    which is cyclic and power-preserving. A common probe/object translation
    leaves ptychographic data invariant, so this pins that gauge; to keep the
    data invariant the object has to be shifted by the same amount.

    Parameters
    ----------
    fields : numpy.ndarray or torch.Tensor
        Complex real-space modes, shape ``(..., H, W)``.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Shifted modes, same shape and dtype family as ``fields``.
    """
    x, to_numpy = as_tensor(fields)
    x = as_complex(x)
    h, w = x.shape[-2:]
    oy, ox = _centroid_offset((x.abs() ** 2).reshape(-1, h, w).sum(0))
    out = fourier_shift(x, -t.stack((oy, ox)))  # tensor shift: no host sync
    return restore(out, to_numpy)


# Kept copy: a real-space modulation about the grid centre, not a Fourier shift,
# so it does not use physics.fourier.phase_ramp.
def remove_phase_ramp_gauge(fields: ArrayLike) -> ArrayLike:
    """Remove the linear phase ramp that ptychography cannot attribute.

    ``P e^{2 pi i q.r}`` with ``O e^{-2 pi i q.r}`` gives the same data. The
    ramp shifts the aperture-plane intensity by ``q``, so the centroid ``q``
    (pixels, relative to ``(H // 2, W // 2)``) of ``sum_m |A_m|^2`` with
    ``A = fftshift(fft2(ifftshift(psi)))`` is measured and
    ``exp(-2 pi i (q_y (y - H//2) / H + q_x (x - W//2) / W))`` is applied to all
    modes. Real-space intensity is unchanged; a non-integer ``q`` makes the
    ramp slightly non-periodic on the grid.

    Parameters
    ----------
    fields : numpy.ndarray or torch.Tensor
        Complex real-space modes centred on ``(H // 2, W // 2)``, shape
        ``(..., H, W)``.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        De-ramped modes, same shape and array family as ``fields``.
    """
    x, to_numpy = as_tensor(fields)
    x = as_complex(x)
    h, w = x.shape[-2:]
    ap = to_aperture_plane(x)
    qy, qx = _centroid_offset((cast(t.Tensor, ap).abs() ** 2).reshape(-1, h, w).sum(0))
    ry = (_coords(h, x) - h // 2)[:, None]
    rx = (_coords(w, x) - w // 2)[None, :]
    ramp = t.exp(-2j * np.pi * (qy * ry / h + qx * rx / w))
    return restore((x * ramp).to(x.dtype), to_numpy)


def remove_probe_gauges(
    probe: ArrayLike,
    *,
    drop_last: bool = False,
    translation: bool = True,
    phase_ramp: bool = True,
) -> ArrayLike:
    """Canonical mode stack of a reconstructed probe, gauges removed.

    Parameters
    ----------
    probe : numpy.ndarray or torch.Tensor
        Complex real-space modes, shape ``(M, H, W)`` or the solver layout
        ``(M, 1, H, W)``.
    drop_last : bool, optional
        Discard the last mode before anything else (some mixed-state solvers
        reserve it as a background/"junk" mode). Ignored for ``M == 1``.
        Default False.
    translation : bool, optional
        Apply :func:`remove_translation_gauge`. Default True.
    phase_ramp : bool, optional
        Apply :func:`remove_phase_ramp_gauge`. Default True.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Complex modes, shape ``(M', H, W)``.

    Raises
    ------
    ValueError
        If ``probe`` is neither ``(M, H, W)`` nor ``(M, 1, H, W)``.
    """
    x, to_numpy = as_tensor(probe)
    if x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 3:
        raise ValueError(
            f"probe shape must be (M, H, W) or (M, 1, H, W), got {tuple(x.shape)}"
        )
    x = as_complex(x)
    if drop_last and x.shape[0] > 1:
        x = x[:-1]
    if translation:
        x = remove_translation_gauge(x)
    if phase_ramp:
        x = remove_phase_ramp_gauge(x)
    return restore(cast(t.Tensor, x), to_numpy)
