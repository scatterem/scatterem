"""Probe support in the aperture plane, with the probe power kept.

:func:`probe_aperture_support` projects the probe modes in the centred
aperture plane ``A = fftshift(fft2(ifftshift(P)))`` onto a given support and
rescales them to the power they had before. It is a ``regularization:`` probe
constraint (key ``probe_aperture_support``) of the unified ``Ptychography``
module. Compared with a disk of the convergence angle (power not kept, every
mode), the support is any map, the power is kept and the last mode of a mixed
state stays free.

The helpers build the supports of the fast-probe research driver
(``PROBE_SUPPORT=det`` and ``CHOP_MASK=1``) on the model grid:

* :func:`measured_bbox_support`: the bounding box of the recorded pixels that
  hold data;
* :func:`ray_band_mask`: a ray-shaped band from a centre point (for example a
  needle or electrode shadow), to set to zero in the support;
* :func:`embed_centred`: a mask on the recorded detector, placed on the
  larger model grid of ``detector_pad_to``.

Every layout is centred: zero frequency at pixel ``(N // 2, N // 2)``.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
from torch import Tensor

from .probe_modes import _get_probe_parameter

logger = logging.getLogger(__name__)

__all__ = [
    "embed_centred",
    "measured_bbox_support",
    "probe_aperture_support",
    "ray_band_mask",
]


def _to_aperture(p: Tensor) -> Tensor:
    shifted = torch.fft.ifftshift(p, dim=(-2, -1))
    return torch.fft.fftshift(torch.fft.fft2(shifted, dim=(-2, -1)), dim=(-2, -1))


def _to_real(a: Tensor) -> Tensor:
    shifted = torch.fft.ifftshift(a, dim=(-2, -1))
    return torch.fft.fftshift(torch.fft.ifft2(shifted, dim=(-2, -1)), dim=(-2, -1))


def probe_aperture_support(
    model: Any, regul_dict: dict[str, Any], probe_index: int = 0
) -> Tensor:
    """Project the probe modes onto an aperture-plane support, power kept.

    The constrained modes are all modes but the last for a mixed state
    (``exclude_last_mode``, default True; the last mode stays free to take
    what the model cannot describe) and the one mode of a single-mode probe.
    They are multiplied by ``support`` in the centred aperture plane and
    then rescaled together, by one factor, to their summed power before the
    projection.

    Parameters
    ----------
    model : Any
        Forward model exposing ``probe_model``.
    regul_dict : dict
        ``support``: real ``(H, W)`` map (bool or float in ``[0, 1]``) on the
        centred aperture plane of the probe grid (required).
        ``exclude_last_mode`` (default True). ``preserve_power`` (default
        True).
    probe_index : int, optional
        Probe (tilt) to constrain. Default 0.

    Returns
    -------
    torch.Tensor
        Constrained probe, same shape and dtype as the probe.

    Raises
    ------
    KeyError
        If ``support`` is missing.
    ValueError
        If the support shape is not the probe grid.
    """
    probe = _get_probe_parameter(model, probe_index).detach()
    h, w = probe.shape[-2:]
    support = torch.as_tensor(regul_dict["support"], device=probe.device)
    if tuple(support.shape) != (h, w):
        raise ValueError(f"support shape {tuple(support.shape)} != probe grid {(h, w)}")
    n_modes = probe.shape[0] if probe.ndim > 2 else 1
    exclude_last = bool(regul_dict.get("exclude_last_mode", True))
    k = n_modes - 1 if exclude_last and n_modes > 1 else n_modes
    q = probe[:k] if probe.ndim > 2 else probe
    p0 = (q.abs() ** 2).sum()
    a = _to_aperture(q) * support.to(probe.real.dtype)
    qp = _to_real(a)
    if bool(regul_dict.get("preserve_power", True)):
        p1 = (qp.abs() ** 2).sum().clamp_min(1e-30)
        qp = qp * torch.sqrt(p0 / p1)
    out = probe.clone()
    if probe.ndim > 2:
        out[:k] = qp.to(probe.dtype)
        return out
    return qp.to(probe.dtype)


def embed_centred(mask: Tensor, shape: tuple[int, int]) -> Tensor:
    """Place a centred recorded-detector map on a larger centred grid.

    Pixel ``(My // 2, Mx // 2)`` of ``mask`` goes to ``(N0 // 2, N1 // 2)``,
    as the ``detector_pad_to`` model grid places the recorded block. The
    pixels outside the block are zero (False).

    Parameters
    ----------
    mask : torch.Tensor
        ``(My, Mx)`` map, centred.
    shape : tuple of int
        Target ``(N0, N1)``, at least ``(My, Mx)``.

    Returns
    -------
    torch.Tensor
        ``(N0, N1)`` map, dtype and device of ``mask``.

    Raises
    ------
    ValueError
        If the target is smaller than ``mask``.
    """
    my, mx = mask.shape[-2:]
    n0, n1 = int(shape[0]), int(shape[1])
    if n0 < my or n1 < mx:
        raise ValueError(f"grid {(n0, n1)} is smaller than the mask {(my, mx)}")
    oy, ox = n0 // 2 - my // 2, n1 // 2 - mx // 2
    out = torch.zeros((n0, n1), dtype=mask.dtype, device=mask.device)
    out[oy : oy + my, ox : ox + mx] = mask
    return out


def measured_bbox_support(
    measured: Tensor, shape: tuple[int, int] | None = None
) -> Tensor:
    """The bounding box of the measured pixels, as a float support.

    Parameters
    ----------
    measured : torch.Tensor
        Bool ``(My, Mx)``, True for a pixel that holds data (for example the
        sum of the patterns over the scan is positive), centred.
    shape : tuple of int or None, optional
        Probe (model) grid; the box is placed on it by :func:`embed_centred`.
        None keeps the recorded grid. Default None.

    Returns
    -------
    torch.Tensor
        float32 map, 1 inside the box and 0 outside.

    Raises
    ------
    ValueError
        If no pixel is measured.
    """
    if not bool(measured.any()):
        raise ValueError("no measured pixel")
    rows = measured.any(1).nonzero().flatten()
    cols = measured.any(0).nonzero().flatten()
    box = torch.zeros(measured.shape, dtype=torch.float32, device=measured.device)
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    box[r0:r1, c0:c1] = 1.0
    return box if shape is None else embed_centred(box, shape)


def ray_band_mask(
    shape: tuple[int, int],
    center: tuple[float, float],
    angle_deg: float,
    half_width: float,
    device: torch.device | str | None = None,
) -> Tensor:
    """A band of ``half_width`` pixels along a ray from ``center``.

    With ``(dy, dx)`` the offset of a pixel from ``center``, the ray points
    along ``(sin a, cos a)`` (``a = angle_deg``, from the +x axis towards +y).
    A pixel is in the band when its distance from the ray's line is at most
    ``half_width`` and it is on the ray's side of the centre.

    Parameters
    ----------
    shape : tuple of int
        Grid ``(H, W)``.
    center : tuple of float
        Start point ``(cy, cx)`` in pixels (for example the intensity
        centroid of the mean pattern).
    angle_deg : float
        Ray direction in degrees.
    half_width : float
        Half width of the band in pixels.
    device : torch.device or str or None, optional
        Device of the result. Default None (CPU).

    Returns
    -------
    torch.Tensor
        Bool ``(H, W)``, True in the band.
    """
    h, w = int(shape[0]), int(shape[1])
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float64, device=device),
        torch.arange(w, dtype=torch.float64, device=device),
        indexing="ij",
    )
    dy, dx = yy - float(center[0]), xx - float(center[1])
    sn, cs = math.sin(math.radians(angle_deg)), math.cos(math.radians(angle_deg))
    across = (dy * cs - dx * sn).abs()
    along = dy * sn + dx * cs
    return (across <= half_width) & (along >= 0)
