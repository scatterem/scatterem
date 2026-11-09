"""Operators on the scan grid for the scan-rotation solver.

Discrete curl and divergence with a cell-centred 2 × 2 stencil, the six
float64 moments of the field, and the closed-form objective curves. See
:mod:`._rotation_solver` for the physics and the conventions.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from scatterem.data.calibration._common import _to_tensor, resolve_device

__all__ = ["curl_2d", "divergence_2d", "wrap_angle_deg"]


def wrap_angle_deg(angle_in_deg: float) -> float:
    """Wrap an angle in degrees into the half-open interval ``[-180, 180)``."""
    w = float(angle_in_deg) - 360.0 * math.floor((float(angle_in_deg) + 180.0) / 360.0)
    if w >= 180.0:
        w -= 360.0
    if w < -180.0:
        w += 360.0
    return w


def _as_scan_spacing(scan_spacing) -> tuple[float, float]:
    """Return ``(s0, s1)`` from a float or a pair; raise on invalid values."""
    try:
        steps = (
            (float(scan_spacing),) * 2
            if np.ndim(scan_spacing) == 0
            else tuple(float(v) for v in scan_spacing)
        )
    except (TypeError, ValueError):
        raise ValueError(
            f"scan_spacing must be a float or a pair, got {scan_spacing!r}"
        ) from None
    if len(steps) != 2 or not all(math.isfinite(s) and s > 0 for s in steps):
        raise ValueError(
            f"scan_spacing must be finite and positive, got {scan_spacing!r}"
        )
    return steps


def _validate_components(comp_a0, comp_a1) -> tuple[Tensor, Tensor]:
    """Validate a pair of real floating tensors for the stencil."""
    fx = torch.as_tensor(comp_a0)
    fy = torch.as_tensor(comp_a1)
    for f in (fx, fy):
        if not f.is_floating_point():
            raise ValueError(f"field dtype must be real floating, got {f.dtype}")
    if fx.shape != fy.shape:
        raise ValueError(
            f"field shapes differ: {tuple(fx.shape)} and {tuple(fy.shape)}"
        )
    if fx.ndim < 2 or fx.shape[-2] < 2 or fx.shape[-1] < 2:
        raise ValueError(
            f"need a field of shape (..., n0>=2, n1>=2), got {tuple(fx.shape)}"
        )
    if fy.device != fx.device:
        fy = fy.to(fx.device)
    return fx, fy


def _jacobian(fx: Tensor, fy: Tensor, s0: float, s1: float):
    """Return the Jacobian ``a, b, c, d`` at the cell centres.

    ``a = ∂f_x/∂x``, ``b = ∂f_x/∂y``, ``c = ∂f_y/∂x``, ``d = ∂f_y/∂y``. Each
    derivative is a forward difference averaged over the two neighbouring
    lines, so all four entries sit at ``(i + ½, j + ½)``.
    """

    def d0(g):
        diff = g[..., 1:, :] - g[..., :-1, :]
        return 0.5 * (diff[..., :, 1:] + diff[..., :, :-1]) / s0

    def d1(g):
        diff = g[..., :, 1:] - g[..., :, :-1]
        return 0.5 * (diff[..., 1:, :] + diff[..., :-1, :]) / s1

    return d0(fx), d1(fx), d0(fy), d1(fy)


def curl_2d(comp_a0, comp_a1, *, scan_spacing=(1.0, 1.0)) -> Tensor:
    """Discrete curl ``∂f_y/∂x − ∂f_x/∂y`` of a 2-D vector field.

    The stencil is cell-centred (2 × 2). Output point ``(i, j)`` sits at
    ``(i + ½, j + ½)``. Each derivative is a forward difference averaged over
    the two neighbouring lines. Reasons: all four Jacobian entries sit at one
    point, so curl and divergence obey the rotation identities exactly; the
    stencil is exact for linear fields and second order (``O(h²)``); it uses
    only measured values, with no padding and no one-sided edge formula; its
    only null pattern is the checkerboard.

    Parameters
    ----------
    comp_a0, comp_a1 : Tensor
        Real floating tensors of one shape ``(..., n0, n1)``. ``comp_a0`` is
        the component along axis −2 ("x"), ``comp_a1`` along axis −1 ("y").
        Leading axes are batch axes.
    scan_spacing : float or (float, float), optional
        Step between scan positions along axis −2 and axis −1. Derivatives
        are per unit of this step. Default ``(1, 1)``: per scan pixel.

    Returns
    -------
    Tensor
        Shape ``(..., n0 − 1, n1 − 1)``, same dtype and device as the input.
        The smallest valid field is ``2 × 2``; it gives a ``1 × 1`` output.

    Raises
    ------
    ValueError
        If the shapes differ, ``ndim < 2``, ``n0 < 2`` or ``n1 < 2``, a dtype
        is not real floating, or ``scan_spacing`` is invalid.
    """
    s0, s1 = _as_scan_spacing(scan_spacing)
    fx, fy = _validate_components(comp_a0, comp_a1)
    _, b, c, _ = _jacobian(fx, fy, s0, s1)
    return c - b


def divergence_2d(comp_a0, comp_a1, *, scan_spacing=(1.0, 1.0)) -> Tensor:
    """Discrete divergence ``∂f_x/∂x + ∂f_y/∂y`` of a 2-D vector field.

    Same stencil, shapes, dtypes and errors as :func:`curl_2d`.
    """
    s0, s1 = _as_scan_spacing(scan_spacing)
    fx, fy = _validate_components(comp_a0, comp_a1)
    a, _, _, d = _jacobian(fx, fy, s0, s1)
    return a + d


def _as_angle_grid(angle_grid_deg) -> np.ndarray:
    """Return the trial angles as a float64 array; raise if invalid."""
    if isinstance(angle_grid_deg, Tensor):
        angle_grid_deg = angle_grid_deg.detach().cpu().numpy()
    try:
        arr = np.asarray(angle_grid_deg, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("angle_grid_deg must be a 1-D sequence of numbers") from None
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("angle_grid_deg must be 1-D, non-empty and finite")
    return arr


def _angle_trig(angles: np.ndarray, device) -> tuple[Tensor, Tensor]:
    """Return ``cos 2θ`` and ``sin 2θ`` as float64 tensors on ``device``."""
    two = torch.as_tensor(np.radians(angles) * 2.0, dtype=torch.float64, device=device)
    return torch.cos(two), torch.sin(two)


def _moments(fx: Tensor, fy: Tensor, s0: float, s1: float) -> Tensor:
    """Return the six float64 sums ``K, D, M, K_P, D_P, M_P`` of the field."""
    a, b, c, d = _jacobian(fx.to(torch.float64), fy.to(torch.float64), s0, s1)
    kappa, delta = c - b, a + d
    kappa_p, delta_p = a - d, b + c
    return torch.stack(
        [
            (kappa * kappa).sum(),
            (delta * delta).sum(),
            (kappa * delta).sum(),
            (kappa_p * kappa_p).sum(),
            (delta_p * delta_p).sum(),
            (kappa_p * delta_p).sum(),
        ]
    )


def _curves_from_moments(
    mom: Tensor, c2: Tensor, s2: Tensor, curve_kind: str
) -> Tensor:
    """Evaluate both curves ``(2, A)`` from the moments in closed form.

    ``F = ½(K + D + K_P + D_P)`` equals the sum of the squared Jacobian
    entries. It is the same for both hypotheses, and its form does not change
    when the components are exchanged.
    """
    k, d, m, kp, dp, mp = mom.unbind()
    f = 0.5 * ((k + d) + (kp + dp))
    rows = []
    for kk, dd, mm in ((k, d, m), (kp, dp, mp)):
        if curve_kind == "curl":
            num = 0.5 * (kk + dd) + 0.5 * (kk - dd) * c2 + mm * s2
        else:
            num = 0.5 * (kk + dd) - 0.5 * (kk - dd) * c2 - mm * s2
        rows.append(torch.where(f > 0, num / torch.where(f > 0, f, 1.0), 0.0))
    return torch.stack(rows)


def _as_input(x, name: str):
    """Validate one CoM component; return it unchanged (tensor or ndarray)."""
    if not isinstance(x, (Tensor, np.ndarray)):
        try:
            x = np.asarray(x)
        except (TypeError, ValueError) as err:
            raise ValueError(f"{name} must be a tensor or an array") from err
    kind_ok = (
        (x.is_floating_point() or not (x.is_complex() or x.dtype == torch.bool))
        if isinstance(x, Tensor)
        else x.dtype.kind in "fiu"
    )
    if not kind_ok:
        raise ValueError(f"{name} must be real (not complex or bool), got {x.dtype}")
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {tuple(x.shape)}")
    return x


def _dt(x) -> torch.dtype:
    """torch dtype of a tensor or an ndarray."""
    if isinstance(x, Tensor):
        return x.dtype
    return torch.float64 if x.dtype == np.float64 else torch.float32


def com_field_tensors(centroid_a0, centroid_a1, run_device) -> tuple[Tensor, Tensor]:
    """The two CoM components as finite tensors of one dtype on one device.

    Each component must be a real 2-D tensor or array; both must have one
    shape of at least ``2 x 2``. The dtype is float64 if either input is
    float64, else float32. The device follows ``resolve_device`` (an
    explicit device wins; else the first tensor input; else the CPU).

    Raises
    ------
    ValueError
        For a component that is not real and 2-D, different shapes, a field
        smaller than ``2 x 2``, or a non-finite value.
    """
    centroid_a0 = _as_input(centroid_a0, "centroid_a0")
    centroid_a1 = _as_input(centroid_a1, "centroid_a1")
    if tuple(centroid_a0.shape) != tuple(centroid_a1.shape):
        raise ValueError(
            f"CoM shapes differ: {tuple(centroid_a0.shape)} and {tuple(centroid_a1.shape)}"
        )
    if centroid_a0.shape[0] < 2 or centroid_a0.shape[1] < 2:
        raise ValueError(
            f"the CoM field must be at least 2 x 2, got {tuple(centroid_a0.shape)}"
        )
    dtype = (
        torch.float64
        if torch.float64 in (_dt(centroid_a0), _dt(centroid_a1))
        else torch.float32
    )
    tensors = [t for t in (centroid_a0, centroid_a1) if isinstance(t, Tensor)]
    dev = resolve_device(run_device, *tensors)
    fx = _to_tensor(centroid_a0, dtype=dtype, device=dev)
    fy = _to_tensor(centroid_a1, dtype=dtype, device=dev)
    if not bool(torch.isfinite(fx).all() & torch.isfinite(fy).all()):
        raise ValueError("the CoM field must be finite")
    return fx, fy
