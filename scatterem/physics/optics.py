"""Lens wavefront of orders 1 to 3, and round masks with a smooth rim.

This module was written for scatterem from the following literature:

- Krivanek, O. L., Dellby, N. and Lupini, A. R. (1999). Towards sub-Angstrom
  electron beams. Ultramicroscopy 78, 1-11.
- Haider, M., Uhlemann, S. and Zach, J. (2000). Upper limits for the residual
  aberrations of a high-resolution aberration-corrected STEM.
  Ultramicroscopy 81, 163-175.
- Kirkland, E. J. (2020). Advanced Computing in Electron Microscopy, 3rd ed.,
  ch. 3 (the aberration function of the objective lens).

Terms. A "lens vector" holds the 12 real coefficients [Angstrom] in the order
of ``LENS_LAYOUT``. One entry is a "slot". The "wavefront" is

    W(k) = (2 pi / lambda) sum_j c_j theta^(n_j+1) / (n_j+1) T_j(m_j phi)

[rad], with ``theta = lambda |k|``, ``T = cos`` for a slot of part ``c`` and
``T = sin`` for part ``s``. The "slope" is ``(dW/dk_y, dW/dk_x)``
[rad Angstrom]. The module evaluates both as polynomials in the angles
``lambda k_x`` and ``lambda k_y``, so they are exact zero at ``k = 0`` and
have finite derivatives everywhere.

The masks have a logistic "rim": at the distance ``d`` [pixel] from the grid
origin, a disc of radius ``R`` and rim width ``w`` has the value
``1 / (1 + exp(4 (d - R) / w))``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch

from scatterem.physics._checks import _checked, _host_real
from scatterem.physics._lens_basis import TERMS, basis_slope_stacks, basis_stack
from scatterem.physics._smooth_masks import (
    RIM_WIDTH_PX,
    illumination_disc,
    smooth_annulus,
    smooth_disc,
)

__all__ = [
    "AMPLITUDE_LAYOUT",
    "LENS_LAYOUT",
    "LENS_TERMS",
    "RIM_WIDTH_PX",
    "illumination_disc",
    "lens_from_amplitude",
    "lens_to_amplitude",
    "lens_wavefront",
    "lens_wavefront_slope",
    "pack_lens",
    "smooth_annulus",
    "smooth_disc",
    "unpack_lens",
]

LENS_TERMS: tuple[tuple[int, int, str], ...] = TERMS
"""``(order n, multiplicity m, part)`` of each slot. Part ``c`` multiplies
``cos(m phi)``, part ``s`` multiplies ``sin(m phi)``."""

LENS_LAYOUT: tuple[str, ...] = tuple(
    f"C{n}{m}" if m == 0 else f"C{n}{m}{'a' if p == 'c' else 'b'}" for n, m, p in TERMS
)
"""Labels of the 12 slots of a lens vector, in order: the Krivanek names
``"C10"``, ``"C12a"``, ``"C12b"``, ..., ``"C34b"`` (Krivanek et al. 1999).
The order is by order ``n``, then multiplicity ``m``, then the ``a`` part
(factor ``cos(m phi)``) before the ``b`` part (factor ``sin(m phi)``)."""

_MAGNITUDE_SLOTS: tuple[int, ...] = tuple(
    j for j, (_, m, part) in enumerate(TERMS) if m == 0 or part == "c"
)
"""Lens slots whose term gives one magnitude (or the signed ``m = 0`` value)."""

_ANGLE_SLOTS: tuple[int, ...] = tuple(
    j for j, (_, m, part) in enumerate(TERMS) if m > 0 and part == "c"
)
"""Lens slots of the ``cos(m phi)`` part of each term with ``m > 0``."""

AMPLITUDE_LAYOUT: tuple[str, ...] = tuple(
    f"C{TERMS[j][0]}{TERMS[j][1]}" for j in _MAGNITUDE_SLOTS
) + tuple(f"phi{TERMS[j][0]}{TERMS[j][1]}" for j in _ANGLE_SLOTS)
"""Labels of the magnitude-angle form: first the seven magnitudes
``"C10"``, ``"C12"``, ``"C21"``, ``"C23"``, ``"C30"``, ``"C32"``, ``"C34"``,
then the five angles ``"phi12"``, ``"phi21"``, ``"phi23"``, ``"phi32"``,
``"phi34"``. Each part is in the order of ``LENS_TERMS``. A slot with
``m = 0`` holds its signed value. For ``m > 0`` the ``"C<n><m>"`` slot holds
the magnitude ``M`` [Angstrom] and the ``"phi<n><m>"`` slot holds the angle
``alpha`` [rad]."""

_REAL_DTYPES = (torch.float32, torch.float64)


def _check_layout(layout: Sequence[str]) -> tuple[str, ...]:
    """Return the known layout that equals ``layout``, or raise."""
    if layout is LENS_LAYOUT or tuple(layout) == LENS_LAYOUT:
        return LENS_LAYOUT
    if layout is AMPLITUDE_LAYOUT or tuple(layout) == AMPLITUDE_LAYOUT:
        return AMPLITUDE_LAYOUT
    raise ValueError("layout must be LENS_LAYOUT or AMPLITUDE_LAYOUT")


# --- labelled form ---------------------------------------------------------


def pack_lens(
    values: Mapping[str, float],
    *,
    layout: Sequence[str] = LENS_LAYOUT,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Make a lens vector from labelled values.

    Parameters
    ----------
    values : Mapping[str, float]
        Map from slot label to value. Lengths are in [Angstrom], angles of
        ``AMPLITUDE_LAYOUT`` are in [rad]. A missing label gives 0 in its
        slot (also 0 rad for a missing angle).
    layout : tuple of str, optional
        ``LENS_LAYOUT`` (default) or ``AMPLITUDE_LAYOUT``.
    dtype : torch.dtype, optional
        Real dtype of the result. The default float64 keeps all digits of
        coefficients that span many decades.
    device : torch.device or str, optional
        Device of the result. ``None`` means CPU.

    Returns
    -------
    torch.Tensor
        Shape ``(12,)``, in the order of ``layout``.

    Raises
    ------
    TypeError
        If ``values`` is not a mapping, a value is not real, or ``dtype`` is
        not float32 or float64.
    ValueError
        If a label is not in ``layout``, a value is not finite, or ``layout``
        is unknown.
    """
    order = _check_layout(layout)
    if not isinstance(values, Mapping):
        raise TypeError(f"values must be a mapping, got {type(values).__name__}")
    if dtype not in _REAL_DTYPES:
        raise TypeError(f"dtype must be torch.float32 or torch.float64, got {dtype}")
    flat = [0.0] * len(order)
    for label, value in values.items():
        if label not in order:
            raise ValueError(f"values has the unknown label {label!r}")
        x = _host_real(f"values[{label!r}]", value)
        if not math.isfinite(x):
            raise ValueError(f"values[{label!r}] must be finite, got {x}")
        flat[order.index(label)] = x
    return torch.tensor(flat, dtype=dtype, device=device)


def unpack_lens(
    vector: torch.Tensor | Sequence[float], *, layout: Sequence[str] = LENS_LAYOUT
) -> dict[str, float]:
    """Make labelled values from a lens vector.

    This is a host-side helper. It copies a CUDA tensor to the host.

    Parameters
    ----------
    vector : torch.Tensor or sequence of float
        Shape ``(12,)``, in the order of ``layout``.
    layout : tuple of str, optional
        ``LENS_LAYOUT`` (default) or ``AMPLITUDE_LAYOUT``.

    Returns
    -------
    dict of str to float
        All 12 labels with their values.

    Raises
    ------
    ValueError
        If the shape is not ``(12,)`` or ``layout`` is unknown.
    """
    order = _check_layout(layout)
    if isinstance(vector, torch.Tensor):
        flat = vector.detach().to("cpu", torch.float64)
    else:
        flat = torch.as_tensor(vector, dtype=torch.float64)
    if tuple(flat.shape) != (len(order),):
        raise ValueError(f"vector must have shape (12,), got {tuple(flat.shape)}")
    return dict(zip(order, flat.tolist(), strict=True))


# --- wavefront and slope ---------------------------------------------------


def _prepare(
    lens: object, fy: object, fx: object, lambda_a: object
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Check the inputs and return ``(lens, u, v, lambda)``."""
    for name, t in (("fy", fy), ("fx", fx)):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(t).__name__}")
        if t.dtype not in _REAL_DTYPES:
            raise TypeError(f"{name} must be float32 or float64, got {t.dtype}")
    if fy.device != fx.device:  # pyright: ignore[reportAttributeAccessIssue]
        raise ValueError(f"fy is on {fy.device} but fx is on {fx.device}")  # pyright: ignore[reportAttributeAccessIssue]
    lam = _checked("lambda_a", lambda_a, zero_ok=False)
    dtype = torch.promote_types(fy.dtype, fx.dtype)  # pyright: ignore[reportAttributeAccessIssue]
    try:
        gy, gx = torch.broadcast_tensors(fy, fx)
    except RuntimeError as err:
        raise ValueError(f"fy and fx do not broadcast: {err}") from err
    if isinstance(lens, torch.Tensor):
        if not lens.is_floating_point():
            raise TypeError(f"lens must have a real floating dtype, got {lens.dtype}")
        if lens.device != fy.device:  # pyright: ignore[reportAttributeAccessIssue]
            raise ValueError(f"lens is on {lens.device} but fy is on {fy.device}")  # pyright: ignore[reportAttributeAccessIssue]
        coef = lens.to(dtype)
    else:
        coef = torch.as_tensor(lens, dtype=dtype, device=fy.device)  # pyright: ignore[reportAttributeAccessIssue]
    if coef.dim() == 0 or coef.shape[-1] != len(TERMS):
        raise ValueError(
            f"lens must have a last dimension of 12, got shape {tuple(coef.shape)}"
        )
    return coef, lam * gx.to(dtype), lam * gy.to(dtype), lam


def lens_wavefront(
    lens: torch.Tensor | Sequence[float],
    fy: torch.Tensor,
    fx: torch.Tensor,
    *,
    lambda_a: float,
) -> torch.Tensor:
    """Evaluate the lens wavefront ``W`` [rad] on frequency coordinates.

    Batch rule (outer product): the result has the shape ``(*B, *G)``, with
    ``B`` the batch shape of ``lens`` and ``G`` the broadcast shape of ``fy``
    and ``fx``. Lens ``b`` gives the wavefront at ``[b]``. The function never
    pairs lens dimensions with grid dimensions.

    Parameters
    ----------
    lens : torch.Tensor or sequence of float
        Lens vector ``(12,)`` or batch ``(*B, 12)`` [Angstrom], in the order of
        ``LENS_LAYOUT``. A tensor must be on the device of ``fy``. The
        function casts it to the result dtype and keeps its autograd graph.
        A sequence is copied to the device of ``fy``; this copy can make the
        host wait.
    fy, fx : torch.Tensor
        Frequency coordinates ``k_y`` and ``k_x`` [1/Angstrom], float32 or
        float64, on one device, broadcastable to the grid shape ``G``.
    lambda_a : float
        Wavelength [Angstrom], a finite host scalar larger than zero.

    Returns
    -------
    torch.Tensor
        Wavefront [rad], shape ``(*B, *G)``, dtype promoted from ``fy`` and
        ``fx``. With a lens tensor on CUDA, the call does not synchronise the
        host.

    Raises
    ------
    TypeError
        If ``fy`` or ``fx`` is not a real float tensor, ``lens`` is not a
        floating tensor or ``lambda_a`` is not a host real scalar.
    ValueError
        If the devices differ, the grids do not broadcast, the last dimension
        of ``lens`` is not 12 or ``lambda_a`` is not finite and positive.
    """
    coef, u, v, lam = _prepare(lens, fy, fx, lambda_a)
    basis = basis_stack(u, v) * (2.0 * math.pi / lam)
    return torch.tensordot(coef, basis, dims=([-1], [0]))  # pyright: ignore[reportArgumentType]


def lens_wavefront_slope(
    lens: torch.Tensor | Sequence[float],
    fy: torch.Tensor,
    fx: torch.Tensor,
    *,
    lambda_a: float,
) -> torch.Tensor:
    """Evaluate the slope ``(dW/dk_y, dW/dk_x)`` [rad Angstrom] of the wavefront.

    The function uses the exact derivatives of the basis polynomials. The
    inputs, the batch rule, the dtype rule and the device rule are those of
    :func:`lens_wavefront`. The autograd graph allows a second backward pass.
    ``slope / (2 pi)`` is the beam shift ``(y, x)`` [Angstrom].

    Parameters
    ----------
    lens : torch.Tensor or sequence of float
        Lens vector ``(12,)`` or batch ``(*B, 12)`` [Angstrom].
    fy, fx : torch.Tensor
        Frequency coordinates ``k_y`` and ``k_x`` [1/Angstrom].
    lambda_a : float
        Wavelength [Angstrom], finite and larger than zero.

    Returns
    -------
    torch.Tensor
        Shape ``(*B, *G, 2)`` [rad Angstrom]. ``[..., 0]`` is ``dW/dk_y`` and
        ``[..., 1]`` is ``dW/dk_x``.

    Raises
    ------
    TypeError
        As :func:`lens_wavefront`.
    ValueError
        As :func:`lens_wavefront`.
    """
    coef, u, v, _ = _prepare(lens, fy, fx, lambda_a)
    d_u, d_v = basis_slope_stacks(u, v)
    full_turn = 2.0 * math.pi
    slope_y = torch.tensordot(coef, d_v * full_turn, dims=([-1], [0]))  # pyright: ignore[reportArgumentType]
    slope_x = torch.tensordot(coef, d_u * full_turn, dims=([-1], [0]))  # pyright: ignore[reportArgumentType]
    return torch.stack((slope_y, slope_x), dim=-1)


# --- magnitude-angle form --------------------------------------------------


def _as_lens_tensor(name: str, value: object) -> torch.Tensor:
    """Return a floating tensor with a last dimension of 12, or raise."""
    if isinstance(value, torch.Tensor):
        if not value.is_floating_point():
            raise TypeError(
                f"{name} must have a real floating dtype, got {value.dtype}"
            )
        t = value
    else:
        t = torch.as_tensor(value, dtype=torch.float64)
    if t.dim() == 0 or t.shape[-1] != len(TERMS):
        raise ValueError(
            f"{name} must have a last dimension of 12, got shape {tuple(t.shape)}"
        )
    return t


def lens_to_amplitude(lens: torch.Tensor | Sequence[float]) -> torch.Tensor:
    """Convert a lens vector to the magnitude-angle form.

    For a term with ``m > 0``, ``(A, B)`` becomes ``M = hypot(A, B)``
    [Angstrom] and ``alpha`` [rad] with ``A = M cos(m alpha)`` and
    ``B = M sin(m alpha)``. The angle is in ``[0, 2 pi / m)``. ``M = 0``
    gives ``alpha = 0``; there the derivative by the coefficients is not
    defined. A slot with ``m = 0`` keeps its signed value. The order of the
    result is ``AMPLITUDE_LAYOUT``.

    Parameters
    ----------
    lens : torch.Tensor or sequence of float
        Shape ``(..., 12)`` in the order of ``LENS_LAYOUT`` [Angstrom]. A
        sequence becomes a float64 CPU tensor.

    Returns
    -------
    torch.Tensor
        Same shape, dtype and device as ``lens``.

    Raises
    ------
    TypeError
        If ``lens`` is a tensor without a floating dtype.
    ValueError
        If the last dimension is not 12.
    """
    cols = _as_lens_tensor("lens", lens).unbind(-1)
    magnitudes = []
    for j in _MAGNITUDE_SLOTS:
        m = TERMS[j][1]
        magnitudes.append(cols[j] if m == 0 else torch.hypot(cols[j], cols[j + 1]))
    angles = []
    for j in _ANGLE_SLOTS:
        m = TERMS[j][1]
        a, b = cols[j], cols[j + 1]
        period = 2.0 * math.pi / m
        alpha = torch.remainder(torch.atan2(b, a), 2.0 * math.pi) / m
        angles.append(torch.where(alpha >= period, torch.zeros_like(alpha), alpha))
    return torch.stack(magnitudes + angles, dim=-1)


def lens_from_amplitude(amp: torch.Tensor | Sequence[float]) -> torch.Tensor:
    """Convert the magnitude-angle form back to a lens vector.

    ``A = M cos(m alpha)`` and ``B = M sin(m alpha)``. Any real ``M`` and
    ``alpha`` are accepted.

    Parameters
    ----------
    amp : torch.Tensor or sequence of float
        Shape ``(..., 12)`` in the order of ``AMPLITUDE_LAYOUT``. Magnitudes
        are in [Angstrom], angles in [rad].

    Returns
    -------
    torch.Tensor
        Lens vector in the order of ``LENS_LAYOUT`` [Angstrom], same shape,
        dtype and device.

    Raises
    ------
    TypeError
        If ``amp`` is a tensor without a floating dtype.
    ValueError
        If the last dimension is not 12.
    """
    vals = _as_lens_tensor("amp", amp).unbind(-1)
    n_mag = len(_MAGNITUDE_SLOTS)
    angle_of = {j: vals[n_mag + k] for k, j in enumerate(_ANGLE_SLOTS)}
    cols: list[torch.Tensor] = [vals[0]] * len(TERMS)
    for k, j in enumerate(_MAGNITUDE_SLOTS):
        mag = vals[k]
        m = TERMS[j][1]
        if m == 0:
            cols[j] = mag
            continue
        alpha = angle_of[j]
        cols[j] = mag * torch.cos(m * alpha)
        cols[j + 1] = mag * torch.sin(m * alpha)
    return torch.stack(cols, dim=-1)
