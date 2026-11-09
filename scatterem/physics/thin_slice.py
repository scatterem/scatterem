"""Phase and thin-slice transmission function of projected potentials.

Literature: the phase ``sigma V`` and the transmission function
``exp(i sigma V)`` follow Kirkland (2020, *Advanced Computing in Electron
Microscopy*, 3rd ed., ch. 5 and 6). The absorptive potential follows Hall
and Hirsch (1965), Proc. R. Soc. Lond. A 286, 158-177. Written for scatterem
from this literature.

Units: projected potentials [Volt Angstrom], kinetic energy [eV], phase
[rad].
"""

from __future__ import annotations

import logging
import math
import numbers

import numpy as np
import torch

from .constants import phase_per_potential_at
from .periodic_sampling import lowpass_ellipse

__all__ = ["phase_from_projection", "thin_slice_transmission"]

_log = logging.getLogger(__name__)


def _sigma(beam_ev) -> float:
    if isinstance(beam_ev, (bool, np.bool_)) or not isinstance(beam_ev, numbers.Real):
        raise TypeError(f"beam_ev: must be a real number [eV]; got {beam_ev!r}")
    e = float(beam_ev)
    if not math.isfinite(e) or e <= 0.0:
        raise ValueError(f"beam_ev: must be finite and > 0 [eV]; got {beam_ev!r}")
    return phase_per_potential_at(e)


def _check_projected(projected) -> None:
    if not isinstance(projected, torch.Tensor):
        raise TypeError(
            f"projected: must be a torch.Tensor; got {type(projected).__name__}"
        )
    if projected.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"projected: must be float32 or float64; got {projected.dtype}")


def phase_from_projection(beam_ev, projected: torch.Tensor) -> torch.Tensor:
    """Return the phase shift ``phi = sigma(E) V`` [rad].

    Parameters
    ----------
    beam_ev : float
        Kinetic energy ``E`` [eV], finite and > 0.
    projected : torch.Tensor
        Projected potential ``V`` [Volt Angstrom], float32 or float64, any
        shape.

    Returns
    -------
    torch.Tensor
        New tensor with the shape, dtype, device and autograd graph of
        ``projected``. ``sigma`` is :func:`~scatterem.physics.constants.phase_per_potential_at`
        [rad / (Volt Angstrom)].

    Raises
    ------
    TypeError
        For a non-tensor or another dtype, or a non-real ``beam_ev``.
    ValueError
        For an energy that is not finite or ``<= 0``.
    """
    _check_projected(projected)
    return projected * _sigma(beam_ev)


def _absorption(attenuation, projected: torch.Tensor) -> torch.Tensor:
    """Return the absorptive part on the device and dtype of ``projected``."""
    if isinstance(attenuation, torch.Tensor):
        if attenuation.is_complex():
            raise TypeError("attenuation: must be real; got a complex tensor")
        if attenuation.dtype == torch.bool:
            raise TypeError("attenuation: must be real numbers; got bool")
        value = attenuation.to(device=projected.device, dtype=projected.dtype)
    elif isinstance(attenuation, (bool, np.bool_)) or not isinstance(
        attenuation, numbers.Real
    ):
        raise TypeError(
            f"attenuation: must be a real tensor or number; got {attenuation!r}"
        )
    else:
        value = torch.tensor(
            float(attenuation), dtype=projected.dtype, device=projected.device
        )
    try:
        joint = torch.broadcast_shapes(projected.shape, value.shape)
    except RuntimeError:
        joint = None
    if joint != projected.shape:
        raise ValueError(
            f"attenuation: shape {tuple(value.shape)} must broadcast to the shape "
            f"{tuple(projected.shape)} of projected without enlarging it"
        )
    return value


def thin_slice_transmission(
    beam_ev,
    projected: torch.Tensor,
    *,
    attenuation=None,
    bandlimit=None,
) -> torch.Tensor:
    """Return the thin-slice transmission function of each slice.

    ``t = exp(i sigma V) * exp(-sigma V')`` with ``sigma =
    phase_per_potential_at(E)``. The code evaluates ``torch.polar(amp, phi)``
    with ``phi = sigma V`` and ``amp = 1`` (no absorption) or ``amp =
    exp(-sigma V')``. So ``|t| = 1`` to rounding without absorption.

    Parameters
    ----------
    beam_ev : float
        Kinetic energy [eV], finite and > 0.
    projected : torch.Tensor
        Real float32 or float64 projected potential ``V`` ``(..., Ny, Nx)``
        [Volt Angstrom], at least 2-D, no empty dimension. Leading
        dimensions are slices or batches.
    attenuation : torch.Tensor or float, optional
        Projected absorptive potential ``V'`` [Volt Angstrom] (Hall and
        Hirsch 1965). It is given in the unit of ``V``, because it comes from
        the same projected build with absorptive form spectra; so no slice
        length enters. Real; it broadcasts to the shape of ``V`` without
        enlarging it, and moves to the device and dtype of ``V`` with its
        graph.
    bandlimit : float or tuple of float, optional
        If given, the result is ``lowpass_ellipse(t, bandlimit)``: a fraction
        of the Nyquist frequency, one value or ``(rho_y, rho_x)``. There is
        no default fraction; the alias-free limit is 2/3 (see
        :func:`~scatterem.physics.periodic_sampling.ellipse_mask`). Without
        it, no FFT runs.

    Returns
    -------
    torch.Tensor
        The transmission, complex64 for float32 input and complex128 for
        float64 input, with the
        shape and device of ``V``. The inputs do not change; the gradient
        reaches ``V`` and ``V'``.

    Raises
    ------
    TypeError
        For a non-tensor or non-real ``V``, a complex ``V'`` or a bad energy
        type.
    ValueError
        For fewer than 2 dimensions, an empty dimension, a ``V'`` that would
        enlarge the shape, or a bad energy value.
    """
    _check_projected(projected)
    if projected.dim() < 2 or 0 in projected.shape:
        raise ValueError(
            f"projected: needs at least 2 dimensions and no empty one; got "
            f"{tuple(projected.shape)}"
        )
    phi = phase_from_projection(beam_ev, projected)
    if attenuation is None:
        amp = torch.ones_like(phi)
    else:
        loss = _absorption(attenuation, projected)
        amp = torch.exp(loss * (-_sigma(beam_ev))).expand(phi.shape)
    out = torch.polar(amp, phi)
    if bandlimit is None:
        return out
    return lowpass_ellipse(out, bandlimit)
