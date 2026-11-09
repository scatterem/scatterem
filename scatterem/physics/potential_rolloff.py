"""Low-pass rolloff of the projected potential spectrum.

Literature: the rolloff is the Hann window of Harris (1978), Proc. IEEE 66,
51-83, on the normalised radius of the Nyquist ellipse. The alias-free band
of 2/3 of the Nyquist frequency follows Kirkland (2020, *Advanced Computing
in Electron Microscopy*, 3rd ed., ch. 6). Written for scatterem from this
literature.

Units: the rolloff depth is a fraction of the normalised radius
[dimensionless]; frequencies [1/Angstrom].
"""

from __future__ import annotations

import logging
import math
import numbers

import numpy as np
import torch

from .periodic_sampling import ellipse_mask

__all__ = ["DEFAULT_ROLLOFF", "check_rolloff", "rolloff_mask"]

_log = logging.getLogger(__name__)

#: Default rolloff depth ``w``. Reason: with ``w = 1/3`` the rolloff is 1 for
#: ``r <= 2/3``, so it does not change any frequency of the alias-free
#: band that multislice keeps. It removes the ringing of the truncated
#: series near the atom cores, the Nyquist frequencies and the band corners
#: where the residual alias error is largest.
DEFAULT_ROLLOFF = 1.0 / 3.0


def check_rolloff(rolloff_depth) -> float | None:
    """Return a valid rolloff setting.

    Parameters
    ----------
    rolloff_depth : float or None
        ``None`` (no rolloff) or the depth ``w`` [normalised radius] in
        ``(0, 1)``.

    Returns
    -------
    float or None
        ``None`` or ``float(rolloff_depth)``.

    Raises
    ------
    TypeError
        If ``rolloff_depth`` is a ``bool``, a ``str`` or not a real number.
    ValueError
        If ``rolloff_depth`` is not finite or not in ``(0, 1)``.
    """
    if rolloff_depth is None:
        return None
    if isinstance(rolloff_depth, (bool, np.bool_)) or not isinstance(
        rolloff_depth, numbers.Real
    ):
        raise TypeError(
            f"rolloff: must be None or a real number; got {rolloff_depth!r}"
        )
    w = float(rolloff_depth)
    if not math.isfinite(w) or not 0.0 < w < 1.0:
        raise ValueError(
            f"rolloff: must be finite and in (0, 1); got {rolloff_depth!r}"
        )
    return w


def rolloff_mask(
    points,
    rolloff,
    *,
    half_layout: bool = False,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Return the rolloff ``T`` on the DFT grid, in FFT order.

    With the normalised radius ``r = sqrt((q_y / qN_y)**2 + (q_x / qN_x)**2)``,
    ``qN_i = N_i / (2 L_i)``, and the depth ``w``::

        T = 1                              for r <= 1 - w
        T = sin(pi/2 * (1 - r) / w)**2     for 1 - w < r < 1
        T = 0                              for r >= 1

    This is ``ellipse_mask(points, 1.0, fade_depth=w)``. ``T(0) = 1``, ``T``
    does not increase along any ray from 0, and ``T = 0`` at and beyond the
    Nyquist frequency.

    Parameters
    ----------
    points : tuple of int
        Grid point counts ``(Ny, Nx)``.
    rolloff : float or None
        Depth ``w`` in ``(0, 1)``, or ``None`` for ones.
    half_layout : bool, optional
        If True, return the ``rfft2`` layout ``(Ny, Nx // 2 + 1)``.
    device : torch.device or str, optional
        Device. Default the CPU.
    dtype : torch.dtype, optional
        Real dtype. Default the torch default dtype.

    Returns
    -------
    torch.Tensor
        Real rolloff, ``(Ny, Nx)`` or ``(Ny, Nx // 2 + 1)``.
    """
    w = check_rolloff(rolloff)
    ny, nx = (int(points[0]), int(points[1]))
    shape = (ny, nx // 2 + 1) if half_layout else (ny, nx)
    if w is None:
        dt = dtype if dtype is not None else torch.get_default_dtype()
        return torch.ones(shape, dtype=dt, device=device)
    return ellipse_mask(
        (ny, nx),
        1.0,
        half_last_axis=bool(half_layout),
        fade_depth=w,
        device=device,
        dtype=dtype,
    )
