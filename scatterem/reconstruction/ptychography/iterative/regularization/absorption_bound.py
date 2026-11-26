"""Object absorption bound with removal of 2 pi phase jumps.

A pixel of a dense object can go opaque during a fit: its absorption
``Im V`` grows until the transmission ``exp(-Im V)`` is near zero. Such a pixel
has no gradient left and stays dead, and its phase is free and drifts.
:func:`bound_absorption` finds these pixels by a bound on the absorption about
its median, resets them to the median of their good neighbours, and removes
2 pi jumps of the phase against the local median. The last step does not change
``exp(i V)``, so it does not change the data fit.

:func:`absorption_bound` is the same step as a ``regularization:`` object
constraint (key ``absorption_bound``) of the unified ``Ptychography``
module.

Ported from the fast-probe research driver (``OBJ_CLAMP``).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

__all__ = ["absorption_bound", "bound_absorption", "median5"]


def median5(x: Tensor, bad: Tensor | None = None) -> Tensor:
    """5 x 5 median filter of a 2-D tensor, optionally over good pixels only.

    The border is padded by replication. A pixel marked in ``bad`` does not
    enter any median. Where all 25 neighbours are bad, the input value is kept.
    For an even number of good neighbours, the lower median is used
    (:meth:`torch.Tensor.nanmedian`).

    Parameters
    ----------
    x : torch.Tensor
        Real ``(H, W)`` tensor.
    bad : torch.Tensor or None, optional
        Bool ``(H, W)``, True for a pixel to leave out. Default None.

    Returns
    -------
    torch.Tensor
        The filtered ``(H, W)`` tensor, dtype of ``x``.
    """
    xb = x if bad is None else torch.where(bad, torch.full_like(x, math.nan), x)
    xp = torch.nn.functional.pad(xb[None, None], (2, 2, 2, 2), mode="replicate")[0, 0]
    windows = xp.unfold(0, 5, 1).unfold(1, 5, 1).reshape(*x.shape, 25)
    m = windows.nanmedian(-1).values
    return torch.where(torch.isnan(m), x, m)


def bound_absorption(
    phase: Tensor, absorption: Tensor, width: float, margin: int = 64
) -> tuple[Tensor, Tensor, int]:
    """Reset opaque pixels and remove 2 pi phase jumps of one object slice.

    1. A pixel is bad when ``|absorption - med| > width``, with ``med`` the
       median absorption of the central region (a ``margin`` pixel border
       left out, clipped to leave at least 2 x 2 pixels).
    2. A bad pixel gets the 5 x 5 median (:func:`median5`) of the phase and
       of the absorption over its good neighbours; the absorption is then
       clamped to ``med +- width``.
    3. Every pixel loses the multiple of 2 pi that brings its phase nearest
       to the 5 x 5 median phase.

    Parameters
    ----------
    phase : torch.Tensor
        Real ``(H, W)`` phase ``Re V`` in rad.
    absorption : torch.Tensor
        Real ``(H, W)`` absorption ``Im V``; ``inf`` for a fully opaque pixel
        is allowed.
    width : float
        Allowed distance of the absorption from its median.
    margin : int, optional
        Border, in pixels, left out of the median. Default 64.

    Returns
    -------
    tuple of (torch.Tensor, torch.Tensor, int)
        The new phase, the new absorption and the number of bad pixels.
    """
    h, w = phase.shape[-2:]
    mm = max(0, min(int(margin), h // 2 - 1, w // 2 - 1))
    med = absorption[mm : h - mm, mm : w - mm].median()
    bad = (absorption - med).abs() > width
    n_bad = int(bad.sum())
    if n_bad:
        phase = torch.where(bad, median5(phase, bad), phase)
        absorption = torch.where(bad, median5(absorption, bad), absorption)
        absorption = absorption.clamp(med - width, med + width)
    two_pi = 2 * math.pi
    phase = phase - two_pi * torch.round((phase - median5(phase)) / two_pi)
    return phase, absorption, n_bad


def absorption_bound(
    amplitude: Tensor, phase: Tensor, regul_dict: dict[str, Any]
) -> tuple[Tensor, Tensor]:
    """:func:`bound_absorption` as a ``regularization:`` object constraint.

    The absorption is ``-log(amplitude)``; a zero amplitude (an opaque pixel
    whose transmission underflowed) gives ``inf`` and counts as bad. Each
    2-D slice of a 3-D object is bounded on its own.

    Parameters
    ----------
    amplitude : torch.Tensor
        Transmission amplitude ``exp(-Im V)``, shape ``(..., H, W)``.
    phase : torch.Tensor
        Phase ``Re V``, same shape.
    regul_dict : dict
        ``width`` (required, for example 2.5) and ``margin`` in pixels
        (default 64).

    Returns
    -------
    tuple of torch.Tensor
        The new ``(amplitude, phase)``, same shapes.

    Raises
    ------
    KeyError
        If ``width`` is missing.
    """
    width = float(regul_dict["width"])
    margin = int(regul_dict.get("margin", 64))
    shape = phase.shape
    h, w = shape[-2:]
    absorption = -torch.log(amplitude).reshape(-1, h, w)
    phases = phase.reshape(-1, h, w)
    out_amp, out_phase, n_total = [], [], 0
    for ph, ab in zip(phases, absorption, strict=True):
        ph2, ab2, n = bound_absorption(ph, ab, width, margin)
        out_phase.append(ph2)
        out_amp.append(torch.exp(-ab2))
        n_total += n
    if n_total:
        logger.info("absorption_bound: %d pixels reset", n_total)
    amp = torch.stack(out_amp).reshape(shape).to(amplitude.dtype)
    return amp, torch.stack(out_phase).reshape(shape).to(phase.dtype)
