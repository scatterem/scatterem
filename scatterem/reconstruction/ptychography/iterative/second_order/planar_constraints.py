"""The per-pass constraints of :class:`~.planar.PlanarPtychography`.

:class:`PlanarConstraints` applies, in this order, the constraints of the
mixed-state BH runs of the MEMS vortex probe work:

1. ``orthogonalize``: the modes are made orthogonal, dominant mode first
   (:func:`~..regularization.probe_modes.orthogonalize_probe`);
2. ``radial_support``: all modes but the last are multiplied by a soft
   real-space disk of radius ``radius_frac * N / 2`` and a cosine edge of
   ``margin`` pixels (power not kept: the last, free mode takes the rest);
3. ``aperture_support``: all modes but the last are projected onto a map in
   the centred aperture plane, power kept
   (:func:`~..regularization.aperture_support.probe_aperture_support`), for
   example the bounding box of the measured pixels without the needle band;
4. ``absorption_width``: object pixels whose absorption ``Im V`` is more than
   this from its median are reset, and ``2 pi`` jumps of the phase are
   removed (:func:`~..regularization.absorption_bound.bound_absorption`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ..regularization.absorption_bound import bound_absorption
from ..regularization.aperture_support import probe_aperture_support
from ..regularization.probe_modes import orthogonalize_probe
from ._planar_geometry import radial_support_mask

__all__ = ["PlanarConstraints", "radial_support"]


def radial_support(probe: Tensor, radius_frac: float, margin: float) -> Tensor:
    """All modes but the last times a soft real-space disk (a new tensor).

    Parameters
    ----------
    probe : Tensor
        ``(M, 1, N, N)`` complex modes.
    radius_frac : float
        Radius of the flat part as a fraction of ``N / 2``.
    margin : float
        Width of the cosine edge in pixels.

    Returns
    -------
    Tensor
        The constrained modes; a one-mode probe comes back unchanged.
    """
    out = probe.clone()
    if probe.shape[0] < 2:
        return out
    n = int(probe.shape[-1])
    mask = radial_support_mask(n, radius_frac * (n / 2), margin, probe.device)
    out[:-1] = out[:-1] * mask.to(probe.dtype)
    return out


@dataclass
class PlanarConstraints:
    """Per-pass constraints of the 2D model; each one is off by default.

    Parameters
    ----------
    orthogonalize : bool
        Orthogonalize the modes, dominant first.
    radial_support : tuple of float or None
        ``(radius_frac, margin)`` of the real-space disk.
    aperture_support : Tensor or None
        ``(N, N)`` real map on the centred aperture plane of the model grid.
    absorption_width : float or None
        Allowed distance of ``Im V`` from its median.
    absorption_margin : int
        Border, in pixels, left out of that median.
    """

    orthogonalize: bool = False
    radial_support: tuple[float, float] | None = None
    aperture_support: Tensor | None = None
    absorption_width: float | None = None
    absorption_margin: int = 64

    @torch.no_grad()
    def apply(self, model: Any) -> int:
        """Apply the constraints to ``model`` in place.

        Parameters
        ----------
        model : PlanarPtychography
            The model; its ``probe`` and ``object`` change.

        Returns
        -------
        int
            The number of object pixels the absorption bound reset.
        """
        probe = model.probe
        if self.orthogonalize:
            probe.copy_(orthogonalize_probe(model, {}))
        if self.radial_support is not None:
            probe.copy_(radial_support(probe.detach(), *self.radial_support))
        if self.aperture_support is not None:
            support = {"support": self.aperture_support}
            probe.copy_(probe_aperture_support(model, support))
        if self.absorption_width is None:
            return 0
        obj = model.object
        phase, absorption, n_bad = bound_absorption(
            obj.real, obj.imag, self.absorption_width, self.absorption_margin
        )
        obj.copy_(torch.complex(phase, absorption))
        return n_bad
