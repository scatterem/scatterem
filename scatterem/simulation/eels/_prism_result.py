"""The result container of :class:`~.prism_eels_simulator.PrismEelsSimulator`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from torch import Tensor

__all__ = ["Edge", "PrismResult"]

Edge = tuple[int, int, int]


@dataclass(frozen=True)
class PrismResult:
    """The output of one :class:`PrismEelsSimulator` call.

    Parameters
    ----------
    data : torch.Tensor
        Real, non-negative intensities, averaged over the frozen-phonon
        configurations. The shape depends on ``output``:

        - ``"map"``: ``scan``;
        - ``"qeels"``: ``scan + (n_q,)``, one bin per entry of ``q_pixels``;
        - ``"cube"``: ``scan + pixels``, corner-origin diffraction patterns;
        - ``"cbed"``: ``scan + window``, corner-origin diffraction patterns on
          the PRISM window ``pixels // interpolation_factor``.
    output : {"map", "qeels", "cube", "cbed"}
        What ``data`` holds.
    energy : float
        Probe energy [eV].
    detector_sampling : tuple of float
        Reciprocal pixel ``(dky, dkx)`` of the diffraction axes of ``data``
        [1/Angstrom]. For ``"qeels"`` it converts ``q_pixels`` to 1/Angstrom.
    edge : tuple of int or None, optional
        ``(Z, n, l)`` of the ionized subshell; None for ``"cbed"``.
    epsilon : float or None, optional
        Energy of the ejected electron above the threshold [eV]; None for
        ``"cbed"``.
    energy_losses : numpy.ndarray or None, optional
        Energy loss of each summed transition [eV]; None for ``"cbed"``.
    q_pixels : torch.Tensor or None, optional
        For ``"qeels"``: the detector coordinates that ``prism_eels_image``
        returns with the patterns, in grid pixels. ``(n_q, 2)`` in
        ``(qy, qx)`` order for ``qeels_axis="both"``, ``(n_q,)`` for a one-axis
        projection. None for the other outputs.
    """

    data: Tensor
    output: Literal["map", "qeels", "cube", "cbed"]
    energy: float
    detector_sampling: tuple[float, float]
    edge: Edge | None = None
    epsilon: float | None = None
    energy_losses: np.ndarray | None = None
    q_pixels: Tensor | None = None
