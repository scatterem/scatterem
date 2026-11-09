"""Ionized sites of one specimen configuration of :class:`PrismEelsSimulator`.

The elastic potential and the ionization operators must see the same atoms.
Two rules give that here:

1. Positions: a configuration with thermal displacements moves each ionized
   atom by the offset that ``potential_stack`` gives the same row for the same
   :class:`~scatterem.physics.thermal_jitter.ThermalJitter`
   (:func:`~scatterem.physics.thermal_jitter.jitter_offsets`, one stream).
2. Occupancy: a site ionizes with the weight that its row has in the elastic
   potential. That is ``site_fraction`` when the structure has a partial
   occupancy, else 1 (the rule of ``potential_stack``). The signal of one
   site is an incoherent sum, so its weight multiplies its contribution. A
   site with weight 0 holds no atom and does not ionize.

Units: fractional coordinates [dimensionless], offsets [Angstrom].
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor

from scatterem.physics.atom_set import AtomSet, has_partial_occupancy
from scatterem.physics.thermal_jitter import ThermalJitter, jitter_offsets

__all__ = ["configuration_jitter", "ionized_sites", "occupancy_sum"]


def configuration_jitter(
    displacements: bool, seed: int | None, config: int
) -> ThermalJitter | None:
    """Return the thermal draw of configuration ``config``, or None.

    With ``seed=None`` each call draws a fresh seed. Use one returned object
    for every build of the configuration.
    """
    if not displacements:
        return None
    return ThermalJitter(draw_seed=None if seed is None else seed + config)


def ionized_sites(
    structure: AtomSet,
    Z: int,
    jitter: ThermalJitter | None,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the positions and the weights of the atoms of element ``Z``.

    Parameters
    ----------
    structure : AtomSet
        The specimen cell.
    Z : int
        Atomic number of the ionized element.
    jitter : ThermalJitter or None
        The draw of the configuration; None gives the equilibrium positions.
    device : torch.device
        Device of the potential build, so that the draws are the same stream.

    Returns
    -------
    tuple of numpy.ndarray
        ``(n, 3)`` float64 fractional ``(y, x, depth)`` positions and ``(n,)``
        float64 weights in ``[0, 1]``, one row for each row of element ``Z``.

    Raises
    ------
    ValueError
        If the structure has no atom of element ``Z``.
    """
    pick = np.asarray(structure.element_z) == int(Z)
    if not pick.any():
        raise ValueError(f"structure contains no atoms of element Z={Z}")
    sites = np.array(structure.fractional, dtype=np.float64)
    if jitter is not None:
        offsets = jitter_offsets(structure, (1, 1), jitter, device=device)[0, 0]
        sites = sites + offsets.cpu().numpy() / np.asarray(structure.sides)
    fill = np.asarray(structure.site_fraction, dtype=np.float64)
    weights = fill if has_partial_occupancy(fill) else np.ones_like(fill)
    return np.atleast_2d(sites[pick]), weights[pick]


def occupancy_sum(
    run: Callable[[np.ndarray], Tensor | tuple[Tensor, Tensor]],
    sites: np.ndarray,
    weights: np.ndarray,
) -> tuple[Tensor, Tensor | None]:
    """Sum one EELS kernel over the sites, each with its weight.

    The kernel sums the sites incoherently, so the sites of one weight go into
    one call and the call result is scaled by that weight. With every weight
    equal to 1 this is one call on all sites, without a scale. With every
    weight equal to 0 the result is zero, in the shape of one call.

    Parameters
    ----------
    run : callable
        ``run(sites)`` calls the kernel on a ``(m, 3)`` subset of ``sites``. It
        returns the output, or ``(output, q_pixels)``.
    sites : numpy.ndarray
        ``(n, 3)`` fractional positions.
    weights : numpy.ndarray
        ``(n,)`` weights in ``[0, 1]``.

    Returns
    -------
    tuple
        ``(output, q_pixels)``; ``q_pixels`` is None when ``run`` returns one
        tensor.
    """
    present = sorted({float(w) for w in weights if w > 0.0})
    total, q_pixels = None, None
    for w in present or [0.0]:
        rows = sites if w == 0.0 else sites[weights == w]
        out = run(rows)
        if isinstance(out, tuple):
            out, q_pixels = out
        if w == 0.0:
            out = torch.zeros_like(out)
        elif w != 1.0:
            out = out * w
        total = out if total is None else total + out
    assert total is not None  # the loop runs at least once
    return total, q_pixels
