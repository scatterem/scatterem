"""Elastic CBED of :class:`~.prism_eels_simulator.PrismEelsSimulator`.

Moved out of ``prism_eels_simulator.py`` unchanged, so that the driver
module stays below the module size limit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover - type names only
    from .prism_eels_simulator import PrismEelsSimulator

__all__ = ["elastic_cbed_data"]


def elastic_cbed_data(
    sim: PrismEelsSimulator,
    n_radial: int | None,
    n_angular: int,
    focal_backprop: float | str,
) -> Tensor:
    """Return the CBED stack of :meth:`PrismEelsSimulator.elastic_cbed`.

    Returns
    -------
    torch.Tensor
        ``scan + window`` float32 intensities, averaged over the
        configurations.
    """
    # Local: ``scatterem.nn`` imports ``scatterem.simulation`` while it
    # initialises, so a module-level import here would be circular.
    from scatterem.nn.modules.propagator import Propagator  # noqa: PLC0415
    from scatterem.nn.scan_exit_wave.prism import PrismScanOp  # noqa: PLC0415
    from scatterem.nn.scan_exit_wave.prism_smeta import (  # noqa: PLC0415
        build_prism_smeta,
    )

    f = sim.interpolation_factor
    window = (sim.pixels[0] // f, sim.pixels[1] // f)
    s_meta = build_prism_smeta(
        pixels=window,
        f=f,
        dx=np.asarray(sim.sampling, dtype=np.float32),
        wavelength=sim.wavelength,
        semiconvergence_angle=sim.semiconvergence_angle * 1e-3,
        fov_shape=None,
        device=sim.device,
        n_radial=n_radial,
        n_angular=n_angular,
    )
    op = PrismScanOp(s_meta, f=f, focal_backprop=focal_backprop)
    grid = (sim.slice_thickness, sim.wavelength, sim.sampling, sim.pixels)
    propagator = Propagator(*grid, sim.device)
    probe_q = sim._aperture(window, torch.complex64)
    probe_q = probe_q / probe_q.abs().pow(2).sum().sqrt()
    # The scan operator takes the probe centred in its window and the
    # window's top-left corner on the grid, split into pixel and sub-pixel.
    probe = torch.fft.fftshift(torch.fft.ifft2(probe_q, norm="ortho"))
    corner = sim.scan_pixels() - torch.tensor(
        [window[0] // 2, window[1] // 2], dtype=torch.float64, device=sim.device
    )
    positions = torch.round(corner)
    dr = (corner - positions).float()
    probes = probe[None, None].expand(1, corner.shape[0], *window).contiguous()
    total = None
    with torch.no_grad():
        for c in range(sim.num_frozen_phonons):
            T = sim.transmissions(c).to(torch.complex64)
            exit_waves = op(T, probes, positions.int(), dr, propagator)[0]
            cbed = torch.fft.fft2(exit_waves, norm="ortho").abs().pow(2)
            total = cbed if total is None else total + cbed
    assert total is not None  # num_frozen_phonons >= 1
    return (total / sim.num_frozen_phonons).reshape(*sim.scan, *window)
