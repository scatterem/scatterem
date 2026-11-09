"""``scan_chunk`` and ``contract`` of :func:`prism_eels_image` move no numbers.

``scan_chunk`` splits one atom's scan positions into bounded blocks, and
``contract`` picks one of two associations of the same sum. Both change only
the floating-point grouping, so every setting must agree with the unbounded
``"auto"`` result to float64 round-off.
"""

import numpy as np
import pytest
import torch

from scatterem.simulation.eels.forward.prism_eels_image import prism_eels_image
from scatterem.simulation.eels.physics.transition_potentials import (
    TransitionPotentials,
)


@pytest.fixture(scope="module")
def case():
    n = 12
    y, x = torch.meshgrid(
        torch.arange(n, dtype=torch.float64),
        torch.arange(n, dtype=torch.float64),
        indexing="ij",
    )
    sy, sx = (y + n // 2) % n - n // 2, (x + n // 2) % n - n // 2
    h = torch.exp(-(sy**2 + sx**2) / 2)
    channels = torch.stack([h, h * (sy + 1j * sx) / 2]).to(torch.complex128)
    tp = TransitionPotentials(
        channels, 8, 100_000.0, (0.5, 0.5), np.array([530.0, 531.0])
    )
    transmissions = torch.exp(
        1j * torch.stack([0.4 * torch.sin((y + j) * 0.7) for j in range(3)])
    )
    pq = (sy**2 + sx**2 <= 9).to(torch.complex128)
    sites = np.array([[0.02, 0.98, 0.01], [0.4, 0.6, 0.45]])
    gy, gx = torch.meshgrid(
        torch.arange(0.0, 12.0, 1.5, dtype=torch.float64),
        torch.arange(0.0, 12.0, 2.0, dtype=torch.float64),
        indexing="ij",
    )
    scan = torch.stack([gy.ravel(), gx.ravel()], 1)
    common = dict(
        wavelength=0.037,
        gridsize=(6.0, 6.0),
        slice_distance=2.0,
        detector_mrad=24.0,
        prism_mask=False,
    )
    return (pq, transmissions, tp, sites, scan), common


@pytest.mark.parametrize("partitioned", [False, True])
@pytest.mark.parametrize("contract", ["auto", "columns", "probe"])
@pytest.mark.parametrize("scan_chunk", [1, 5, None])
def test_scan_chunk_and_contract_agree(case, partitioned, contract, scan_chunk):
    args, common = case
    if partitioned:
        common = dict(common, partition={"n_radial": 2}, partition_s2={"n_radial": 2})
    reference = prism_eels_image(*args, **common, scan_chunk=None)
    got = prism_eels_image(*args, **common, contract=contract, scan_chunk=scan_chunk)
    torch.testing.assert_close(got, reference, rtol=1e-11, atol=1e-13)


def test_unknown_contract_or_axis_is_rejected(case):
    args, common = case
    with pytest.raises(ValueError, match="contract"):
        prism_eels_image(*args, **common, contract="window")
    with pytest.raises(ValueError, match="qeels_axis"):
        prism_eels_image(*args, **common, qeels_axis=2)
