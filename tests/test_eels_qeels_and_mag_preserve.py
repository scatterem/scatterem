"""The two options the double-channeling image gained: qEELS, and mag_preserve.

Neither shipped with a test. The one that matters for qEELS is a recovery
identity: resolving a momentum axis only regroups the per-beam ``|amp|^2``
additions, so summing the spectrum back over that axis must return the scalar
map the same call produces with ``qeels_axis=None``. If the binning ever drops,
duplicates or misroutes a detector beam, that sum stops matching -- which no
check on the spectrum's shape would catch.

The tolerance is 1 ULP, not zero, and that is a property of the arithmetic
rather than slack: float64 addition is not associative, and the two paths group
the same summands differently (the scalar path reduces the detector axis inside
one ``sum``, the qEELS path reduces it through ``index_add_`` into bins). The
deviation measured here is 2e-16 relative. The upstream docstring claimed the
recovery was exact; it is exact in exact arithmetic only.

``mag_preserve`` is not an identity: it deliberately changes the partitioned
reconstruction. What is pinned here is that it changes NOTHING when there is no
partition to interpolate (the columns are then exact), and that it moves the
amplitude in the direction it claims when there is.
"""

import numpy as np
import pytest
import torch

from scatterem.simulation.eels.forward.prism_eels_image import prism_eels_image
from scatterem.simulation.eels.physics.transition_potentials import (
    build_transition_potentials,
    subshell_transitions,
)


def _toy(nz=3, gpts=(40, 40), seed=0):
    """Same small double-channeling setup the neighbouring EELS tests use."""
    torch.manual_seed(seed)
    sampling = (0.18, 0.18)
    gridsize = (gpts[0] * sampling[0], gpts[1] * sampling[1])
    ny, nx = gpts
    T = torch.exp(1j * 0.3 * torch.rand(nz, *gpts, dtype=torch.float64)).to(
        torch.complex128
    )
    transitions = subshell_transitions(8, 1, 0, 1.0, wavefunctions="hydrogenic")
    tp = build_transition_potentials(
        transitions, 8, gpts, sampling, eV=100_000.0, dtype=torch.complex128
    )
    qy, qx = np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx), indexing="ij")
    pq = torch.as_tensor((qy**2 + qx**2) <= 0.14**2, dtype=torch.complex128)
    pq = pq / (pq.abs().pow(2).sum() / pq.numel()).sqrt()
    sites = np.array([[0.45, 0.55, 0.5], [0.3, 0.4, 0.5]])
    scan = torch.tensor(
        [[7.0, 9.0], [15.0, 20.0], [3.0, 25.0], [20.0, 20.0]], dtype=torch.float64
    )
    common = dict(wavelength=0.0037, gridsize=gridsize, slice_distance=1.0)
    return pq, T, tp, sites, scan, common


def _call(pq, T, tp, sites, scan, common, **kw):
    return prism_eels_image(
        probe_q=pq,
        transmissions=T,
        transition_potentials=tp,
        sites=sites,
        scan_pixels=scan,
        detector_mrad=18.0,
        **common,
        **kw,
    )


@pytest.mark.parametrize("axis", [0, 1])
def test_qeels_spectrum_sums_back_to_the_scalar_map(axis):
    """Summing the resolved axis reproduces the map EXACTLY, not approximately."""
    args = _toy()
    scalar = _call(*args)
    spectrum, q_pix = _call(*args, qeels_axis=axis)

    assert spectrum.shape == (scalar.shape[0], q_pix.shape[0])
    # A few ULP of float64. The bound is what the regrouping costs (2e-16 here,
    # 4e-16 partitioned, which stacks more summation stages); a beam that were
    # actually dropped or double-counted would show up at ~1/n_q, i.e. 1e-2.
    torch.testing.assert_close(spectrum.sum(dim=-1), scalar, rtol=1e-15, atol=0.0)


@pytest.mark.parametrize("axis", [0, 1])
def test_qeels_bins_are_the_sorted_signed_frequencies(axis):
    """Each bin is one signed pixel frequency, sorted and distinct."""
    _, q_pix = _call(*_toy(), qeels_axis=axis)

    assert q_pix.ndim == 1
    assert torch.equal(q_pix, torch.sort(q_pix).values)
    assert torch.equal(q_pix, torch.unique(q_pix))
    # A centred round detector spans both signs of the resolved momentum.
    assert int(q_pix.min()) < 0 < int(q_pix.max())


def test_qeels_survives_the_scan_reshape():
    args = _toy()
    flat, q_pix = _call(*args, qeels_axis=0)
    shaped, q_pix2 = _call(*args, qeels_axis=0, scan_shape=(2, 2))

    assert shaped.shape == (2, 2, q_pix.shape[0])
    assert torch.equal(shaped.reshape(-1, q_pix.shape[0]), flat)
    assert torch.equal(q_pix2, q_pix)


def test_qeels_holds_under_partitioning():
    """The binning is post-|amp|^2, so partitioning cannot break the identity."""
    args = _toy()
    part = dict(partition={"n_radial": 2}, partition_s2={"n_radial": 2})
    scalar = _call(*args, **part)
    spectrum, _ = _call(*args, qeels_axis=1, **part)

    torch.testing.assert_close(spectrum.sum(dim=-1), scalar, rtol=1e-15, atol=0.0)


def test_mag_preserve_is_inert_without_a_partition():
    """With exact columns there is no interpolated magnitude to restore."""
    args = _toy()
    assert torch.equal(
        _call(*args, mag_preserve=True), _call(*args, mag_preserve=False)
    )


def test_mag_preserve_raises_the_partitioned_amplitude():
    """It exists because the complex NNW average shrinks |S|; check it does.

    The complex average of parent columns that disagree in phase is smaller in
    magnitude than the average of their magnitudes, so restoring the latter can
    only move the reconstructed intensity up. Asserted as a direction, not a
    value: the size of the deficit is what the option is there to measure.
    """
    args = _toy()
    part = dict(partition={"n_radial": 2}, partition_s2={"n_radial": 2})
    on = _call(*args, mag_preserve=True, **part)
    off = _call(*args, mag_preserve=False, **part)

    assert not torch.equal(on, off)
    assert float(on.sum()) > float(off.sum())
