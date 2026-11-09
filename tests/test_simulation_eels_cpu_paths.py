"""CPU-reachable branches that the sim-eels-cpu lane was no longer counting.

A change made ``_abs_complex``'s hypot fast path CUDA-only, so the CPU lane lost
a line and a branch it used to get for free. These tests pin other eager-side
arms the lane can still reach -- magnitude-restore off, the partitioned-columns
type guard, qEELS binning, and the shift-table reuse across co-lateral atoms --
without touching CUDA / Triton / Warp wrappers.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

from scatterem.physics.propagation import propagator_kernel
from scatterem.simulation._wave.partitioned_smatrix import PartitionedScatteringMatrix
from scatterem.simulation._wave.smatrix import ScatteringMatrix
from scatterem.simulation.eels.forward._detector_exit import DetectorExitSMatrix
from scatterem.simulation.eels.forward.prism_eels import _partitioned_columns
from scatterem.simulation.eels.forward.prism_eels_image import (
    _periodic_window,
    _qeels_bins,
    prism_eels_image,
)
from scatterem.simulation.eels.physics.transition_potentials import TransitionPotentials

pei = importlib.import_module("scatterem.simulation.eels.forward.prism_eels_image")

_WAVELENGTH = 0.0037


def _tp(gpts, n_trans=2, seed=0) -> TransitionPotentials:
    g = torch.Generator().manual_seed(seed)
    arr = torch.randn(n_trans, *gpts, generator=g, dtype=torch.float64) + 1j * (
        torch.randn(n_trans, *gpts, generator=g, dtype=torch.float64)
    )
    return TransitionPotentials(
        array=arr.to(torch.complex128),
        Z=8,
        energy=100_000.0,
        sampling=(0.5, 0.5),
        energy_losses=np.full(n_trans, 532.0),
    )


def _transmissions(nz, gpts, seed=0, strength=0.25):
    g = torch.Generator().manual_seed(seed)
    return torch.exp(
        1j * strength * torch.rand(nz, *gpts, generator=g, dtype=torch.float64)
    ).to(torch.complex128)


def _aperture(gpts, radius=0.2):
    qy, qx = np.meshgrid(
        np.fft.fftfreq(gpts[0]), np.fft.fftfreq(gpts[1]), indexing="ij"
    )
    pq = torch.as_tensor((qy**2 + qx**2) <= radius**2, dtype=torch.complex128)
    return pq / (pq.abs().pow(2).sum() / pq.numel()).sqrt()


def _det_idx(gpts, radius=0.2):
    qy = torch.fft.fftfreq(gpts[0])
    qx = torch.fft.fftfreq(gpts[1])
    return torch.nonzero(
        qy[:, None] ** 2 + qx[None, :] ** 2 <= radius**2, as_tuple=False
    )


def _image_case(gpts=(16, 16), nz=2, sites=None):
    tp = _tp(gpts, 2, seed=5)
    T = _transmissions(nz, gpts, seed=6)
    pq = _aperture(gpts, 0.2)
    if sites is None:
        sites = np.array([[0.3, 0.3, 0.6], [0.7, 0.6, 0.6]])
    ry = (np.arange(4) + 0.5) * gpts[0] / 4
    rx = (np.arange(4) + 0.5) * gpts[1] / 4
    grid = np.stack(np.meshgrid(ry, rx, indexing="ij"), axis=-1).reshape(-1, 2)
    scan = torch.as_tensor(grid, dtype=torch.float64)
    common = dict(
        wavelength=_WAVELENGTH,
        gridsize=(gpts[0] * 0.5, gpts[1] * 0.5),
        slice_distance=1.0,
        detector_mrad=40.0,
    )
    return tp, T, pq, sites, scan, common


# --------------------------------------------------------------------------- #
# mag_preserve=False: the False arm of the partitioned magnitude restore
# --------------------------------------------------------------------------- #
def test_detector_exit_mag_preserve_raises_partitioned_column_amplitude():
    """Complex NNW averaging shrinks |S|; restoring |S| can only raise it."""
    gpts = (16, 16)
    T = _transmissions(2, gpts, seed=3)
    k = propagator_kernel(gpts, (8.0, 8.0), _WAVELENGTH, 1.0, dtype=torch.complex128)
    det = _det_idx(gpts)
    part = dict(n_radial=3)
    on = DetectorExitSMatrix(det, T, k, partition=part, mag_preserve=True)
    off = DetectorExitSMatrix(det, T, k, partition=part, mag_preserve=False)
    iy = _periodic_window(4, 6, gpts[0], T.device)
    ix = _periodic_window(9, 6, gpts[1], T.device)
    cols_on = on.columns_window(iy, ix)
    cols_off = off.columns_window(iy, ix)
    assert not torch.equal(cols_on, cols_off)
    assert float(cols_on.abs().sum()) > float(cols_off.abs().sum())


def test_partitioned_s1_mag_preserve_raises_window_amplitude():
    """Same direction claim on S1's reconstruct_columns_window."""
    gpts = (24, 24)
    T = _transmissions(2, gpts, seed=13)
    k = propagator_kernel(gpts, (8.0, 8.0), _WAVELENGTH, 1.0, dtype=torch.complex128)
    pq = _aperture(gpts, 0.2)
    common = dict(n_radial=3, focal_backprop=0.0)
    on = PartitionedScatteringMatrix(pq, T, k, mag_preserve=True, **common)
    off = PartitionedScatteringMatrix(pq, T, k, mag_preserve=False, **common)
    on.advance_to(1)
    off.advance_to(1)
    iy = torch.arange(4, 10)
    ix = torch.arange(11, 17)
    cols_on = on.reconstruct_columns_window(iy, ix)
    cols_off = off.reconstruct_columns_window(iy, ix)
    assert not torch.equal(cols_on, cols_off)
    assert float(cols_on.abs().sum()) > float(cols_off.abs().sum())


# --------------------------------------------------------------------------- #
# _partitioned_columns: the typed dispatch guard
# --------------------------------------------------------------------------- #
def test_partitioned_columns_rejects_an_exact_scattering_matrix():
    gpts = (16, 16)
    T = _transmissions(1, gpts, seed=1)
    k = propagator_kernel(gpts, (8.0, 8.0), _WAVELENGTH, 0.0, dtype=torch.complex128)
    sm = ScatteringMatrix(_aperture(gpts, 0.2), T, k)
    with pytest.raises(
        TypeError, match="partitioned dispatch requires a PartitionedScatteringMatrix"
    ):
        _partitioned_columns(sm)


# --------------------------------------------------------------------------- #
# qEELS binning (the axis!=None arm of _qeels_bins)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("axis", [0, 1])
def test_qeels_bins_are_sorted_signed_frequencies_on_the_resolved_axis(axis):
    """Each bin is one distinct signed pixel frequency along the chosen axis."""
    ny, nx = 16, 16
    det_idx = torch.tensor([[0, 0], [1, 2], [3, 1], [15, 14], [8, 7]])
    img, q_bin, n_q, q_pix = _qeels_bins(det_idx, axis, (ny, nx), P=4, device="cpu")
    assert img.shape == (4, n_q)
    assert q_bin is not None and q_bin.shape == (det_idx.shape[0],)
    assert n_q == int(q_pix.shape[0])
    assert torch.equal(q_pix, torch.sort(q_pix).values)
    assert torch.equal(q_pix, torch.unique(q_pix))
    nres = ny if axis == 0 else nx
    signed = (det_idx[:, axis] + nres // 2) % nres - nres // 2
    assert torch.equal(q_pix, torch.unique(signed, sorted=True))


def test_qeels_spectrum_under_partitioned_s2_sums_back_to_the_scalar_map():
    """Partitioned S2 takes the per-atom fold; summing q recovers the scalar map."""
    tp, T, pq, sites, scan, common = _image_case()
    part = dict(partition_s2=dict(n_radial=2))
    scalar = prism_eels_image(pq, T, tp, sites, scan, **common, **part)
    spectrum, q_pix = prism_eels_image(
        pq, T, tp, sites, scan, qeels_axis=0, **common, **part
    )
    assert spectrum.shape == (scalar.shape[0], q_pix.shape[0])
    torch.testing.assert_close(spectrum.sum(dim=-1), scalar, rtol=1e-15, atol=0.0)


# --------------------------------------------------------------------------- #
# Shift-table reuse across co-lateral atoms in different slices
# --------------------------------------------------------------------------- #
def test_co_lateral_atoms_reuse_the_shift_crop_stack(monkeypatch):
    """Identical lateral shifts across slices rebuild the crop table once."""
    gpts = (16, 16)
    # Same (y, x), different z -> two non-empty slices, one shift table.
    sites = np.array([[0.3, 0.3, 0.25], [0.3, 0.3, 0.75]])
    tp, T, pq, sites, scan, common = _image_case(gpts=gpts, nz=2, sites=sites)
    calls = {"n": 0}
    real = pei._shift_crop_stack

    def counted(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(pei, "_shift_crop_stack", counted)
    out = prism_eels_image(pq, T, tp, sites, scan, **common)
    assert calls["n"] == 1
    assert float(out.sum()) > 0.0
    assert torch.isfinite(out).all()
