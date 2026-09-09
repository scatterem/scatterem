"""Reuse must preserve the original per-atom scientific calculation."""

import importlib
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scatterem.simulation.eels.transition_potentials import TransitionPotentials

module = importlib.import_module("scatterem.simulation.eels.prism_eels_image")
devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def specimen(device, dtype):
    ny, nx = 12, 16
    y, x = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float64),
        torch.arange(nx, device=device, dtype=torch.float64),
        indexing="ij",
    )
    gy, gx = (y + ny // 2) % ny - ny // 2, (x + nx // 2) % nx - nx // 2
    h = torch.exp(-(gy.square() + gx.square()) / 3)
    tp = TransitionPotentials(
        torch.stack([h, h * (gy + 1j * gx)]).to(dtype), 8, 100000.0, (0.5, 0.5)
    )
    t = torch.exp(
        1j * torch.stack([0.2 * torch.sin(y + j) * torch.cos(x / 2) for j in range(3)])
    ).to(dtype)
    probe = (gy.square() + gx.square() <= 16).to(dtype)
    sites = np.array(
        [
            (a / 8 + 0.01, (a * 0.37 + 0.97) % 1, (z + 0.1) / 3)
            for z in range(3)
            for a in range(8)
        ]
    )
    scan = torch.tensor([[0.0, 0.0], [3.25, 4.5], [11.0, 15.0]], device=device)
    return (probe, t, tp, sites, scan), dict(
        wavelength=0.037,
        gridsize=(6.0, 8.0),
        slice_distance=2.0,
        detector_mrad=22.0,
        prism_mask=False,
        inelastic_crop=2,
    )


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("legs", ["s1", "s2", "both"])
@pytest.mark.parametrize(
    "contract,resolved,chunk",
    [
        ("columns", None, 2),
        ("probe", "both", 2),
        ("columns", None, None),
        ("probe", None, None),
    ],
)
def test_reuse_preserves_output_and_reduces_reconstruction(
    monkeypatch, device, dtype, legs, contract, resolved, chunk
):
    args, options = specimen(device, dtype)
    options.update(contract=contract, qeels_axis=resolved, scan_chunk=chunk)
    if legs in ("s1", "both"):
        options["partition"] = {"n_radial": 2}
    if legs in ("s2", "both"):
        options["partition_s2"] = {"n_radial": 2}
    counts = []
    for cls in (module.PartitionedScatteringMatrix, module.DetectorExitSMatrix):
        old = cls._nnw_window

        def record(self, *a, _old=old, **kw):
            counts.append(tuple(a[0].shape))
            return _old(self, *a, **kw)

        monkeypatch.setattr(cls, "_nnw_window", record)
    monkeypatch.setattr(module, "_EELS_RECONSTRUCTION_BYTES", 0, raising=False)
    reference = module.prism_eels_image(*args, **options)
    n_reference = len(counts)
    counts.clear()
    monkeypatch.setattr(module, "_EELS_RECONSTRUCTION_BYTES", 1 << 30)
    actual = module.prism_eels_image(*args, **options)
    assert len(counts) == n_reference // 8
    if resolved:
        torch.testing.assert_close(actual[1], reference[1])
        actual, reference = actual[0], reference[0]
    tol = 2e-5 if dtype == torch.complex64 else 2e-11
    torch.testing.assert_close(
        actual, reference, rtol=tol, atol=tol * reference.abs().max()
    )


@pytest.mark.parametrize(
    "mag,dechirp", [(False, False), (True, False), (False, True), (True, True)]
)
def test_reuse_keeps_magnitude_and_dechirp(monkeypatch, mag, dechirp):
    args, opts = specimen("cpu", torch.complex128)
    opts.update(
        partition={"n_radial": 2},
        partition_s2={"n_radial": 2, "dechirp": dechirp},
        mag_preserve=mag,
    )
    monkeypatch.setattr(module, "_EELS_RECONSTRUCTION_BYTES", 0, raising=False)
    ref = module.prism_eels_image(*args, **opts)
    monkeypatch.setattr(module, "_EELS_RECONSTRUCTION_BYTES", 1 << 30)
    actual = module.prism_eels_image(*args, **opts)
    torch.testing.assert_close(actual, ref, rtol=2e-11, atol=2e-11 * ref.abs().max())


@pytest.mark.parametrize(
    "active,width,budget,focal,grad",
    [
        (0, 16, 1 << 30, 0, False),
        (1, 16, 1 << 30, 0, False),
        (2, 2, 1 << 30, 0, False),
        (8, 16, 0, 0, False),
        (8, 16, 1 << 30, 1, False),
        (8, 16, 1 << 30, 0, True),
    ],
)
def test_ineligible_reconstruction_does_no_work(active, width, budget, focal, grad):
    def forbidden(*args):
        pytest.fail("ineligible full-grid reconstruction was evaluated")

    matrix = SimpleNamespace(
        S=torch.ones(4, 16, 16, dtype=torch.complex64, requires_grad=grad),
        _backprop_slices=lambda: focal,
        _nnw_window=forbidden,
    )
    assert (
        module._reconstruct_slice_or_none(matrix, 20, active, width, width, budget)
        is None
    )


def test_cache_released_before_next_depth(monkeypatch):
    args, opts = specimen("cpu", torch.complex128)
    opts.update(partition={"n_radial": 2}, partition_s2={"n_radial": 2})
    refs = []
    build = module._reconstruct_slice_or_none

    def record(*args):
        result = build(*args)
        if result is not None:
            refs.append(weakref.ref(result))
        return result

    advance = module.PartitionedScatteringMatrix.advance_to

    def advance_checked(self, *a):
        assert all(ref() is None for ref in refs)
        return advance(self, *a)

    monkeypatch.setattr(module, "_reconstruct_slice_or_none", record)
    monkeypatch.setattr(
        module.PartitionedScatteringMatrix, "advance_to", advance_checked
    )
    module.prism_eels_image(*args, **opts)
    assert len(refs) == 6
    assert all(ref() is None for ref in refs)


def test_active_focal_path_preserved(monkeypatch):
    args, opts = specimen("cpu", torch.complex128)
    opts.update(
        partition={"n_radial": 2, "focal_backprop": "centroid"},
        partition_s2={"n_radial": 2, "focal_backprop": "centroid"},
    )
    monkeypatch.setattr(module, "_EELS_RECONSTRUCTION_BYTES", 0)
    ref = module.prism_eels_image(*args, **opts)
    monkeypatch.setattr(module, "_EELS_RECONSTRUCTION_BYTES", 1 << 30)
    actual = module.prism_eels_image(*args, **opts)
    torch.testing.assert_close(actual, ref, rtol=2e-11, atol=2e-11 * ref.abs().max())


def test_empty_scan_selections_do_not_reconstruct(monkeypatch):
    args, opts = specimen("cpu", torch.complex128)
    args = list(args)
    args[3] = np.array([[0.5, 0.5, 0.1]] * 8)  # Other slices are empty.
    args[4] = torch.zeros(1, 2)
    opts.update(
        prism_mask=True,
        interpolation_factor=2,
        partition={"n_radial": 2},
        partition_s2={"n_radial": 2},
    )

    def forbidden(*args, **kwargs):
        pytest.fail("an unobserved atom triggered reconstruction")

    monkeypatch.setattr(module.PartitionedScatteringMatrix, "_nnw_window", forbidden)
    monkeypatch.setattr(module.DetectorExitSMatrix, "_nnw_window", forbidden)
    assert torch.count_nonzero(module.prism_eels_image(*args, **opts)) == 0
