"""`_crop_stack_or_none`'s flat-index gather is the advanced index."""

import numpy as np
import pytest
import torch

import scatterem.simulation.eels.forward.prism_eels_image as pei
from scatterem.simulation.eels.forward._detector_exit import _periodic_window_stack

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _windows(n_atoms, wy, wx, ny, nx, device, seed):
    rng = np.random.default_rng(seed)
    iy = _periodic_window_stack(rng.integers(0, ny, n_atoms), wy, ny, device)
    ix = _periodic_window_stack(rng.integers(0, nx, n_atoms), wx, nx, device)
    lin = (iy[:, :, None] * nx + ix[:, None, :]).reshape(-1, wy * wx)
    return iy, ix, lin


def _reference(S, iy, ix):
    return S[:, iy[:, :, None], ix[:, None, :]].permute(1, 0, 2, 3)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "C,ny,nx,wy,wx,na",
    [(29, 192, 192, 8, 8, 8), (21, 64, 48, 7, 5, 3), (3, 16, 16, 16, 16, 2)],
)
def test_lin_gather_equals_advanced_index(device, C, ny, nx, wy, wx, na):
    g = torch.Generator().manual_seed(C * ny + wx)
    S = torch.randn(C, ny, nx, dtype=torch.complex64, generator=g).to(device)
    iy, ix, lin = _windows(na, wy, wx, ny, nx, device, seed=na)
    ref = _reference(S, iy, ix)
    out = pei._crop_stack_or_none(S, iy, ix, wy, wx, lin)
    assert out is not None
    assert out.shape == ref.shape
    assert out.stride() == ref.stride()
    assert torch.equal(out, ref)
    # Also equal to the per-atom chained form the stack replaced.
    for j in range(na):
        assert torch.equal(out[j], S[:, iy[j]][:, :, ix[j]])


@pytest.mark.parametrize("device", DEVICES)
def test_noncontiguous_S_falls_back(device):
    S = torch.randn(48, 40, 5, dtype=torch.complex64).to(device).permute(2, 0, 1)
    assert not S.is_contiguous()
    iy, ix, lin = _windows(4, 6, 6, 48, 40, device, seed=1)
    out = pei._crop_stack_or_none(S, iy, ix, 6, 6, lin)
    assert out is not None
    assert torch.equal(out, _reference(S, iy, ix))


def test_transposed_flat_index_is_caught():
    """Teeth: a flat index that swaps the window axes must NOT pass the equality."""
    S = torch.randn(4, 32, 32, dtype=torch.complex64)
    iy, ix, _ = _windows(3, 5, 5, 32, 32, "cpu", seed=2)
    bad = (iy[:, None, :] * 32 + ix[:, :, None]).reshape(-1, 25)
    out = pei._crop_stack_or_none(S, iy, ix, 5, 5, bad)
    assert out is not None
    assert not torch.equal(out, _reference(S, iy, ix))


def test_single_atom_and_budget_still_decline(monkeypatch):
    S = torch.randn(4, 16, 16, dtype=torch.complex64)
    iy, ix, lin = _windows(1, 4, 4, 16, 16, "cpu", seed=3)
    assert pei._crop_stack_or_none(S, iy, ix, 4, 4, lin) is None
    iy, ix, lin = _windows(3, 4, 4, 16, 16, "cpu", seed=4)
    monkeypatch.setattr(pei, "_EELS_CROP_BLOCK_BYTES", 1)
    assert pei._crop_stack_or_none(S, iy, ix, 4, 4, lin) is None
