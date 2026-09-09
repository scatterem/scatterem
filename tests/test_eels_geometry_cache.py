"""Invariant partitioned-S1 geometry must stay on its matrix's device."""

import numpy as np
import pytest
import torch

from scatterem.simulation.eels.prism_eels import PartitionedScatteringMatrix

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _matrix(device, dtype, mag_preserve=True):
    ny, nx = 12, 16
    y, x = torch.meshgrid(torch.arange(ny), torch.arange(nx), indexing="ij")
    gy, gx = (y + ny // 2) % ny - ny // 2, (x + nx // 2) % nx - nx // 2
    probe = (gy.square() + gx.square() <= 25).to(device=device, dtype=dtype)
    transmissions = torch.exp(0.2j * torch.sin(y.double()) * torch.cos(x.double()))
    transmissions = transmissions[None].repeat(2, 1, 1).to(device=device, dtype=dtype)
    kernel = torch.exp(-0.03j * (gy.square() + gx.square())).to(
        device=device, dtype=dtype
    )
    return PartitionedScatteringMatrix(
        probe, transmissions, kernel, n_radial=2, mag_preserve=mag_preserve
    )


@pytest.mark.parametrize("device", DEVICES)
def test_signed_beam_coordinates_are_reused_across_depths(device):
    sm = _matrix(device, torch.complex128)
    gy, gx = sm._signed_beam_freqs()
    sm.advance_to(1)
    next_gy, next_gx = sm._signed_beam_freqs()
    assert next_gy is gy and next_gx is gx
    assert gy.dtype == gx.dtype == torch.float64
    assert gy.device == gx.device == sm.S.device
    torch.testing.assert_close(
        gy, ((sm._by.double() + sm.ny // 2) % sm.ny) - sm.ny // 2, rtol=0, atol=0
    )
    torch.testing.assert_close(
        gx, ((sm._bx.double() + sm.nx // 2) % sm.nx) - sm.nx // 2, rtol=0, atol=0
    )


@pytest.mark.parametrize("device", DEVICES)
def test_reconstruction_helpers_do_not_repeat_parent_host_conversion(
    device, monkeypatch
):
    sm = _matrix(device, torch.complex128)
    iy = torch.tensor([11, 0, 1], device=device)
    ix = torch.tensor([15, 0, 1, 2], device=device)
    scan = torch.tensor([[1.0, 2.0]], device=device)
    conversions = []
    original = torch.as_tensor

    def record_host_conversion(data, *args, **kwargs):
        if isinstance(data, np.ndarray):
            conversions.append(data)
        return original(data, *args, **kwargs)

    monkeypatch.setattr(torch, "as_tensor", record_host_conversion)
    for depth in (0, 1, 2):
        sm.advance_to(depth)
        sm.reconstruct_columns_window(iy, ix)
        sm.reconstruct_columns()
        sm.probe_at_current_plane(scan)
        sm.probe_at_current_plane(scan + 0.25)
    assert not conversions, (
        "Invariant parent coordinates were converted again after construction"
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("mag_preserve", [False, True])
def test_cached_window_matches_uncached_geometry_and_gradients(
    device, dtype, mag_preserve
):
    sm = _matrix(device, dtype, mag_preserve)
    sm.advance_to(2)
    iy = torch.tensor([11, 0, 1], device=device)
    ix = torch.tensor([15, 0, 1, 2], device=device)
    sw = sm.S[:, iy][:, :, ix].detach().requires_grad_()
    # Original uncached geometry and arithmetic provide the numerical oracle.
    gp = torch.as_tensor(sm._parent_signed, dtype=torch.float64, device=device)
    gy = ((sm._by.double() + sm.ny // 2) % sm.ny) - sm.ny // 2
    gx = ((sm._bx.double() + sm.nx // 2) % sm.nx) - sm.nx // 2
    yy, xx = iy.double().view(1, -1, 1), ix.double().view(1, 1, -1)
    detilt = torch.exp(
        -2j
        * torch.pi
        * (gp[:, 0, None, None] * yy / sm.ny + gp[:, 1, None, None] * xx / sm.nx)
    ).to(dtype)
    sd = sw * detilt
    expected = torch.einsum("bp,pwv->bwv", sm._w, sd)
    if mag_preserve:
        mag = torch.einsum("bp,pwv->bwv", sm._w, sd.abs().to(dtype)).real
        expected = mag * (expected / expected.abs().clamp_min(1e-20))
    expected = expected * torch.exp(
        2j
        * torch.pi
        * (gy[:, None, None] * yy / sm.ny + gx[:, None, None] * xx / sm.nx)
    ).to(dtype)
    actual = sm._nnw_window(sw, iy, ix)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(
        actual.abs().square().sum(), sw, retain_graph=True
    )[0]
    expected_grad = torch.autograd.grad(expected.abs().square().sum(), sw)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
