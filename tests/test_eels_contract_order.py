"""New test: the two contraction orders must agree exactly (they are a reassociation)."""

import functools
import sys

import numpy as np
import pytest
import torch

from scatterem.simulation.eels.forward.prism_eels_image import prism_eels_image
from scatterem.simulation.eels.physics.transition_potentials import (
    build_transition_potentials,
    subshell_transitions,
)

pei = sys.modules[prism_eels_image.__module__]


def _case(prism_mask, f):
    torch.manual_seed(0)
    gpts = (48, 48)
    sampling = (0.18, 0.18)
    ny, nx = gpts
    T = torch.exp(1j * 0.3 * torch.rand(3, *gpts, dtype=torch.float64)).to(
        torch.complex128
    )
    transitions = subshell_transitions(8, 1, 0, 1.0, wavefunctions="hydrogenic")
    tp = build_transition_potentials(
        transitions, 8, gpts, sampling, eV=100_000.0, dtype=torch.complex128
    )
    qy, qx = np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx), indexing="ij")
    pq = torch.as_tensor((qy**2 + qx**2) <= 0.14**2, dtype=torch.complex128)
    pq = pq / (pq.abs().pow(2).sum() / pq.numel()).sqrt()
    sites = np.array([[0.45, 0.55, 0.5], [0.3, 0.4, 0.5], [0.62, 0.18, 0.83]])
    scan = torch.tensor(
        [[7.0, 9.0], [15.0, 20.0], [3.0, 25.0], [30.0, 11.0]], dtype=torch.float64
    )
    common = dict(
        wavelength=0.0037,
        gridsize=(ny * sampling[0], nx * sampling[1]),
        slice_distance=1.0,
        detector_mrad=18.0,
        interpolation_factor=f,
        inelastic_crop=None,
        prism_mask=prism_mask,
    )
    return pq, T, tp, sites, scan, common


@functools.cache
def _image(order, prism_mask, f):
    """``prism_eels_image`` on ``_case(prism_mask, f)`` with one contraction order.

    Two tests read the ``"window"`` and the ``"probe"`` image of the same case.
    The cache computes each image once per process. The tests only read the
    result, so they can share it.
    """
    args = _case(prism_mask, f)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pei, "_EELS_CONTRACT_ORDER", order)
        return prism_eels_image(*args[:5], **args[5])


@pytest.mark.parametrize("prism_mask,f", [(False, 1), (True, 2)])
def test_contract_orders_agree(prism_mask, f):
    ref = _image("window", prism_mask, f)
    new = _image("probe", prism_mask, f)
    assert torch.allclose(ref, new, rtol=1e-9, atol=1e-11 * float(ref.abs().max())), (
        (ref - new).abs().max().item()
    )


def test_auto_matches_window():
    ref = _image("window", False, 1)
    auto = _image("auto", False, 1)
    assert torch.allclose(ref, auto, rtol=1e-9, atol=1e-11 * float(ref.abs().max()))


def test_probe_order_respects_trans_chunking(monkeypatch):
    full = _image("probe", False, 1)
    args = _case(False, 1)
    monkeypatch.setattr(pei, "_EELS_CONTRACT_ORDER", "probe")
    monkeypatch.setattr(pei, "_EELS_TRANS_BLOCK_BYTES", 1)  # force tblk == 1
    chunked = prism_eels_image(*args[:5], **args[5])
    assert torch.allclose(
        full, chunked, rtol=1e-9, atol=1e-11 * float(full.abs().max())
    )
