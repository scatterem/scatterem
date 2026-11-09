"""prism_eels_image's deferred scan accumulation is the per-call ``index_add_``, bitwise.

``_OrderedScanAccumulator`` replaces one ``img.index_add_`` per atom
with a single ``cumsum`` over a ``(calls + 1, *img.shape)`` matrix.  That is only
exact because the rows are summed in CALL order; these tests pin that with keys that
overlap across calls and values spread over 16 decades, so any reordering (reversed
rows, a tree ``sum``, one atomic scatter) changes the float64 result.
"""

import pytest
import torch

import scatterem.simulation.eels.forward._scan_accumulator as sa
from scatterem.simulation._backends import EELS_SCAN_ACC
from scatterem.utils.env import resolve_backend

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA path only")


def _calls(P, K, Pm, seed, n_q=None):
    g = torch.Generator(device="cpu").manual_seed(seed)
    tail = () if n_q is None else (n_q,)
    sels = [torch.randperm(P, generator=g)[:Pm].sort().values.cuda() for _ in range(K)]
    vals = [
        (
            torch.rand((Pm, *tail), generator=g, dtype=torch.float64)
            * 10.0 ** torch.randint(-8, 8, (Pm, *tail), generator=g)
        ).cuda()
        for _ in range(K)
    ]
    return sels, vals


def _reference(P, sels, vals):
    img = torch.zeros((P, *vals[0].shape[1:]), dtype=torch.float64, device="cuda")
    for s, v in zip(sels, vals, strict=True):
        img.index_add_(0, s, v)
    return img


def _accumulate(P, sels, vals, group):
    img = torch.zeros((P, *vals[0].shape[1:]), dtype=torch.float64, device="cuda")
    acc = sa._OrderedScanAccumulator(img)
    assert acc.deferred
    for k in range(0, len(sels), group):
        ss, vv = sels[k : k + group], vals[k : k + group]
        if len(ss) == 1:
            acc.add(ss, vv)
        else:
            acc.add(ss, torch.stack(vv), torch.cat(ss), torch.cat(vv))
    return acc.flush()


@cuda
@pytest.mark.parametrize(
    "P,K,Pm", [(256, 128, 16), (1024, 40, 64), (64, 9, 64), (256, 3, 16)]
)
@pytest.mark.parametrize("group", [1, 8])
@pytest.mark.parametrize("n_q", [None, 5])
def test_bitwise_equal_to_per_call_index_add(P, K, Pm, group, n_q):
    sels, vals = _calls(P, K, Pm, seed=P + K + Pm, n_q=n_q)
    ref = _reference(P, sels, vals)
    f0 = sa._SCAN_FLUSHES[0]
    got = _accumulate(P, sels, vals, group)
    assert torch.equal(got, ref)
    # K >= the floor takes the cumsum; below it the calls are replayed in order
    assert sa._SCAN_FLUSHES[0] - f0 == int(K >= sa._EELS_SCAN_ACC_MIN_CALLS)


@cuda
def test_grouped_calls_without_a_concat_are_exact():
    sels, vals = _calls(256, 32, 16, seed=5)
    acc = sa._OrderedScanAccumulator(
        torch.zeros(256, dtype=torch.float64, device="cuda")
    )
    for k in range(0, 32, 4):
        acc.add(sels[k : k + 4], vals[k : k + 4])
    assert torch.equal(acc.flush(), _reference(256, sels, vals))


@cuda
def test_early_flush_under_budget_is_exact(monkeypatch):
    monkeypatch.setattr(sa, "_EELS_SCAN_ACC_BLOCK_BYTES", 256 * 8 * 12)
    sels, vals = _calls(256, 128, 16, seed=7)
    f0 = sa._SCAN_FLUSHES[0]
    got = _accumulate(256, sels, vals, group=1)
    assert sa._SCAN_FLUSHES[0] - f0 > 1  # really flushed more than once
    assert torch.equal(got, _reference(256, sels, vals))


def test_row_declines_cpu_and_takes_cuda():
    cpu = torch.zeros(8, dtype=torch.float64)
    assert resolve_backend(EELS_SCAN_ACC, None, cpu) == "index_add"
    assert not sa._OrderedScanAccumulator(cpu).deferred
    if torch.cuda.is_available():
        assert resolve_backend(EELS_SCAN_ACC, None, cpu.cuda()) == "scan"


def test_index_add_path_is_the_per_call_scatter():
    g = torch.Generator().manual_seed(3)
    sels = [torch.randperm(32, generator=g)[:8] for _ in range(12)]
    vals = [torch.rand(8, generator=g, dtype=torch.float64) for _ in range(12)]
    acc = sa._OrderedScanAccumulator(torch.zeros(32, dtype=torch.float64))
    for s, v in zip(sels, vals, strict=True):
        acc.add((s,), (v,))
    ref = torch.zeros(32, dtype=torch.float64)
    for s, v in zip(sels, vals, strict=True):
        ref.index_add_(0, s, v)
    assert torch.equal(acc.flush(), ref)


@cuda
def test_backend_row_can_force_index_add(monkeypatch):
    monkeypatch.setattr(sa, "resolve_backend", lambda *a, **k: "index_add")
    acc = sa._OrderedScanAccumulator(torch.zeros(8, dtype=torch.float64, device="cuda"))
    assert not acc.deferred


@cuda
def test_order_matters_so_the_test_has_teeth():
    """The same rows summed in any other order are NOT equal -- the gate can fail."""
    P, K, Pm = 256, 128, 64
    sels, vals = _calls(P, K, Pm, seed=11)
    ref = _reference(P, sels, vals)
    D = torch.zeros(K, P, dtype=torch.float64, device="cuda")
    for k, (s, v) in enumerate(zip(sels, vals, strict=True)):
        D[k, s] = v
    assert torch.equal(torch.cumsum(D, 0)[-1], ref)
    assert not torch.equal(torch.cumsum(D.flip(0), 0)[-1], ref)
    assert not torch.equal(D.sum(0), ref)


@cuda
@pytest.mark.parametrize("prism_mask,f,crop", [(True, 2, 4), (False, 1, None)])
def test_prism_eels_image_reaches_the_scan_and_is_unchanged(
    monkeypatch, prism_mask, f, crop
):
    """End to end: the cumsum flush is TAKEN and the image is the index_add_ one."""
    import numpy as np

    from scatterem.simulation.eels.forward.prism_eels_image import prism_eels_image
    from scatterem.simulation.eels.physics.transition_potentials import (
        build_transition_potentials,
        subshell_transitions,
    )

    torch.manual_seed(0)
    gpts, sampling = (48, 48), (0.18, 0.18)
    T = torch.exp(1j * 0.3 * torch.rand(3, *gpts, dtype=torch.float64))
    tr = subshell_transitions(8, 1, 0, 1.0, wavefunctions="hydrogenic")
    tp = build_transition_potentials(
        tr, 8, gpts, sampling, eV=100_000.0, dtype=torch.complex64
    )
    tp.array = tp.array.cuda()
    qy, qx = np.meshgrid(np.fft.fftfreq(48), np.fft.fftfreq(48), indexing="ij")
    pq = torch.as_tensor((qy**2 + qx**2) <= 0.14**2, dtype=torch.complex64)
    rng = np.random.default_rng(1)
    sites = np.column_stack([rng.random((12, 2)).repeat(2, 0), rng.random(24)])
    yy, xx = torch.meshgrid(
        torch.arange(0, 48, 4.0), torch.arange(0, 48, 4.0), indexing="ij"
    )
    scan = torch.stack([yy.reshape(-1), xx.reshape(-1)], 1).double().cuda()

    def run():
        return prism_eels_image(
            pq.cuda(),
            T.to(torch.complex64).cuda(),
            tp,
            sites,
            scan,
            wavelength=0.0037,
            gridsize=(48 * 0.18, 48 * 0.18),
            slice_distance=1.0,
            detector_mrad=18.0,
            interpolation_factor=f,
            inelastic_crop=crop,
            prism_mask=prism_mask,
        )

    f0 = sa._SCAN_FLUSHES[0]
    on = run()
    assert sa._SCAN_FLUSHES[0] == f0 + 1
    monkeypatch.setattr(sa, "resolve_backend", lambda *a, **k: "index_add")
    off = run()
    assert sa._SCAN_FLUSHES[0] == f0 + 1
    assert torch.equal(on, off)
