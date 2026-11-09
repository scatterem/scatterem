"""Host-side dispatch helpers and the S2 adjoint of the PRISM EELS *image* path.

``prism_eels_image`` reassociates its per-atom contraction several ways and
batches the atom axis wherever the shapes allow.  Every one of those is claimed
to be bit-identical to the per-atom spelling it replaces, so the tests below
drive each branch and compare it against that spelling rather than merely
running it.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

from scatterem.physics.fourier import fourier_shift
from scatterem.physics.propagation import propagator_kernel
from scatterem.simulation._wave.exit_walk import _slice_exit_wave
from scatterem.simulation.eels.forward._detector_exit import (
    _conj_propagate,
    _periodic_window,
    _periodic_window_stack,
)
from scatterem.simulation.eels.forward.prism_eels_image import (
    DetectorExitSMatrix,
    _coeff_stack_or_none,
    _contract_orders,
    _crop_stack_or_none,
    _hs1_stack_or_none,
    _probe_batch_size,
    _shift_crop_stack,
    prism_eels_image,
)
from scatterem.simulation.eels.physics.transition_potentials import TransitionPotentials

# The package ``__init__`` re-exports the *function* under the module's own name,
# so the module object has to be fetched explicitly to patch its constants.
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


# --------------------------------------------------------------------------- #
# Window indexing
# --------------------------------------------------------------------------- #
def test_periodic_window_are_centred_and_wrap_periodically():
    idx = _periodic_window(1, 4, 8, torch.device("cpu"))
    assert idx.tolist() == [7, 0, 1, 2]  # centre at 1, half-width 2, wrapped
    odd = _periodic_window(5, 3, 8, torch.device("cpu"))
    assert odd.tolist() == [4, 5, 6]
    whole = _periodic_window(3, 8, 8, torch.device("cpu"))
    assert sorted(whole.tolist()) == list(range(8))


def test_periodic_window_stack_is_the_per_atom_form():
    centres = np.array([0, 5, 7])
    stack = _periodic_window_stack(centres, 4, 8, torch.device("cpu"))
    assert stack.shape == (3, 4)
    for j, c in enumerate(centres):
        assert torch.equal(
            stack[j], _periodic_window(int(c), 4, 8, torch.device("cpu"))
        )


# --------------------------------------------------------------------------- #
# Contraction-order dispatch
# --------------------------------------------------------------------------- #
def test_contract_orders_can_be_forced(monkeypatch):
    monkeypatch.setattr(pei, "_EELS_CONTRACT_ORDER", "window")
    assert _contract_orders([1, 2, 3], 4, 5, 6, 7) == ["window"] * 3
    monkeypatch.setattr(pei, "_EELS_CONTRACT_ORDER", "probe")
    assert _contract_orders([1, 2, 3], 4, 5, 6, 7) == ["probe"] * 3


def test_auto_contract_order_switches_from_probe_to_window_with_the_scan_count():
    n_trans, ndet, nbeams, W = 27, 37, 49, 16384
    orders = _contract_orders([1, 16, 64, 10**6], n_trans, ndet, nbeams, W)
    assert orders[:3] == ["probe", "probe", "probe"]
    assert orders[-1] == "window"  # a huge scan count amortises the reuse
    # The crossover is monotone: once "window" wins it stays won.
    seq = _contract_orders(list(range(1, 4000, 37)), n_trans, ndet, nbeams, W)
    flipped = [i for i in range(1, len(seq)) if seq[i] != seq[i - 1]]
    assert len(flipped) <= 1


def test_auto_contract_order_matches_its_own_cost_model():
    n_trans, ndet, nbeams, W = 3, 5, 7, 11
    for pm in (1, 4, 16, 64, 256):
        cost_window = pei._EELS_WINDOW_COST_SCALE * (
            n_trans * ndet * nbeams * W + n_trans * pm * ndet * nbeams
        )
        cost_probe = pm * nbeams * W + n_trans * pm * W * ndet
        expected = "probe" if cost_probe <= cost_window else "window"
        assert _contract_orders([pm], n_trans, ndet, nbeams, W) == [expected]


# --------------------------------------------------------------------------- #
# Batched-stack helpers: every "None" condition, and the batched value itself
# --------------------------------------------------------------------------- #
def test_crop_stack_matches_the_per_atom_gathers():
    torch.manual_seed(0)
    S = torch.randn(3, 8, 8, dtype=torch.complex128)
    iy = torch.stack([_periodic_window(c, 4, 8, S.device) for c in (1, 6)])
    ix = torch.stack([_periodic_window(c, 4, 8, S.device) for c in (7, 3)])
    stack = _crop_stack_or_none(S, iy, ix, 4, 4)
    assert stack is not None and stack.shape == (2, 3, 4, 4)
    for j in range(2):
        assert torch.equal(stack[j], S[:, iy[j]][:, :, ix[j]])


def test_crop_stack_declines_for_one_atom_and_over_budget(monkeypatch):
    S = torch.zeros(3, 8, 8, dtype=torch.complex128)
    one = torch.arange(4).view(1, 4)
    assert _crop_stack_or_none(S, one, one, 4, 4) is None
    two = torch.arange(4).view(1, 4).repeat(2, 1)
    monkeypatch.setattr(pei, "_EELS_CROP_BLOCK_BYTES", 0)
    assert _crop_stack_or_none(S, two, two, 4, 4) is None


def test_hs1_stack_matches_the_per_atom_product_and_declines_when_it_must(
    monkeypatch,
):
    torch.manual_seed(0)
    h = torch.randn(2, 3, 4, 4, dtype=torch.complex128)  # (na, n_trans, wy, wx)
    s1 = torch.randn(2, 5, 4, 4, dtype=torch.complex128)  # (na, Nbeams, wy, wx)
    stack = _hs1_stack_or_none(h, s1)
    assert stack is not None and stack.shape == (2, 3, 5, 4, 4)
    for j in range(2):
        assert torch.equal(stack[j], h[j][:, None] * s1[j][None])
    assert _hs1_stack_or_none(h, None) is None  # partitioned S1
    assert _hs1_stack_or_none(h[:1], s1[:1]) is None  # a single atom
    monkeypatch.setattr(pei, "_EELS_HS1_BLOCK_BYTES", 0)
    assert _hs1_stack_or_none(h, s1) is None


def test_coeff_stack_matches_the_per_atom_gather_and_declines_when_it_must(
    monkeypatch,
):
    torch.manual_seed(0)
    coeffs = torch.randn(10, 4, dtype=torch.complex128)
    sel = torch.tensor([0, 3, 7, 2, 9])
    counts = [2, 3]
    stack = _coeff_stack_or_none(coeffs, sel, counts)
    assert stack is not None and len(stack) == 2
    assert torch.equal(stack[0], coeffs.index_select(0, sel[:2]))
    assert torch.equal(stack[1], coeffs.index_select(0, sel[2:]))
    assert _coeff_stack_or_none(coeffs, None, counts) is None  # prism_mask off
    assert _coeff_stack_or_none(coeffs, sel, [5]) is None  # one atom
    assert _coeff_stack_or_none(coeffs, sel[:0], [0, 0]) is None  # nothing selected
    monkeypatch.setattr(pei, "_EELS_COEFF_BLOCK_BYTES", 0)
    assert _coeff_stack_or_none(coeffs, sel, counts) is None


def test_probe_batch_size_accepts_a_uniform_slice_and_refuses_everything_else(
    monkeypatch,
):
    S1 = torch.zeros(2, 3, 4, 4, dtype=torch.complex128)
    S2 = torch.zeros(2, 5, 4, 4, dtype=torch.complex128)
    sel = torch.arange(8)
    kw = dict(na=2, n_trans=3, ndet=5, W=16)
    assert _probe_batch_size(["probe"] * 2, [4, 4], sel, S1, S2, **kw) == 4
    # Each refusal below is a property of the slice, not a tuning choice.
    assert _probe_batch_size(["probe"], [4], sel, S1, S2, **{**kw, "na": 1}) is None
    assert _probe_batch_size(["probe"] * 2, [4, 4], None, S1, S2, **kw) is None
    assert _probe_batch_size(["probe"] * 2, [4, 4], sel, None, S2, **kw) is None
    assert _probe_batch_size(["probe"] * 2, [4, 4], sel, S1, None, **kw) is None
    assert _probe_batch_size(["probe", "window"], [4, 4], sel, S1, S2, **kw) is None
    assert _probe_batch_size(["probe"] * 2, [4, 5], sel, S1, S2, **kw) is None
    assert _probe_batch_size(["probe"] * 2, [0, 0], sel, S1, S2, **kw) is None
    assert _probe_batch_size(["probe"] * 2, [4], sel, S1, S2, **kw) is None
    monkeypatch.setattr(pei, "_EELS_TRANS_BLOCK_BYTES", 0)
    assert _probe_batch_size(["probe"] * 2, [4, 4], sel, S1, S2, **kw) is None


@pytest.mark.parametrize("na", [1, 3])
def test_shift_crop_stack_matches_the_per_atom_shift(na, monkeypatch):
    torch.manual_seed(0)
    gpts = (8, 8)
    H = torch.randn(2, *gpts, dtype=torch.complex128)
    Hq = torch.fft.fft2(H, dim=(-2, -1))
    ky = torch.fft.fftfreq(gpts[0], dtype=torch.float64).view(gpts[0], 1)
    kx = torch.fft.fftfreq(gpts[1], dtype=torch.float64).view(1, gpts[1])
    cy = _periodic_window(gpts[0] // 2, 4, gpts[0], H.device)
    cx = _periodic_window(gpts[1] // 2, 4, gpts[1], H.device)
    shifts = np.array([[0.3, -1.7], [4.0, 2.5], [-2.25, 0.0]])[:na]
    got = _shift_crop_stack(Hq, ky, kx, shifts, cy, cx, H.dtype)
    assert got.shape == (na, 2, 4, 4)
    for j in range(na):
        ref = fourier_shift(H, (shifts[j, 0], shifts[j, 1]))[:, cy][:, :, cx]
        torch.testing.assert_close(got[j], ref)
    if na > 1:  # force the chunked branch and demand the same answer
        monkeypatch.setattr(pei, "_EELS_SHIFT_BLOCK_BYTES", 1)
        torch.testing.assert_close(
            _shift_crop_stack(Hq, ky, kx, shifts, cy, cx, H.dtype), got
        )


# --------------------------------------------------------------------------- #
# The detector (S2) scattering matrix
# --------------------------------------------------------------------------- #
def _det_idx(gpts, n=6):
    ys = torch.tensor([0, 1, gpts[0] - 1, 0, 2, gpts[0] - 2][:n])
    xs = torch.tensor([0, gpts[1] - 1, 1, 3, 0, 2][:n])
    return torch.stack([ys, xs], dim=1)


def test_conj_propagate_inverts_the_fresnel_step():
    torch.manual_seed(0)
    gpts = (8, 8)
    k = propagator_kernel(gpts, (4.0, 4.0), _WAVELENGTH, 1.3, dtype=torch.complex128)
    w = torch.randn(2, *gpts, dtype=torch.complex128)
    fwd = torch.fft.ifft2(torch.fft.fft2(w, dim=(-2, -1)) * k, dim=(-2, -1))
    torch.testing.assert_close(_conj_propagate(fwd, k), w)


@pytest.mark.parametrize("start", [0, 1, 2])
def test_detector_smatrix_is_the_adjoint_of_the_forward_exit(start):
    """sum_r S2_d(r) w(r) must equal fft(forward_exit(w))[d] for every beam."""
    torch.manual_seed(0)
    gpts = (8, 8)
    T = _transmissions(3, gpts, seed=1)
    k = propagator_kernel(gpts, (4.0, 4.0), _WAVELENGTH, 1.0, dtype=torch.complex128)
    det_idx = _det_idx(gpts)
    S2 = DetectorExitSMatrix(det_idx, T, k)
    S2.peel_to(start)
    assert S2._start == start
    w = torch.randn(*gpts, dtype=torch.complex128)
    got = torch.einsum("dyx,yx->d", S2.S, w)
    exit_wave = _slice_exit_wave(w[None].clone(), T, k, start, None)[0]
    ref = torch.fft.fft2(exit_wave, dim=(-2, -1))[det_idx[:, 0], det_idx[:, 1]]
    torch.testing.assert_close(got, ref, rtol=1e-9, atol=1e-12)


def test_detector_smatrix_columns_window_is_a_plain_crop_when_exact():
    gpts = (8, 8)
    T = _transmissions(2, gpts, seed=2)
    k = propagator_kernel(gpts, (4.0, 4.0), _WAVELENGTH, 1.0, dtype=torch.complex128)
    S2 = DetectorExitSMatrix(_det_idx(gpts), T, k)
    iy = _periodic_window(2, 4, gpts[0], T.device)
    ix = _periodic_window(5, 4, gpts[1], T.device)
    assert not S2.partitioned
    assert torch.equal(S2.columns_window(iy, ix), S2.S[:, iy][:, :, ix])


def test_partitioned_detector_smatrix_builds_on_parent_beams_only():
    gpts = (16, 16)
    T = _transmissions(2, gpts, seed=3)
    k = propagator_kernel(gpts, (8.0, 8.0), _WAVELENGTH, 1.0, dtype=torch.complex128)
    qy = torch.fft.fftfreq(gpts[0])
    qx = torch.fft.fftfreq(gpts[1])
    q2 = qy[:, None] ** 2 + qx[None, :] ** 2
    det_idx = torch.nonzero(q2 <= 0.2**2, as_tuple=False)
    S2 = DetectorExitSMatrix(det_idx, T, k, partition=dict(n_radial=3))
    assert S2.partitioned
    assert S2.S.shape[0] < det_idx.shape[0]  # only the parents are propagated
    iy = _periodic_window(4, 6, gpts[0], T.device)
    ix = _periodic_window(9, 6, gpts[1], T.device)
    cols = S2.columns_window(iy, ix)
    assert cols.shape == (det_idx.shape[0], 6, 6)
    assert torch.isfinite(cols.abs()).all()


def _vacuum_exit_case(n=96, nz=4):
    """Vacuum, where the exact S2 column is a pure plane wave.

    A correct de-tilt leaves every parent column spatially constant, so the
    natural-neighbour reconstruction has nothing to interpolate.
    """
    T = torch.ones(nz, n, n, dtype=torch.complex128)
    k = propagator_kernel(
        (n, n), (n * 0.2, n * 0.2), _WAVELENGTH, 1.0, dtype=torch.complex128
    )
    qy, qx = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="ij")
    det = torch.nonzero(torch.as_tensor((qy**2 + qx**2) <= 0.18**2), as_tuple=False)
    iy = _periodic_window(30, 24, n, T.device)
    ix = _periodic_window(40, 24, n, T.device)
    exact = DetectorExitSMatrix(det, T, k)
    exact.peel_to(1)
    return det, T, k, iy, ix, exact.columns_window(iy, ix)


@pytest.mark.parametrize("mag_preserve", [True, False])
def test_partitioned_detector_smatrix_is_near_exact_in_vacuum(mag_preserve):
    """Regression: the de-tilt once DOUBLED the S2 carrier instead of removing it.

    An S2 column carries exp(-2 pi i g.r) (it is built from the conjugate
    detector plane wave), so the de-tilt must multiply by exp(+2 pi i g.r).
    With the sign inverted the relative error here was 1.22, worse than
    predicting zero; corrected it is 4e-4.
    """
    det, T, k, iy, ix, exact = _vacuum_exit_case()
    S2 = DetectorExitSMatrix(
        det, T, k, partition=dict(n_radial=4), mag_preserve=mag_preserve
    )
    S2.peel_to(1)
    err = float((S2.columns_window(iy, ix) - exact).norm() / exact.norm())
    assert err < 1e-2, f"partitioned S2 is not near-exact in vacuum: {err}"


@pytest.mark.parametrize("mag_preserve", [True, False])
def test_partitioned_detector_smatrix_dechirp_is_exact_in_vacuum(mag_preserve):
    """With the vacuum propagator phase removed, every parent column is the
    same constant, so the reconstruction is exact up to rounding."""
    det, T, k, iy, ix, exact = _vacuum_exit_case()
    S2 = DetectorExitSMatrix(
        det,
        T,
        k,
        partition=dict(n_radial=4, dechirp=True),
        mag_preserve=mag_preserve,
    )
    S2.peel_to(1)
    err = float((S2.columns_window(iy, ix) - exact).norm() / exact.norm())
    assert err < 1e-9, f"de-chirped partitioned S2 is not exact in vacuum: {err}"


# --------------------------------------------------------------------------- #
# prism_eels_image option branches
# --------------------------------------------------------------------------- #
def _image_case(gpts=(16, 16), nz=2, n_sites=2):
    tp = _tp(gpts, 2, seed=5)
    T = _transmissions(nz, gpts, seed=6)
    pq = _aperture(gpts, 0.2)
    sites = np.array([[0.3, 0.3, 0.6], [0.7, 0.6, 0.6]])[:n_sites]
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


def test_image_scan_shape_reshapes_the_flat_map():
    tp, T, pq, sites, scan, common = _image_case()
    flat = prism_eels_image(pq, T, tp, sites, scan, **common)
    shaped = prism_eels_image(pq, T, tp, sites, scan, scan_shape=(4, 4), **common)
    assert flat.shape == (16,)
    assert shaped.shape == (4, 4)
    torch.testing.assert_close(shaped.reshape(-1), flat)


def test_image_without_the_prism_mask_collects_every_scan_position():
    tp, T, pq, sites, scan, common = _image_case()
    masked = prism_eels_image(
        pq, T, tp, sites, scan, interpolation_factor=2, prism_mask=True, **common
    )
    unmasked = prism_eels_image(
        pq, T, tp, sites, scan, interpolation_factor=2, prism_mask=False, **common
    )
    # The mask sets positions outside each atom's L/2f window identically to zero,
    # so it can only remove signal -- and it must actually remove some.
    assert torch.all(unmasked >= masked - 1e-12)
    assert float(unmasked.sum()) > float(masked.sum())
    assert float(masked.min()) >= 0.0


@pytest.fixture
def one_torch_thread():
    """Pin one intra-op thread for a batched-vs-per-matrix GEMM comparison.

    The bit-identity of ``torch.bmm`` and a per-matrix ``@`` at the same shapes and
    strides is a cuBLAS property. MKL in ``MKL_CBWR=COMPATIBLE`` mode (set by the
    CI workflows) splits the two differently at 3 and 5 threads (1, 2, 4, 10
    and 20 agree; ~1e-13 on a 2e3 image), which is what ``pytest -n 4`` gives
    each worker on a 20-core machine.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.usefixtures("one_torch_thread")
def test_atom_batched_probe_order_is_bit_identical_to_the_per_atom_loop(monkeypatch):
    """The slice-batched "probe" body must reproduce the per-atom launch train."""
    tp, T, pq, sites, scan, common = _image_case()
    batched = prism_eels_image(pq, T, tp, sites, scan, **common)
    # Denying the crop stacks forces the per-atom spelling (and with it a
    # per-atom coefficient gather and contraction).
    monkeypatch.setattr(pei, "_EELS_CROP_BLOCK_BYTES", 0)
    per_atom = prism_eels_image(pq, T, tp, sites, scan, **common)
    assert torch.equal(batched, per_atom)


def test_window_order_reproduces_the_probe_order_with_a_crop(monkeypatch):
    tp, T, pq, sites, scan, common = _image_case()
    monkeypatch.setattr(pei, "_EELS_CONTRACT_ORDER", "probe")
    probe = prism_eels_image(pq, T, tp, sites, scan, inelastic_crop=2, **common)
    monkeypatch.setattr(pei, "_EELS_CONTRACT_ORDER", "window")
    window = prism_eels_image(pq, T, tp, sites, scan, inelastic_crop=2, **common)
    torch.testing.assert_close(window, probe, rtol=1e-9, atol=1e-14)


def test_image_with_no_ionizable_atom_in_a_slice_still_walks_the_stack():
    gpts = (16, 16)
    tp = _tp(gpts, 1, seed=7)
    T = _transmissions(3, gpts, seed=8)
    pq = _aperture(gpts, 0.2)
    scan = torch.tensor([[4.0, 4.0], [8.0, 8.0]], dtype=torch.float64)
    common = dict(
        wavelength=_WAVELENGTH,
        gridsize=(8.0, 8.0),
        slice_distance=1.0,
        detector_mrad=40.0,
    )
    out = prism_eels_image(
        pq, T, tp, np.array([[0.5, 0.5, 0.9]]), scan, prism_mask=False, **common
    )
    assert out.shape == (2,)
    assert float(out.sum()) > 0.0
    assert torch.isfinite(out).all()


def test_fold_detectors_without_q_bins_is_the_plain_sum():
    """The documented identity: no q-binning means the scalar map's plain sum."""
    from scatterem.simulation.eels.forward.prism_eels_image import _fold_detectors

    power = torch.rand((3, 5, 7), dtype=torch.float64)
    folded = _fold_detectors(power, None, 0)
    assert torch.equal(folded, power.sum(dim=-1))


def test_fold_detectors_with_q_bins_conserves_the_total():
    """Binning regroups the additions, so the total is preserved.

    Not bit-for-bit: ``index_add_`` accumulates per bin and then the bins are
    summed, which is a different association than one flat reduction, so the
    two totals agree to float64 rounding rather than exactly.
    """
    from scatterem.simulation.eels.forward.prism_eels_image import _fold_detectors

    power = torch.rand((2, 6), dtype=torch.float64)
    q_bin = torch.tensor([0, 0, 1, 1, 2, 2])
    folded = _fold_detectors(power, q_bin, 3)
    assert folded.shape == (2, 3)
    assert torch.allclose(
        folded.sum(dim=-1), _fold_detectors(power, None, 0), rtol=0, atol=1e-14
    )


def test_cufft_batch_invariance_is_computed_once_then_cached():
    """The probe is expensive, so its verdict is memoised per (device, shape, block).

    On CPU the FFT is batch-invariant, so the verdict is True; the point of the
    test is the cache-miss body and that a second call reuses the stored answer
    rather than re-running four transforms.
    """
    from scatterem.simulation._wave import exit_walk as me

    flat = torch.rand((6, 8, 8), dtype=torch.complex128)
    me._CUFFT_BATCH_EXACT.clear()
    try:
        verdict = me._cufft_batch_bit_invariant(flat, 4)
        assert verdict is True
        assert len(me._CUFFT_BATCH_EXACT) == 1

        # second call: same key, served from the cache
        again = me._cufft_batch_bit_invariant(flat, 4)
        assert again is verdict
        assert len(me._CUFFT_BATCH_EXACT) == 1
    finally:
        me._CUFFT_BATCH_EXACT.clear()
