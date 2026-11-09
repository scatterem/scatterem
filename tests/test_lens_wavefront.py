"""Tests of the lens wavefront and its slope (brief TESTS.md, part B).

The main oracle evaluates the contract formula in NumPy float64 with the
radius ``theta = lambda |k|`` and the azimuth ``phi = atan2(k_y, k_x)``. The
test keeps its own table of the meaning of each label.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from scatterem.physics import optics as O

LAM = 0.025
# Meaning of each label, written in the test: (order n, multiplicity m, trig).
LABEL_MEANING = {
    "C10": (1, 0, np.cos),
    "C12a": (1, 2, np.cos),
    "C12b": (1, 2, np.sin),
    "C21a": (2, 1, np.cos),
    "C21b": (2, 1, np.sin),
    "C23a": (2, 3, np.cos),
    "C23b": (2, 3, np.sin),
    "C30": (3, 0, np.cos),
    "C32a": (3, 2, np.cos),
    "C32b": (3, 2, np.sin),
    "C34a": (3, 4, np.cos),
    "C34b": (3, 4, np.sin),
}
MEANINGS = [LABEL_MEANING[label] for label in O.LENS_LAYOUT]


def term(j: int, fy, fx, lam: float, mirror: bool = False) -> np.ndarray:
    """Return the oracle basis term of slot ``j`` (unit coefficient) [rad]."""
    n, m, trig = MEANINGS[j]
    fy, fx = np.asarray(fy, float), np.asarray(fx, float)
    theta = lam * np.hypot(fy, fx)
    phi = np.arctan2(fy, fx) * (-1.0 if mirror else 1.0)
    return 2 * np.pi / lam * theta ** (n + 1) / (n + 1) * trig(m * phi)


def oracle(c, fy, fx, lam: float = LAM, mirror: bool = False) -> np.ndarray:
    c = np.asarray(c, float)
    return sum(c[j] * term(j, fy, fx, lam, mirror) for j in range(12))


def term_sum(c, fy, fx, lam: float = LAM) -> np.ndarray:
    """Return ``sum_j |c_j term_j|`` [rad], the size scale of rounding errors."""
    c = np.asarray(c, float)
    return sum(abs(c[j] * term(j, fy, fx, lam)) for j in range(12))


def random_lens(rng: np.random.Generator, size=()) -> np.ndarray:
    """Return lens vectors whose terms have similar size near 20 mrad."""
    scale = np.array([0.02 ** (1 - n) * 100.0 for n, _, _ in MEANINGS])
    return rng.standard_normal((*size, 12)) * scale


def random_points(rng: np.random.Generator, n: int = 200):
    fy = torch.tensor(rng.uniform(-1.2, 1.2, n), dtype=torch.float64)
    fx = torch.tensor(rng.uniform(-1.2, 1.2, n), dtype=torch.float64)
    return fy, fx


def wf(c, fy, fx, lam: float = LAM) -> np.ndarray:
    c = torch.as_tensor(np.asarray(c), dtype=torch.float64)
    return O.lens_wavefront(c, fy, fx, lambda_a=lam).numpy()


def max_rel(got: np.ndarray, ref: np.ndarray) -> float:
    return float(np.max(np.abs(got - ref)) / np.max(np.abs(ref)))


# --- B.1 radius-azimuth oracle ----------------------------------------------


def test_wavefront_matches_radius_azimuth_oracle() -> None:
    rng = np.random.default_rng(1)
    fy, fx = random_points(rng, 500)
    for _ in range(5):
        c = random_lens(rng)
        assert max_rel(wf(c, fy, fx), oracle(c, fy, fx)) <= 1e-12


def test_oracle_check_detects_mirrored_azimuth() -> None:
    rng = np.random.default_rng(2)
    fy, fx = random_points(rng)
    c = np.zeros(12)
    c[O.LENS_LAYOUT.index("C12b")] = 50.0
    assert max_rel(wf(c, fy, fx), oracle(c, fy, fx, mirror=True)) > 1e-12


def test_wavefront_is_exactly_zero_at_origin() -> None:
    c = torch.tensor(random_lens(np.random.default_rng(3)))
    zero = torch.zeros(1, dtype=torch.float64)
    assert O.lens_wavefront(c, zero, zero, lambda_a=LAM).item() == 0.0


# --- B.2 basis terms --------------------------------------------------------


def test_layout_and_terms_match_label_table() -> None:
    assert set(O.LENS_LAYOUT) == set(LABEL_MEANING)
    for (n, m, part), label in zip(O.LENS_TERMS, O.LENS_LAYOUT, strict=True):
        n_ref, m_ref, trig = LABEL_MEANING[label]
        assert (n, m) == (n_ref, m_ref)
        assert (part == "c") == (trig is np.cos)


@pytest.mark.parametrize("j", range(12))
def test_unit_vector_gives_one_basis_term(j: int) -> None:
    fy, fx = random_points(np.random.default_rng(4))
    got = wf(np.eye(12)[j], fy, fx)
    assert max_rel(got, term(j, fy, fx, LAM)) <= 1e-13
    for k in range(12):
        if k != j:
            assert max_rel(got, term(k, fy, fx, LAM)) > 1e-3


def test_basis_check_detects_neighbour_slot() -> None:
    fy, fx = random_points(np.random.default_rng(5))
    assert max_rel(wf(np.eye(12)[1], fy, fx), term(2, fy, fx, LAM)) > 1e-3


# --- B.3 rotation -----------------------------------------------------------


def rotate(fy, fx, angle: float):
    c, s = math.cos(angle), math.sin(angle)
    return s * fx + c * fy, c * fx - s * fy


@pytest.mark.parametrize("j", range(12))
def test_term_invariant_under_m_fold_turn(j: int) -> None:
    fy, fx = random_points(np.random.default_rng(6))
    m = MEANINGS[j][1]
    base = wf(np.eye(12)[j], fy, fx)
    if m == 0:
        assert max_rel(wf(np.eye(12)[j], *rotate(fy, fx, 0.731)), base) < 1e-12
        return
    full = wf(np.eye(12)[j], *rotate(fy, fx, 2 * math.pi / m))
    half = wf(np.eye(12)[j], *rotate(fy, fx, math.pi / m))
    assert max_rel(full, base) < 1e-12
    assert max_rel(half, -base) < 1e-12


def test_rotation_check_detects_sign_flip() -> None:
    fy, fx = random_points(np.random.default_rng(7))
    base = wf(np.eye(12)[1], fy, fx)
    half = wf(np.eye(12)[1], *rotate(fy, fx, math.pi / 2))
    assert max_rel(half, base) > 1e-12


# --- B.4, B.5, B.6 scaling and linearity ------------------------------------


@pytest.mark.parametrize("j", range(12))
def test_radial_scaling(j: int) -> None:
    fy, fx = random_points(np.random.default_rng(8))
    n, t = MEANINGS[j][0], 1.7
    scaled = wf(np.eye(12)[j], t * fy, t * fx)
    assert max_rel(scaled, t ** (n + 1) * wf(np.eye(12)[j], fy, fx)) < 1e-12
    assert max_rel(scaled, t**n * wf(np.eye(12)[j], fy, fx)) > 1e-3  # control


def test_wavelength_scaling() -> None:
    rng = np.random.default_rng(9)
    fy, fx = random_points(rng)
    c, t = random_lens(rng), 1.9
    scaled = wf(c, t * fy, t * fx, lam=LAM / t)
    assert max_rel(scaled, t * wf(c, fy, fx)) < 1e-12
    assert max_rel(scaled, wf(c, fy, fx)) > 1e-3  # control


def test_linearity() -> None:
    rng = np.random.default_rng(10)
    fy, fx = random_points(rng)
    a, b = random_lens(rng), random_lens(rng)
    assert max_rel(wf(a + b, fy, fx), wf(a, fy, fx) + wf(b, fy, fx)) < 1e-12
    assert max_rel(wf(a + b, fy, fx), wf(a, fy, fx)) > 1e-3  # control


# --- B.7 batch --------------------------------------------------------------


@pytest.mark.parametrize("batch", [(3,), (2, 3)])
@pytest.mark.parametrize("grid", [((5, 7), (5, 7)), ((5, 1), (1, 7))])
def test_batch_equals_separate_calls(batch, grid) -> None:
    rng = np.random.default_rng(11)
    lens = torch.tensor(random_lens(rng, batch))
    fy = torch.tensor(rng.uniform(-1, 1, grid[0]))
    fx = torch.tensor(rng.uniform(-1, 1, grid[1]))
    for func, tail in ((O.lens_wavefront, ()), (O.lens_wavefront_slope, (2,))):
        out = func(lens, fy, fx, lambda_a=LAM)
        assert out.shape == (*batch, 5, 7, *tail)
        for idx in np.ndindex(*batch):
            single = func(lens[idx], fy, fx, lambda_a=LAM)
            torch.testing.assert_close(out[idx], single, rtol=1e-14, atol=0)


def test_sequence_lens_is_accepted() -> None:
    fy, fx = random_points(np.random.default_rng(12), 10)
    c = random_lens(np.random.default_rng(13))
    out = O.lens_wavefront(c.tolist(), fy, fx, lambda_a=LAM)
    torch.testing.assert_close(
        out, O.lens_wavefront(torch.tensor(c), fy, fx, lambda_a=LAM)
    )


@pytest.mark.parametrize("shape", [(11,), (3, 13), ()])
def test_bad_lens_shape_is_refused(shape) -> None:
    fy = torch.zeros(4, dtype=torch.float64)
    for func in (O.lens_wavefront, O.lens_wavefront_slope):
        with pytest.raises(ValueError, match="lens"):
            func(torch.zeros(shape, dtype=torch.float64), fy, fy, lambda_a=LAM)


def test_bad_inputs_are_refused() -> None:
    f = torch.zeros(4, dtype=torch.float64)
    lens = torch.zeros(12, dtype=torch.float64)
    with pytest.raises(ValueError, match="broadcast"):
        O.lens_wavefront(lens, f, torch.zeros(3, dtype=torch.float64), lambda_a=LAM)
    with pytest.raises(TypeError, match="fx"):
        O.lens_wavefront(lens, f, np.zeros(4), lambda_a=LAM)
    with pytest.raises(TypeError, match="fy"):
        O.lens_wavefront(lens, f.to(torch.complex128), f, lambda_a=LAM)
    with pytest.raises(TypeError, match="lens"):
        O.lens_wavefront(torch.zeros(12, dtype=torch.int64), f, f, lambda_a=LAM)
    for bad, err in (
        (0.0, ValueError),
        (-1.0, ValueError),
        (math.nan, ValueError),
        (True, TypeError),
        (torch.tensor(LAM), TypeError),
    ):
        with pytest.raises(err, match="lambda_a"):
            O.lens_wavefront(lens, f, f, lambda_a=bad)


# --- B.8 autograd -----------------------------------------------------------


def grid_with_origin() -> tuple[torch.Tensor, torch.Tensor]:
    f = torch.tensor([-0.7, 0.0, 0.4], dtype=torch.float64)
    fy, fx = torch.meshgrid(
        f, torch.tensor([0.0, 0.9], dtype=torch.float64), indexing="ij"
    )
    return fy.clone().requires_grad_(), fx.clone().requires_grad_()


def test_gradcheck_wavefront_and_slope() -> None:
    fy, fx = grid_with_origin()
    lens = torch.tensor(
        random_lens(np.random.default_rng(14), (2,)), requires_grad=True
    )
    for func in (O.lens_wavefront, O.lens_wavefront_slope):
        assert torch.autograd.gradcheck(
            lambda c, y, x, f=func: f(c, y, x, lambda_a=LAM), (lens, fy, fx)
        )


def test_zero_lens_gradients_are_basis_terms() -> None:
    fy, fx = grid_with_origin()
    lens = torch.zeros(12, dtype=torch.float64, requires_grad=True)
    O.lens_wavefront(lens, fy, fx, lambda_a=LAM).sum().backward()
    ref = [term(j, fy.detach(), fx.detach(), LAM).sum() for j in range(12)]
    np.testing.assert_allclose(lens.grad.numpy(), ref, rtol=1e-13, atol=0)
    for g in (fy.grad, fx.grad):
        assert torch.isfinite(g).all() and (g == 0).all()


def test_gradient_check_detects_wrong_basis() -> None:
    fy, fx = grid_with_origin()
    lens = torch.zeros(12, dtype=torch.float64, requires_grad=True)
    O.lens_wavefront(lens, fy, fx, lambda_a=LAM).sum().backward()
    wrong = [term((j + 1) % 12, fy.detach(), fx.detach(), LAM).sum() for j in range(12)]
    assert not np.allclose(lens.grad.numpy(), wrong, rtol=1e-6)


# --- B.9 slope --------------------------------------------------------------


def test_slope_matches_autograd() -> None:
    rng = np.random.default_rng(15)
    fy, fx = (t.requires_grad_() for t in random_points(rng, 300))
    lens = torch.tensor(random_lens(rng))
    O.lens_wavefront(lens, fy, fx, lambda_a=LAM).sum().backward()
    auto = torch.stack((fy.grad, fx.grad), dim=-1)
    slope = O.lens_wavefront_slope(lens, fy.detach(), fx.detach(), lambda_a=LAM)
    scale = torch.linalg.vector_norm(auto, dim=-1).max()
    assert ((slope - auto).abs().max() / scale).item() <= 1e-12
    swapped = slope.flip(-1)  # control: swap the two components
    assert ((swapped - auto).abs().max() / scale).item() > 1e-3


def test_slope_matches_converging_finite_differences() -> None:
    rng = np.random.default_rng(16)
    fy, fx = random_points(rng, 50)
    c = random_lens(rng)
    slope = O.lens_wavefront_slope(torch.tensor(c), fy, fx, lambda_a=LAM).numpy()
    scale = np.max(np.abs(slope))
    for axis in (0, 1):
        derivs, errors = [], []
        for h in (2e-2, 1e-2, 5e-3):
            dy, dx = (h, 0.0) if axis == 0 else (0.0, h)
            d = (wf(c, fy + dy, fx + dx) - wf(c, fy - dy, fx - dx)) / (2 * h)
            derivs.append(d)
            errors.append(np.max(np.abs(d - slope[..., axis])))
        assert 3.5 < errors[0] / errors[1] < 4.5
        assert 3.5 < errors[1] / errors[2] < 4.5
        richardson = (4 * derivs[2] - derivs[1]) / 3
        assert np.max(np.abs(richardson - slope[..., axis])) / scale < 1e-8


def test_slope_allows_second_backward_pass() -> None:
    fy, fx = grid_with_origin()
    lens = torch.tensor(random_lens(np.random.default_rng(17)), requires_grad=True)
    assert torch.autograd.gradgradcheck(
        lambda c, y, x: O.lens_wavefront_slope(c, y, x, lambda_a=LAM), (lens, fy, fx)
    )


# --- B.10 beam shift --------------------------------------------------------


def test_defocus_beam_shift() -> None:
    fy, fx = random_points(np.random.default_rng(18), 40)
    a10 = 123.4
    lens = O.pack_lens({"C10": a10})
    shift = O.lens_wavefront_slope(lens, fy, fx, lambda_a=LAM) / (2 * math.pi)
    expected = a10 * LAM * torch.stack((fy, fx), dim=-1)
    torch.testing.assert_close(shift, expected, rtol=1e-13, atol=0)
    wrong = a10 * torch.stack((fy, fx), dim=-1)  # control: no wavelength
    assert not torch.allclose(shift, wrong, rtol=1e-3)


# --- B.11 magnitude-angle form ----------------------------------------------


def amplitude_oracle(amp: np.ndarray, fy, fx, lam: float = LAM, shift=0.0):
    """Evaluate ``W`` with ``M cos(m (phi - alpha))`` from the amplitude form."""
    theta = lam * np.hypot(fy, fx)
    phi = np.arctan2(fy, fx)
    out = 0.0
    index = {label: j for j, label in enumerate(O.AMPLITUDE_LAYOUT)}
    for label, j in index.items():
        if not label.startswith("C"):
            continue
        n, m, _ = LABEL_MEANING[label if label in LABEL_MEANING else label + "a"]
        if m == 0:
            out = out + amp[j] * theta ** (n + 1) / (n + 1)
        else:
            alpha = amp[index[f"phi{n}{m}"]] + shift * math.pi / m
            out = out + amp[j] * theta ** (n + 1) / (n + 1) * np.cos(m * (phi - alpha))
    return 2 * np.pi / lam * out


def test_amplitude_form_round_trip_and_range() -> None:
    rng = np.random.default_rng(19)
    lens = torch.tensor(random_lens(rng, (20,)))
    lens[0, 1:3] = 0.0  # a zero pair gives alpha = 0
    amp = O.lens_to_amplitude(lens)
    back = O.lens_from_amplitude(amp)
    torch.testing.assert_close(back, lens, rtol=1e-12, atol=1e-9)
    for n, m, part in O.LENS_TERMS:
        if m > 0 and part == "c":
            mag = amp[..., O.AMPLITUDE_LAYOUT.index(f"C{n}{m}")]
            alpha = amp[..., O.AMPLITUDE_LAYOUT.index(f"phi{n}{m}")]
            assert (mag >= 0).all()
            assert (alpha >= 0).all()
            assert (alpha < 2 * math.pi / m).all()
    assert amp[0, O.AMPLITUDE_LAYOUT.index("phi12")].item() == 0.0


def test_amplitude_layout_is_magnitudes_then_angles() -> None:
    assert O.AMPLITUDE_LAYOUT == (
        "C10", "C12", "C21", "C23", "C30", "C32", "C34",
        "phi12", "phi21", "phi23", "phi32", "phi34",
    )  # fmt: skip
    lens = torch.zeros(12, dtype=torch.float64)
    lens[O.LENS_LAYOUT.index("C21a")] = 3.0
    lens[O.LENS_LAYOUT.index("C21b")] = 4.0
    amp = O.unpack_lens(O.lens_to_amplitude(lens), layout=O.AMPLITUDE_LAYOUT)
    assert amp["C21"] == pytest.approx(5.0)
    assert amp["phi21"] == pytest.approx(math.atan2(4.0, 3.0))
    assert all(v == 0.0 for k, v in amp.items() if k not in ("C21", "phi21"))


def test_amplitude_form_gives_same_wavefront() -> None:
    rng = np.random.default_rng(20)
    fy, fx = random_points(rng)
    c = random_lens(rng)
    amp = O.lens_to_amplitude(torch.tensor(c)).numpy()
    ref = wf(c, fy, fx)
    assert max_rel(amplitude_oracle(amp, fy.numpy(), fx.numpy()), ref) <= 1e-12
    shifted = amplitude_oracle(amp, fy.numpy(), fx.numpy(), shift=1.0)  # control
    assert max_rel(shifted, ref) > 1e-3


# --- B.12 labelled form -----------------------------------------------------


@pytest.mark.parametrize("layout", [O.LENS_LAYOUT, O.AMPLITUDE_LAYOUT])
def test_labelled_form_round_trip(layout) -> None:
    v = torch.tensor(random_lens(np.random.default_rng(21)))
    labelled = O.unpack_lens(v, layout=layout)
    assert list(labelled) == list(layout)
    assert torch.equal(O.pack_lens(labelled, layout=layout), v)
    part = O.pack_lens({layout[3]: 2.5}, layout=layout)
    assert part[3].item() == 2.5 and (part[[i for i in range(12) if i != 3]] == 0).all()
    with pytest.raises(ValueError, match="unknown label"):
        O.pack_lens({"defocus": 1.0}, layout=layout)


def test_labelled_form_refusals() -> None:
    with pytest.raises(ValueError, match="layout"):
        O.pack_lens({}, layout=("a",) * 12)
    with pytest.raises(TypeError):
        O.pack_lens({"C10": True})
    with pytest.raises(ValueError):
        O.pack_lens({"C10": math.inf})
    with pytest.raises(ValueError, match="shape"):
        O.unpack_lens([0.0] * 11)


# --- B.13 precision and devices ---------------------------------------------


def test_float32_agrees_with_float64() -> None:
    rng = np.random.default_rng(22)
    fy, fx = (t.float() for t in random_points(rng))
    c = torch.tensor(random_lens(rng)).float()
    w32 = O.lens_wavefront(c, fy, fx, lambda_a=LAM)
    assert w32.dtype == torch.float32
    w64 = O.lens_wavefront(c.double(), fy.double(), fx.double(), lambda_a=LAM)
    bound = 32 * 2.0**-23 * term_sum(c.double().numpy(), fy.double(), fx.double())
    assert np.all(np.abs(w32.double().numpy() - w64.numpy()) <= bound)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_agrees_with_cpu(require_cuda, dtype) -> None:
    rng = np.random.default_rng(23)
    fy, fx = (t.to(dtype) for t in random_points(rng))
    c = torch.tensor(random_lens(rng, (3,))).to(dtype)
    eps = torch.finfo(dtype).eps
    for func in (O.lens_wavefront, O.lens_wavefront_slope):
        cpu = func(c, fy, fx, lambda_a=LAM)
        gpu = func(c.cuda(), fy.cuda(), fx.cuda(), lambda_a=LAM).cpu()
        scale = (
            cpu.abs().amax()
            if func is O.lens_wavefront_slope
            else torch.tensor(
                [
                    term_sum(ci.double().numpy(), fy.double(), fx.double()).max()
                    for ci in c
                ]
            ).max()
        )
        assert ((gpu - cpu).abs().max() <= 64 * eps * scale).item()


def test_cuda_call_does_not_sync(require_cuda) -> None:
    dev = require_cuda
    fy = torch.rand(64, 64, device=dev, requires_grad=True)
    fx = torch.rand(64, 64, device=dev, requires_grad=True)
    lens = torch.rand(4, 12, device=dev, requires_grad=True)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for func in (O.lens_wavefront, O.lens_wavefront_slope):
            func(lens, fy, fx, lambda_a=LAM).sum().backward()
        with pytest.raises(RuntimeError):  # control: a host read is detected
            O.lens_wavefront(lens, fy, fx, lambda_a=LAM).sum().item()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
