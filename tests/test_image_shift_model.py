"""The bright-field shift model, ``transfer.aberrations_to_image_shifts``.

This is the model the bright-field aberration fit inverts: for a given
wavefront it predicts where each detector pixel's bright-field image sits.

It used to be computed as a finite difference (``torch.gradient``) of ``chi``
followed by an ``affine_grid``/``grid_sample`` rotation. That had two costs. The
resample was inexact, worst at the array edge. And ``grid_sample``'s CUDA
backward accumulates with atomics, so it has no deterministic implementation --
which made ``determine_aberrations(correction_method="bright-field-shifts")``
return a different answer on every call for the same input, by thousands of
Angstrom in the weakly constrained coefficients.

It is now evaluated in closed form at rotated coordinates
(:func:`scatterem.utils.aberration_basis.cartesian_chi_gradient`), so there is
no interpolation and no atomic accumulation. These tests pin both properties.
"""

import numpy as np
import pytest
import torch

from scatterem.utils.aberration_basis import COEFFICIENT_NAMES
from scatterem.utils.transfer import aberrations_to_image_shifts

SHAPE = (48, 48)
SAMPLING = (0.1, 0.1)
WAVELENGTH = 0.0197


def _coefficients(**named):
    array = torch.zeros(len(COEFFICIENT_NAMES), dtype=torch.float64)
    for name, value in named.items():
        array[COEFFICIENT_NAMES.index(name)] = value
    return array


def _shifts(coefficients, rotation=0.0, device="cpu"):
    coefficients = coefficients.to(device)
    return aberrations_to_image_shifts(
        coefficients,
        torch.tensor(rotation, dtype=coefficients.dtype, device=device),
        np.asarray(SAMPLING),
        WAVELENGTH,
        SHAPE,
    )


# --- determinism: the defect this model was rewritten to remove --------------


def test_repeated_calls_are_bitwise_identical():
    coefficients = _coefficients(C10=-80.0, C12a=12.0, C30=5000.0)
    first = _shifts(coefficients)
    for _ in range(4):
        assert torch.equal(_shifts(coefficients), first)


@pytest.mark.parametrize("rotation", [0.0, 84.0, 180.0])
def test_runs_under_deterministic_algorithms(rotation):
    """``grid_sample``'s CUDA backward would raise here; closed form does not."""
    coefficients = _coefficients(C10=-80.0, C12a=12.0, C30=5000.0)
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        shifts = _shifts(coefficients, rotation=rotation)
    finally:
        torch.use_deterministic_algorithms(previous)
    assert torch.isfinite(shifts).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_repeated_calls_are_bitwise_identical():
    """The non-determinism was CUDA-only, so CPU agreement alone proves nothing."""
    coefficients = _coefficients(C10=-80.0, C12a=12.0, C30=5000.0)
    first = _shifts(coefficients, rotation=84.0, device="cuda")
    for _ in range(4):
        assert torch.equal(_shifts(coefficients, rotation=84.0, device="cuda"), first)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_gradient_is_deterministic():
    """The fit differentiates through this model; that backward must be stable too."""
    coefficients = _coefficients(C10=-80.0, C12a=12.0, C30=5000.0).cuda()
    grads = []
    for _ in range(3):
        leaf = coefficients.clone().requires_grad_(True)
        shifts = aberrations_to_image_shifts(
            leaf,
            torch.tensor(84.0, dtype=torch.float64, device="cuda"),
            np.asarray(SAMPLING),
            WAVELENGTH,
            SHAPE,
        )
        shifts.square().sum().backward()
        grads.append(leaf.grad.clone())
    assert torch.equal(grads[0], grads[1])
    assert torch.equal(grads[0], grads[2])


# --- physics the shift field must obey --------------------------------------


def test_zero_aberrations_give_zero_shift():
    assert torch.all(_shifts(_coefficients()) == 0)


def test_defocus_shift_is_linear_in_q():
    """Defocus tilts the wavefront linearly, so the shift field is a pure ramp.

    An interpolated model cannot satisfy this exactly; a closed-form one does.
    """
    shifts = _shifts(_coefficients(C10=-100.0))
    doubled = shifts[SHAPE[0] // 2 :: 2, SHAPE[1] // 2 :: 2]
    single = shifts[SHAPE[0] // 2 :, SHAPE[1] // 2 :][
        : doubled.shape[0], : doubled.shape[1]
    ]
    assert torch.allclose(doubled, 2.0 * single, rtol=1e-10, atol=1e-14)


def test_defocus_sign_reverses_with_defocus():
    positive = _shifts(_coefficients(C10=100.0))
    negative = _shifts(_coefficients(C10=-100.0))
    assert torch.allclose(positive, -negative, rtol=1e-12, atol=0)


def test_shift_is_linear_in_the_coefficients():
    """chi is linear in the coefficients and so is its gradient."""
    a = _coefficients(C10=-60.0)
    b = _coefficients(C34b=900.0)
    assert torch.allclose(
        _shifts(a + b), _shifts(a) + _shifts(b), rtol=1e-11, atol=1e-15
    )


@pytest.mark.parametrize(
    "name,parity",
    [
        # chi ~ r^2 and r^4: the gradient is odd in q, so q -> -q flips the shift
        ("C10", -1),
        ("C12a", -1),
        ("C12b", -1),
        ("C30", -1),
        ("C32a", -1),
        ("C34a", -1),
        # chi ~ r^3: the gradient is even in q, so the shift is UNCHANGED
        ("C21a", +1),
        ("C21b", +1),
        ("C23a", +1),
        ("C23b", +1),
    ],
)
def test_rotation_by_180_degrees_follows_the_gradient_parity(name, parity):
    """At 180 degrees the model evaluates chi's gradient at -q, so parity decides.

    The even-order terms (r^2, r^4) have an odd gradient and flip sign; the
    odd-order terms (r^3, coma and three-fold astigmatism) have an even gradient
    and are left untouched. Asserting a blanket sign flip would be wrong, and
    lumping the two families together is how a parity error hides. Exact here,
    because no interpolation is involved -- the previous resampling model was
    only approximate even at 180 degrees, where nothing needs interpolating.
    """
    coefficients = _coefficients(**{name: 500.0})
    unrotated = _shifts(coefficients, rotation=0.0)
    rotated = _shifts(coefficients, rotation=180.0)
    scale = float(unrotated.abs().max())
    assert torch.allclose(rotated, parity * unrotated, rtol=0, atol=1e-12 * scale)


def test_rotation_is_periodic_in_360_degrees():
    coefficients = _coefficients(C10=-80.0, C12a=12.0, C23a=300.0)
    scale = float(_shifts(coefficients, rotation=37.0).abs().max())
    assert torch.allclose(
        _shifts(coefficients, rotation=37.0),
        _shifts(coefficients, rotation=397.0),
        rtol=0,
        atol=1e-11 * scale,
    )


def test_rotating_an_isotropic_wavefront_rotates_its_shift_field():
    """For a rotationally symmetric wavefront the field is radial at any rotation.

    So rotation must leave the *magnitude* untouched -- a property that catches a
    rotation applied in the wrong coordinate system.
    """
    coefficients = _coefficients(C10=-80.0, C30=5000.0)
    reference = _shifts(coefficients, rotation=0.0).square().sum(-1).sqrt()
    for rotation in (13.0, 84.0, 251.0):
        magnitude = _shifts(coefficients, rotation=rotation).square().sum(-1).sqrt()
        scale = float(reference.max())
        assert torch.allclose(
            magnitude, reference, rtol=0, atol=1e-12 * scale
        ), rotation


def test_shape_and_units():
    shifts = _shifts(_coefficients(C10=-80.0))
    assert shifts.shape == (SHAPE[0], SHAPE[1], 2)
    # The bright-field shift for 80 A of defocus at 20 mrad is of order
    # defocus * angle = 80 * 0.02 = 1.6 A; assert the right order of magnitude
    # rather than a number, which is what a units error would break.
    assert 0.05 < float(shifts.abs().max()) < 500.0


def test_precision_follows_the_coefficient_dtype():
    """A float64 fit must not be truncated to float32 by its own forward model."""
    coefficients = _coefficients(C10=-80.0, C12a=12.0, C30=5000.0)
    double = _shifts(coefficients, rotation=84.0)
    single = _shifts(coefficients.float(), rotation=84.0)
    assert double.dtype == torch.float64
    assert single.dtype == torch.float32
    scale = float(double.abs().max())
    assert torch.allclose(single.double(), double, rtol=0, atol=1e-6 * scale)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cpu_and_cuda_agree():
    coefficients = _coefficients(C10=-80.0, C12a=12.0, C30=5000.0)
    cpu = _shifts(coefficients, rotation=84.0, device="cpu")
    cuda = _shifts(coefficients, rotation=84.0, device="cuda").cpu()
    scale = float(cpu.abs().max())
    assert torch.allclose(cuda, cpu, rtol=0, atol=1e-12 * scale)


def test_gradient_survives_an_all_zero_wavefront():
    """A fit starts from zeros; the model must still be differentiable there.

    The closed-form model skips zero-valued terms in the forward pass. If it did
    so under autograd too, an all-zero wavefront would detach the result from
    both the coefficients and the rotation, and the fit would fail outright with
    "element 0 of tensors does not require grad".
    """
    coefficients = torch.zeros(len(COEFFICIENT_NAMES), dtype=torch.float64)
    coefficients.requires_grad_(True)
    rotation = torch.tensor(37.0, dtype=torch.float64, requires_grad=True)

    shifts = aberrations_to_image_shifts(
        coefficients, rotation, np.asarray(SAMPLING), WAVELENGTH, SHAPE
    )
    weights = torch.linspace(0.1, 2.0, shifts.numel(), dtype=torch.float64).reshape(
        shifts.shape
    )
    (shifts * weights).sum().backward()

    assert coefficients.grad is not None
    assert float(coefficients.grad.abs().max()) > 0.0
    # The shift field is identically zero for any rotation when the wavefront is
    # zero, so the rotation derivative is legitimately zero -- but it must exist.
    assert rotation.grad is not None
