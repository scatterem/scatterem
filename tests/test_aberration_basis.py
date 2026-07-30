"""Clean-room aberration basis and polar/cartesian conventions.

The basis is the standard wave-aberration expansion,

    chi(q) = (2*pi/lambda) * sum_n (1/n) * r^n * [Ca * cos(m*phi) + Cb * sin(m*phi)]

with ``r = lambda*|q|`` and ``phi = atan2(lambda*qy, lambda*qx)``. Tests assert
the angular symmetry each coefficient is *defined* to have, plus numerical
plus the properties a (magnitude, angle) pair must have to mean anything at
all -- a non-negative magnitude, and the same wavefront as the cartesian pair
it came from.
"""

import numpy as np
import pytest
import torch

from scatterem.utils.aberration_basis import (
    COEFFICIENT_NAMES,
    cartesian_chi,
    cartesian_chi_gradient,
    cartesian_to_polar,
    polar_to_cartesian,
)


def _grid(n=32, extent=0.02):
    q = torch.linspace(-extent, extent, n, dtype=torch.float64)
    qy, qx = torch.meshgrid(q, q, indexing="ij")
    return qy, qx


def _coeffs(**kw):
    c = torch.zeros(12, dtype=torch.float64)
    for name, value in kw.items():
        c[COEFFICIENT_NAMES.index(name)] = value
    return c


def _weights(shape):
    """A non-degenerate weight field for probing derivatives.

    A smooth ramp will not do: summed against a term like r^2*cos(2*phi) over a
    grid symmetric about the origin it cancels to ~1e-19, which reads as "the
    derivative is zero" and makes the test vacuous for every 2- and 4-fold term.
    """
    generator = torch.Generator().manual_seed(11)
    return torch.rand(shape, generator=generator, dtype=torch.float64) + 0.5


def test_coefficient_order_is_the_documented_twelve():
    assert COEFFICIENT_NAMES == (
        "C10",
        "C12a",
        "C12b",
        "C21a",
        "C21b",
        "C23a",
        "C23b",
        "C30",
        "C32a",
        "C32b",
        "C34a",
        "C34b",
    )


# --- angular symmetry: the defining property of each term -------------------


@pytest.mark.parametrize(
    "name,fold",
    [
        ("C10", 0),
        ("C30", 0),
        ("C12a", 2),
        ("C12b", 2),
        ("C32a", 2),
        ("C32b", 2),
        ("C21a", 1),
        ("C21b", 1),
        ("C23a", 3),
        ("C23b", 3),
        ("C34a", 4),
        ("C34b", 4),
    ],
)
def test_term_has_its_defining_rotational_symmetry(name, fold):
    """A `fold`-fold term must be invariant under rotation by 2*pi/fold.

    Evaluated on a ring so the rotation is exact rather than interpolated.
    `fold == 0` means rotationally invariant at any angle.
    """
    lam = 0.025
    radius = 0.01
    n_phi = 720
    phi = torch.linspace(0, 2 * np.pi, n_phi + 1, dtype=torch.float64)[:-1]
    qy, qx = radius * torch.sin(phi), radius * torch.cos(phi)
    chi = cartesian_chi(qy, qx, lam, _coeffs(**{name: 1.0}))

    turn = n_phi // (fold if fold else 8)
    rotated = torch.roll(chi, turn)
    assert torch.allclose(chi, rotated, atol=1e-9 * max(chi.abs().max(), 1e-30))


def test_a_two_fold_term_is_not_four_fold():
    """Guards a symmetry test that would pass for the wrong term."""
    lam = 0.025
    phi = torch.linspace(0, 2 * np.pi, 721, dtype=torch.float64)[:-1]
    qy, qx = 0.01 * torch.sin(phi), 0.01 * torch.cos(phi)
    chi = cartesian_chi(qy, qx, lam, _coeffs(C12a=1.0))
    assert not torch.allclose(chi, torch.roll(chi, 720 // 4), atol=1e-6)


def test_defocus_is_quadratic_in_angle():
    """C10 must scale as r^2, so doubling |q| quadruples chi."""
    lam = 0.025
    c = _coeffs(C10=100.0)
    near = cartesian_chi(torch.tensor([0.005]), torch.tensor([0.0]), lam, c)
    far = cartesian_chi(torch.tensor([0.010]), torch.tensor([0.0]), lam, c)
    assert float(far / near) == pytest.approx(4.0, rel=1e-9)


def test_spherical_aberration_is_quartic_in_angle():
    lam = 0.025
    c = _coeffs(C30=1000.0)
    near = cartesian_chi(torch.tensor([0.005]), torch.tensor([0.0]), lam, c)
    far = cartesian_chi(torch.tensor([0.010]), torch.tensor([0.0]), lam, c)
    assert float(far / near) == pytest.approx(16.0, rel=1e-9)


def test_terms_superpose_linearly():
    """chi is linear in the coefficients; a sum of terms is the sum of chis."""
    lam = 0.025
    qy, qx = _grid()
    a = cartesian_chi(qy, qx, lam, _coeffs(C10=50.0))
    b = cartesian_chi(qy, qx, lam, _coeffs(C34b=700.0))
    both = cartesian_chi(qy, qx, lam, _coeffs(C10=50.0, C34b=700.0))
    assert torch.allclose(both, a + b, atol=1e-10)


def test_zero_coefficients_give_zero_phase():
    qy, qx = _grid()
    assert torch.all(cartesian_chi(qy, qx, 0.025, torch.zeros(12)) == 0)




# --- polar / cartesian conventions ----------------------------------------


@pytest.mark.parametrize(
    "mag,angle,pair,fold",
    [
        ("C12", "phi12", ("C12a", "C12b"), 2),
        ("C21", "phi21", ("C21a", "C21b"), 1),
        ("C23", "phi23", ("C23a", "C23b"), 3),
        ("C32", "phi32", ("C32a", "C32b"), 2),
        ("C34", "phi34", ("C34a", "C34b"), 4),
    ],
)
def test_polar_to_cartesian_follows_the_basis_definition(mag, angle, pair, fold):
    """Ca = C*cos(m*phi_m) and Cb = C*sin(m*phi_m) -- by definition of the basis."""
    c = polar_to_cartesian({mag: 7.0, angle: 0.37})
    assert c[pair[0]] == pytest.approx(7.0 * np.cos(fold * 0.37))
    assert c[pair[1]] == pytest.approx(7.0 * np.sin(fold * 0.37))


@pytest.mark.parametrize(
    "polar",
    [
        {"C10": 120.0},
        {"C12": 15.0, "phi12": 0.3},
        {"C21": 80.0, "phi21": -1.1},
        {"C23": 60.0, "phi23": 0.8},
        {"C30": 5000.0},
        {"C32": 900.0, "phi32": 0.25},
        {"C34": 700.0, "phi34": -0.4},
    ],
)
def test_round_trips(polar):
    back = cartesian_to_polar(polar_to_cartesian(polar))
    for k, v in polar.items():
        assert back[k] == pytest.approx(v, abs=1e-9)


def test_magnitudes_are_never_negative():
    """A magnitude is a length, so the convention forbids a negative one."""
    for pair in (("C12a", "C12b"), ("C23a", "C23b"), ("C34a", "C34b")):
        for sa in (-3.0, 3.0):
            for sb in (-4.0, 4.0):
                polar = cartesian_to_polar({pair[0]: sa, pair[1]: sb})
                mag = polar[pair[0][:3]]
                assert mag == pytest.approx(5.0)
                assert mag >= 0.0


def test_reported_polar_pair_describes_the_same_wavefront():
    """What makes a reported (magnitude, angle) pair meaningful.

    The pair means something only if C*cos(m*(phi - phi_m)) reproduces
    Ca*cos(m*phi) + Cb*sin(m*phi). A convention that gets this wrong prints an
    astigmatism describing a different wavefront than the one that was fitted --
    off by 45 degrees and inverted.
    """
    ca, cb, fold = 8.2534, 5.6464, 2
    polar = cartesian_to_polar({"C12a": ca, "C12b": cb})
    phi = torch.linspace(0, 2 * np.pi, 37, dtype=torch.float64)
    expected = ca * torch.cos(fold * phi) + cb * torch.sin(fold * phi)
    got = polar["C12"] * torch.cos(fold * (phi - polar["phi12"]))
    assert torch.allclose(got, expected, atol=1e-9)


def test_missing_coefficients_are_treated_as_zero():
    assert cartesian_to_polar({})["C10"] == 0.0
    assert polar_to_cartesian({})["C34b"] == 0.0


def _random_coefficients(seed=0, scale=100.0):
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(len(COEFFICIENT_NAMES), generator=generator, dtype=torch.float64)
        * scale
    )


def test_gradient_agrees_with_autograd_of_chi():
    """The gradient is hand-differentiated, so check the algebra against autograd.

    This is the whole justification for the closed form: it replaced a
    ``torch.gradient`` finite difference followed by a ``grid_sample`` resample,
    which cost accuracy at the array edge and -- because ``grid_sample``'s CUDA
    backward uses atomics -- made the aberration fit non-reproducible.
    """
    qy, qx = _grid()
    wavelength = 0.0197
    coefficients = _random_coefficients()

    qy_leaf = qy.clone().requires_grad_(True)
    qx_leaf = qx.clone().requires_grad_(True)
    chi = cartesian_chi(qy_leaf, qx_leaf, wavelength, coefficients)
    expected_y, expected_x = torch.autograd.grad(chi.sum(), (qy_leaf, qx_leaf))

    got = cartesian_chi_gradient(qy, qx, wavelength, coefficients)

    scale = float(torch.stack((expected_y, expected_x)).abs().max())
    assert torch.allclose(got[0], expected_y, rtol=0, atol=1e-9 * scale)
    assert torch.allclose(got[1], expected_x, rtol=0, atol=1e-9 * scale)


def test_gradient_agrees_with_autograd_for_each_coefficient_alone():
    """Per-coefficient, so one wrong monomial cannot hide behind the other eleven."""
    qy, qx = _grid()
    wavelength = 0.0197
    for index, name in enumerate(COEFFICIENT_NAMES):
        coefficients = torch.zeros(len(COEFFICIENT_NAMES), dtype=torch.float64)
        coefficients[index] = 137.0

        qy_leaf = qy.clone().requires_grad_(True)
        qx_leaf = qx.clone().requires_grad_(True)
        chi = cartesian_chi(qy_leaf, qx_leaf, wavelength, coefficients)
        expected_y, expected_x = torch.autograd.grad(chi.sum(), (qy_leaf, qx_leaf))

        got = cartesian_chi_gradient(qy, qx, wavelength, coefficients)
        scale = max(float(torch.stack((expected_y, expected_x)).abs().max()), 1e-30)
        assert torch.allclose(
            got[0], expected_y, rtol=0, atol=1e-9 * scale
        ), f"{name}: qy derivative"
        assert torch.allclose(
            got[1], expected_x, rtol=0, atol=1e-9 * scale
        ), f"{name}: qx derivative"


def test_defocus_gradient_is_the_closed_form():
    """Pure defocus: chi = pi*C10*lambda*q^2, so d chi/d q = 2*pi*C10*lambda*q."""
    qy, qx = _grid()
    wavelength = 0.0197
    defocus = -75.0
    coefficients = torch.zeros(len(COEFFICIENT_NAMES), dtype=torch.float64)
    coefficients[COEFFICIENT_NAMES.index("C10")] = defocus

    got = cartesian_chi_gradient(qy, qx, wavelength, coefficients)

    factor = 2.0 * np.pi * defocus * wavelength
    assert torch.allclose(got[0], factor * qy, rtol=1e-12, atol=0)
    assert torch.allclose(got[1], factor * qx, rtol=1e-12, atol=0)


def test_rotationally_symmetric_terms_give_a_radial_gradient():
    """C10 and C30 depend on |q| only, so their gradient must point along q."""
    qy, qx = _grid()
    wavelength = 0.0197
    for name in ("C10", "C30"):
        coefficients = torch.zeros(len(COEFFICIENT_NAMES), dtype=torch.float64)
        coefficients[COEFFICIENT_NAMES.index(name)] = 250.0
        got = cartesian_chi_gradient(qy, qx, wavelength, coefficients)
        # cross product with q vanishes iff the gradient is parallel to q
        cross = got[0] * qx - got[1] * qy
        scale = float((got.pow(2).sum(0).sqrt() * torch.hypot(qy, qx)).max())
        assert float(cross.abs().max()) < 1e-14 * scale, name


def test_gradient_is_bitwise_reproducible():
    qy, qx = _grid()
    coefficients = _random_coefficients(seed=3)
    first = cartesian_chi_gradient(qy, qx, 0.0197, coefficients)
    second = cartesian_chi_gradient(qy, qx, 0.0197, coefficients)
    assert torch.equal(first, second)


def test_derivative_with_respect_to_a_zero_coefficient_is_not_zero():
    """chi is linear in the coefficients, so d chi/d C_i is independent of C_i.

    Zero-valued terms are skipped as a forward-only shortcut. If that shortcut
    also applied under autograd, every coefficient sitting at zero would report a
    zero derivative and an optimiser started from zeros could never move it --
    which is exactly the state a fresh aberration fit starts in.
    """
    qy, qx = _grid()
    weights = _weights(qy.shape)

    zeros = torch.zeros(len(COEFFICIENT_NAMES), dtype=torch.float64, requires_grad=True)
    (cartesian_chi(qy, qx, 0.0197, zeros) * weights).sum().backward()

    assert zeros.grad is not None
    reference = _reference_derivatives(qy, qx, weights)
    for index, name in enumerate(COEFFICIENT_NAMES):
        got, want = float(zeros.grad[index]), reference[index]
        assert got == pytest.approx(want, rel=1e-9), f"{name} derivative at zero"
        assert abs(got) > 0.1 * abs(want), f"{name} derivative reported as ~zero"


def _reference_derivatives(qy, qx, weights):
    """d/dC_i of the weighted sum, obtained one coefficient at a time.

    Each is evaluated at a non-zero value of that coefficient, where the term is
    never skipped -- so this is independent of the shortcut under test.
    """
    reference = []
    for index in range(len(COEFFICIENT_NAMES)):
        coefficients = torch.zeros(
            len(COEFFICIENT_NAMES), dtype=torch.float64, requires_grad=True
        )
        with torch.no_grad():
            coefficients[index] = 137.0
        (cartesian_chi(qy, qx, 0.0197, coefficients) * weights).sum().backward()
        reference.append(float(coefficients.grad[index]))
    return reference


def test_derivative_at_zero_matches_the_derivative_elsewhere():
    """Linearity again, stated as the property an optimiser depends on."""
    qy, qx = _grid()
    weights = _weights(qy.shape)

    grads = []
    for offset in (0.0, 250.0):
        coefficients = torch.full(
            (len(COEFFICIENT_NAMES),), offset, dtype=torch.float64, requires_grad=True
        )
        (cartesian_chi(qy, qx, 0.0197, coefficients) * weights).sum().backward()
        grads.append(coefficients.grad.clone())

    assert torch.allclose(grads[0], grads[1], rtol=1e-12, atol=0)


def test_gradient_function_derivative_with_respect_to_zero_coefficients_is_not_zero():
    """The same property for the closed-form gradient, which the shift model uses."""
    qy, qx = _grid()
    weights = _weights(qy.shape)

    zeros = torch.zeros(len(COEFFICIENT_NAMES), dtype=torch.float64, requires_grad=True)
    shifts = cartesian_chi_gradient(qy, qx, 0.0197, zeros)
    (shifts * weights).sum().backward()

    assert zeros.grad is not None
    nonzero = torch.full_like(zeros, 137.0).requires_grad_(True)
    (cartesian_chi_gradient(qy, qx, 0.0197, nonzero) * weights).sum().backward()

    for index, name in enumerate(COEFFICIENT_NAMES):
        got, want = float(zeros.grad[index]), float(nonzero.grad[index])
        assert got == pytest.approx(want, rel=1e-9), f"{name} derivative at zero"
        assert abs(got) > 0.1 * abs(want), f"{name} derivative reported as ~zero"
