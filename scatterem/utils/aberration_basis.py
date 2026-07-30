"""The wave-aberration function and its polar/cartesian coefficient conventions.

Written from the standard expansion rather than adapted from an existing
implementation. It replaces two pieces that an Apache-2.0 release cannot carry:
``utils/stem._cartesian_aberrations`` (copied text of unrecorded origin, its
comments OCR-mangled) and ``utils/transfer.{cartesian2polar, polar2cartesian}``
(a verbatim block from GPL-3.0-or-later abTEM, complete with an unexplained
``k = sqrt(3 + sqrt(8))``).

The aberration function is

    chi(q) = (2*pi / lambda) * sum over terms of
             (1/n) * r^n * [ Ca * cos(m*phi) + Cb * sin(m*phi) ]

with ``r = lambda * |q|`` the scattering angle, ``phi`` its azimuth, ``n`` the
radial power and ``m`` the rotational order. Writing it this way -- radial power
times an explicit ``cos(m*phi)`` -- rather than as the expanded cartesian
polynomials makes each term's rotational symmetry visible in the source, which
is the property the tests check.

Naming follows the convention already in use: ``Cnm`` with an ``a`` suffix is the
``cos(m*phi)`` coefficient and ``b`` the ``sin(m*phi)`` one, so

    Ca = C * cos(m * phi_m)      Cb = C * sin(m * phi_m)

and therefore ``C = hypot(Ca, Cb)``, ``phi_m = atan2(Cb, Ca) / m``. Magnitudes
are non-negative by construction. The abTEM conversion this replaces returned a
*negative* magnitude and a sign-flipped angle, which meant the reported polar
pair did not describe the wavefront that had actually been fitted; see
``tests/test_aberration_basis.py::test_reported_polar_pair_describes_the_same_wavefront``.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import torch
from torch import Tensor

#: Coefficient order used by the flat 12-element aberration vectors.
COEFFICIENT_NAMES: Tuple[str, ...] = (
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

#: ``(name, radial power n, rotational order m, "cos" | "sin")`` for each entry
#: of :data:`COEFFICIENT_NAMES`. The prefactor of every term is ``1 / n``.
_TERMS: Tuple[Tuple[str, int, int, str], ...] = (
    ("C10", 2, 0, "cos"),
    ("C12a", 2, 2, "cos"),
    ("C12b", 2, 2, "sin"),
    ("C21a", 3, 1, "cos"),
    ("C21b", 3, 1, "sin"),
    ("C23a", 3, 3, "cos"),
    ("C23b", 3, 3, "sin"),
    ("C30", 4, 0, "cos"),
    ("C32a", 4, 2, "cos"),
    ("C32b", 4, 2, "sin"),
    ("C34a", 4, 4, "cos"),
    ("C34b", 4, 4, "sin"),
)

#: ``polar magnitude -> (angle key, cartesian cos key, cartesian sin key, order)``
_POLAR_PAIRS: Tuple[Tuple[str, str, str, str, int], ...] = (
    ("C12", "phi12", "C12a", "C12b", 2),
    ("C21", "phi21", "C21a", "C21b", 1),
    ("C23", "phi23", "C23a", "C23b", 3),
    ("C32", "phi32", "C32a", "C32b", 2),
    ("C34", "phi34", "C34a", "C34b", 4),
)

#: Coefficients with no azimuthal dependence, carried across unchanged.
_ROTATIONALLY_SYMMETRIC: Tuple[str, ...] = ("C10", "C30")


def cartesian_chi(
    qy: Tensor, qx: Tensor, wavelength: float, coefficients: Sequence[float] | Tensor
) -> Tensor:
    """Evaluate the aberration phase ``chi`` in radians.

    Args:
        qy: reciprocal-space y coordinates, in inverse Angstrom.
        qx: reciprocal-space x coordinates, same shape as ``qy``.
        wavelength: electron wavelength in Angstrom (see
            :func:`scatterem.utils.physics.electron_wavelength`).
        coefficients: the 12 cartesian coefficients in
            :data:`COEFFICIENT_NAMES` order, in Angstrom.

    Returns:
        ``chi`` with the broadcast shape of ``qy`` and ``qx``.
    """
    coefficients = torch.as_tensor(coefficients)
    if coefficients.shape[-1] != len(COEFFICIENT_NAMES):
        raise ValueError(
            f"expected {len(COEFFICIENT_NAMES)} aberration coefficients in "
            f"{COEFFICIENT_NAMES} order; got {tuple(coefficients.shape)}"
        )

    # Work in scattering angle, which is what the expansion is defined over.
    ay = qy * wavelength
    ax = qx * wavelength
    radius = torch.hypot(ay, ax)
    azimuth = torch.atan2(ay, ax)

    # Skipping zero coefficients is a forward-only shortcut. chi is linear in the
    # coefficients, so d chi/d C_i does not depend on C_i -- a zero coefficient
    # still has a non-zero derivative, and skipping its term would report that
    # derivative as zero, leaving an optimiser unable to move it off zero. It
    # would also detach the result from qy/qx entirely when every coefficient is
    # zero, which is a legitimate starting point for a fit.
    skip_zeros = not (coefficients.requires_grad or radius.requires_grad)

    chi = torch.zeros_like(radius)
    for index, (_, power, order, trig) in enumerate(_TERMS):
        value = coefficients[index]
        if skip_zeros and bool(torch.all(value == 0)):
            continue
        angular = (
            torch.cos(order * azimuth) if trig == "cos" else torch.sin(order * azimuth)
        )
        chi = chi + (value / power) * radius**power * angular

    return chi * (2.0 * math.pi / wavelength)


def polar_to_cartesian(polar: Dict[str, float]) -> Dict[str, float]:
    """Convert magnitude/angle aberrations to the cartesian pairs.

    Missing entries are taken as zero, so a caller may pass only the
    coefficients it cares about.
    """
    cartesian: Dict[str, float] = {}
    for name in _ROTATIONALLY_SYMMETRIC:
        cartesian[name] = float(polar.get(name, 0.0))
    for magnitude, angle, cos_key, sin_key, order in _POLAR_PAIRS:
        c = float(polar.get(magnitude, 0.0))
        phi = float(polar.get(angle, 0.0))
        cartesian[cos_key] = c * math.cos(order * phi)
        cartesian[sin_key] = c * math.sin(order * phi)
    return cartesian


def cartesian_to_polar(cartesian: Dict[str, float]) -> Dict[str, float]:
    """Convert cartesian aberration pairs to magnitude and angle.

    The magnitude is non-negative and the angle is the one that satisfies
    ``C * cos(m * (phi - phi_m)) == Ca * cos(m*phi) + Cb * sin(m*phi)``, so the
    returned pair describes the same wavefront as its input. Missing entries are
    taken as zero.

    The angle is only determined modulo ``2*pi/m`` -- an ``m``-fold term is
    physically unchanged by such a rotation -- and is returned in the branch
    ``atan2`` gives, i.e. within ``+-pi/m``.
    """
    polar: Dict[str, float] = {}
    for name in _ROTATIONALLY_SYMMETRIC:
        polar[name] = float(cartesian.get(name, 0.0))
    for magnitude, angle, cos_key, sin_key, order in _POLAR_PAIRS:
        a = float(cartesian.get(cos_key, 0.0))
        b = float(cartesian.get(sin_key, 0.0))
        polar[magnitude] = math.hypot(a, b)
        polar[angle] = math.atan2(b, a) / order
    return polar


#: ``(name, d/du monomials, d/dv monomials)`` where each monomial is
#: ``(coefficient, power of u, power of v)``.
#:
#: chi is a polynomial in the scattering-angle components ``u = lambda*qx`` and
#: ``v = lambda*qy``, so its gradient is closed-form. Differentiating the
#: expansion in :func:`cartesian_chi` term by term, with the ``1/n`` prefactor
#: folded in, gives the table below. Written out rather than derived at runtime so
#: it can be read against the expansion, and checked against autograd -- which
#: ``tests/test_aberration_basis.py`` does.
_GRADIENT_TERMS: Tuple[Tuple[str, Tuple, Tuple], ...] = (
    ("C10", ((1.0, 1, 0),), ((1.0, 0, 1),)),
    ("C12a", ((1.0, 1, 0),), ((-1.0, 0, 1),)),
    ("C12b", ((1.0, 0, 1),), ((1.0, 1, 0),)),
    ("C21a", ((1.0, 2, 0), (1 / 3, 0, 2)), ((2 / 3, 1, 1),)),
    ("C21b", ((2 / 3, 1, 1),), ((1.0, 0, 2), (1 / 3, 2, 0))),
    ("C23a", ((1.0, 2, 0), (-1.0, 0, 2)), ((-2.0, 1, 1),)),
    ("C23b", ((2.0, 1, 1),), ((1.0, 2, 0), (-1.0, 0, 2))),
    ("C30", ((1.0, 3, 0), (1.0, 1, 2)), ((1.0, 2, 1), (1.0, 0, 3))),
    ("C32a", ((1.0, 3, 0),), ((-1.0, 0, 3),)),
    ("C32b", ((1.5, 2, 1), (0.5, 0, 3)), ((0.5, 3, 0), (1.5, 1, 2))),
    ("C34a", ((1.0, 3, 0), (-3.0, 1, 2)), ((-3.0, 2, 1), (1.0, 0, 3))),
    ("C34b", ((3.0, 2, 1), (-1.0, 0, 3)), ((1.0, 3, 0), (-3.0, 1, 2))),
)


def cartesian_chi_gradient(
    qy: Tensor, qx: Tensor, wavelength: float, coefficients: Sequence[float] | Tensor
) -> Tensor:
    """Gradient of the aberration phase, ``[d chi/d qy, d chi/d qx]``.

    Closed-form, from the same expansion as :func:`cartesian_chi`. Two things fall
    out of the algebra and are worth noting because they make the result cheap:
    the ``2*pi/lambda`` prefactor of chi and the ``lambda`` from
    ``d/dq = lambda * d/du`` cancel to a bare ``2*pi``, and every term is a
    monomial in ``u`` and ``v``.

    Args:
        qy: reciprocal-space y coordinates, in inverse Angstrom.
        qx: reciprocal-space x coordinates, same shape as ``qy``.
        wavelength: electron wavelength in Angstrom.
        coefficients: the 12 cartesian coefficients in
            :data:`COEFFICIENT_NAMES` order, in Angstrom.

    Returns:
        ``(2, ...)`` with the ``qy`` derivative first, in radians per inverse
        Angstrom.
    """
    coefficients = torch.as_tensor(coefficients)
    if coefficients.shape[-1] != len(COEFFICIENT_NAMES):
        raise ValueError(
            f"expected {len(COEFFICIENT_NAMES)} aberration coefficients in "
            f"{COEFFICIENT_NAMES} order; got {tuple(coefficients.shape)}"
        )

    u = qx * wavelength
    v = qy * wavelength

    # See cartesian_chi: the zero-coefficient shortcut is forward-only, because
    # the gradient with respect to a coefficient does not vanish when that
    # coefficient does. Here it matters twice over -- the rotation the caller
    # fits enters only through u and v, so short-circuiting an all-zero
    # wavefront would detach the rotation as well.
    skip_zeros = not (coefficients.requires_grad or u.requires_grad or v.requires_grad)

    d_du = torch.zeros_like(u * v)
    d_dv = torch.zeros_like(d_du)
    for index, (_, du_terms, dv_terms) in enumerate(_GRADIENT_TERMS):
        value = coefficients[index]
        if skip_zeros and bool(torch.all(value == 0)):
            continue
        for factor, p, q in du_terms:
            d_du = d_du + value * factor * u**p * v**q
        for factor, p, q in dv_terms:
            d_dv = d_dv + value * factor * u**p * v**q

    two_pi = 2.0 * math.pi
    return torch.stack((two_pi * d_dv, two_pi * d_du))
