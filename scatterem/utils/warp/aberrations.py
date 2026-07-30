"""The wave-aberration function as a Warp device function.

The Warp counterpart of :func:`scatterem.utils.aberration_basis.cartesian_chi`,
written from the same expansion and validated against it
(``tests/test_warp_aberrations.py``). It replaces
``utils/warp/transfer.aberration_function_cartesian``, whose torch sibling was
refuted as a verbatim block from GPL-3.0-or-later abTEM and which an
Apache-2.0 release therefore cannot carry.

The expansion is

    chi(q) = (2*pi / lambda) * sum over terms of
             (1/n) * r^n * [ Ca * cos(m*phi) + Cb * sin(m*phi) ]

with ``r = lambda * |q|`` the scattering angle, ``phi`` its azimuth measured
from the x axis, ``n`` the radial power and ``m`` the rotational order.

Evaluating ``cos(m*phi)`` directly would cost an ``atan2`` plus a trig call per
term, and this runs per detector pixel per scan position inside the direct
ptychography kernels. Instead the whole angular basis follows from one identity,

    r^m * exp(i*m*phi) = (u + i*v)^m        u = lambda*qx,  v = lambda*qy

so ``r^m cos(m*phi)`` and ``r^m sin(m*phi)`` are just the real and imaginary
parts of a complex power, reachable by repeated multiplication:

    (u + i*v)^(k+1) = (u + i*v)^k * (u + i*v)

Three complex multiplies give every angular factor up to ``m = 4`` with no
transcendentals. Terms whose radial power exceeds their rotational order carry
the leftover as a factor of ``r^2`` -- for example ``r^4 cos(2*phi)`` is
``r^2 * (r^2 cos(2*phi))``, i.e. ``r2 * re2`` -- which keeps each term's
symmetry legible in the source instead of hiding it in an expanded polynomial.

Coefficients arrive in the flat 12-element order of
:data:`scatterem.utils.aberration_basis.COEFFICIENT_NAMES`::

    0:C10  1:C12a  2:C12b  3:C21a  4:C21b  5:C23a
    6:C23b 7:C30   8:C32a  9:C32b 10:C34a 11:C34b
"""

import warp as wp


@wp.func
def aberration_function_cartesian(
    qy: wp.float32,
    qx: wp.float32,
    wavelength: wp.float32,
    aberrations: wp.array(dtype=wp.float32),
) -> wp.float32:
    """Evaluate the aberration phase ``chi`` in radians.

    Args:
        qy: reciprocal-space y coordinate, in inverse Angstrom.
        qx: reciprocal-space x coordinate, in inverse Angstrom.
        wavelength: electron wavelength in Angstrom.
        aberrations: the 12 cartesian coefficients in ``COEFFICIENT_NAMES``
            order, in Angstrom.

    Returns:
        ``chi`` in radians.
    """
    # Work in scattering angle, which is what the expansion is defined over.
    u = qx * wavelength
    v = qy * wavelength

    r2 = u * u + v * v

    # (u + i*v)^m by repeated multiplication: re_m = r^m cos(m*phi),
    # im_m = r^m sin(m*phi).
    re2 = u * u - v * v
    im2 = 2.0 * u * v
    re3 = re2 * u - im2 * v
    im3 = re2 * v + im2 * u
    re4 = re3 * u - im3 * v
    im4 = re3 * v + im3 * u

    # n = 2. The 1/n prefactor of the expansion is folded into each literal.
    chi = 0.5 * aberrations[0] * r2  # C10  r^2
    chi += 0.5 * aberrations[1] * re2  # C12a r^2 cos(2 phi)
    chi += 0.5 * aberrations[2] * im2  # C12b r^2 sin(2 phi)

    # n = 3. The m = 1 pair spends its surplus radial power as r^2.
    third = 1.0 / 3.0
    chi += third * aberrations[3] * r2 * u  # C21a r^3 cos(phi)
    chi += third * aberrations[4] * r2 * v  # C21b r^3 sin(phi)
    chi += third * aberrations[5] * re3  # C23a r^3 cos(3 phi)
    chi += third * aberrations[6] * im3  # C23b r^3 sin(3 phi)

    # n = 4. The m = 0 and m = 2 terms likewise carry a factor of r^2.
    chi += 0.25 * aberrations[7] * r2 * r2  # C30  r^4
    chi += 0.25 * aberrations[8] * r2 * re2  # C32a r^4 cos(2 phi)
    chi += 0.25 * aberrations[9] * r2 * im2  # C32b r^4 sin(2 phi)
    chi += 0.25 * aberrations[10] * re4  # C34a r^4 cos(4 phi)
    chi += 0.25 * aberrations[11] * im4  # C34b r^4 sin(4 phi)

    return chi * 2.0 * wp.pi / wavelength
