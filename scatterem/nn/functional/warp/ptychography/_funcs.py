"""Warp device functions shared by the direct-ptychography kernels.

Physics (Rodenburg, McCallum and Nellist, Ultramicroscopy 48 (1993) 304;
Pennycook et al., Ultramicroscopy 151 (2015) 160). With the probe aperture
function ``A(K) = H(K) exp(-i chi(K))``, the scan transform ``G(Q, K)`` of a
weak phase object ``exp(i phi)`` is ``i Phi(Q) Gamma(Q, K)`` for ``Q != 0``,
with the double-overlap transfer function

    Gamma(Q, K) = conj(A(K)) A(K - Q) - A(K) conj(A(K + Q)).

The corrected value is ``G conj(Gamma) / |Gamma|``. The kernels use the
factorised form of ``Gamma`` with ``chip = chi(K + Q)``, ``chim = chi(K - Q)``,
``s = (chip + chim) / 2``, ``d = (chip - chim) / 2`` and
``sigma = sin(s - chi(K))``:

- only ``K - Q`` inside: ``Gamma = exp(i (chi(K) - chim))``;
- only ``K + Q`` inside: ``Gamma = -exp(i (chip - chi(K)))``;
- both inside: ``Gamma = -2 i sigma exp(i d)``.

The both-band phase ``d - pi/2`` does not cancel near a zero of ``Gamma``, so
its float32 error does not grow as ``|Gamma| -> 0``. ``chi(K)`` enters only as
``exp(-i chi(K))``, which is the bright-field factor ``A(K)``.

Magnitude floor: ``FLOOR = GAMMA_FLOOR = 2**-20``. In the both-band case the weight is
``conj(Gamma) / max(|Gamma|, FLOOR)``. Where ``Gamma`` is exactly zero (no
double overlap, ``Q = 0`` or ``sigma == 0``) the corrected value is zero.
"""

import warp as wp

from scatterem.physics.kernels.lens_wavefront_wp import (
    lens_wavefront_f32,
    lens_wavefront_focus_f32,
)
from scatterem.utils.warp.complex import cconj, cexp, cmul
from scatterem.utils.warp.misc import probe_cone_pass

from ._aperture_sq import _aperture_sq

#: Magnitude floor of ``|Gamma|`` (float32 spacing of numbers in [8, 16)), as a
#: Python float. Torch code that must weigh a pair as the kernels do imports
#: this value.
GAMMA_FLOOR = 2.0**-20
#: ``GAMMA_FLOOR`` as a Warp constant, for the kernels.
FLOOR = wp.constant(wp.float32(GAMMA_FLOOR))

#: Case codes in slot 2 of the ``_ssb_terms`` result.
CASE_NONE = wp.constant(wp.float32(0.0))
CASE_MINUS = wp.constant(wp.float32(1.0))
CASE_PLUS = wp.constant(wp.float32(2.0))
CASE_BOTH = wp.constant(wp.float32(3.0))
CASE_FLOOR = wp.constant(wp.float32(4.0))


@wp.func
def dchi_cartesian_aberrations(
    qy: wp.float32,
    qx: wp.float32,
    wavelength: wp.float32,
    j: int,
) -> wp.float32:
    """Derivative of the aberration phase with respect to one coefficient.

    ``chi = (2 pi / lambda) sum_j c_j b_j(u, v)`` with ``u = lambda qx`` and
    ``v = lambda qy`` is linear in ``c``, so the result is
    ``(2 pi / lambda) b_j`` in rad per Angstrom. Other ``j`` give 0.0.
    """
    u = qx * wavelength
    v = qy * wavelength
    r2 = u * u + v * v  # pyright: ignore[reportOperatorIssue]
    re2 = u * u - v * v  # pyright: ignore[reportOperatorIssue]
    im2 = 2.0 * u * v  # pyright: ignore[reportOperatorIssue]
    re3 = re2 * u - im2 * v  # pyright: ignore[reportOperatorIssue]
    im3 = re2 * v + im2 * u  # pyright: ignore[reportOperatorIssue]
    re4 = re3 * u - im3 * v
    im4 = re3 * v + im3 * u
    third = wp.float32(1.0) / wp.float32(3.0)
    b = wp.float32(0.0)
    if j == 0:
        b = 0.5 * r2
    elif j == 1:
        b = 0.5 * re2
    elif j == 2:
        b = 0.5 * im2
    elif j == 3:
        b = third * r2 * u
    elif j == 4:
        b = third * r2 * v
    elif j == 5:
        b = third * re3
    elif j == 6:
        b = third * im3
    elif j == 7:
        b = 0.25 * r2 * r2
    elif j == 8:
        b = 0.25 * r2 * re2
    elif j == 9:
        b = 0.25 * r2 * im2
    elif j == 10:
        b = 0.25 * re4
    elif j == 11:
        b = 0.25 * im4
    return b * 2.0 * wp.pi / wavelength


@wp.func
def _rot_q(
    qy: wp.float32, qx: wp.float32, sin_rot: wp.float32, cos_rot: wp.float32
) -> wp.vec2:
    """Rotated scan frequency as ``(y, x)``, counter-clockwise by theta."""
    return wp.vec2(sin_rot * qx + cos_rot * qy, cos_rot * qx - sin_rot * qy)  # pyright: ignore[reportOperatorIssue]


@wp.func
def _bf_factor(
    ky: wp.float32,
    kx: wp.float32,
    aberrations: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    semiconvergence_angle: wp.float32,
    wavelength: wp.float32,
) -> wp.vec2:
    """Bright-field factor ``A(K) = H(K) exp(-i chi(K))``."""
    h = probe_cone_pass(wavelength, semiconvergence_angle, ky, kx)
    chi = lens_wavefront_f32(aberrations, ky, kx, wavelength)
    return cexp(h, -chi)


@wp.func
def _ssb_terms(
    A0: wp.vec2,
    h0: wp.float32,
    hp: wp.float32,
    hm: wp.float32,
    chip: wp.float32,
    chim: wp.float32,
    q_is_zero: int,
) -> wp.vec4:
    """Weight ``W``, case code and ``|Gamma|`` of one ``(Q, K)`` pair.

    Returns ``(Re W, Im W, case, |Gamma|)``. ``W`` is
    ``conj(Gamma) / max(|Gamma|, FLOOR)``; it is (0, 0) for ``CASE_NONE``.
    """
    out = wp.vec4(0.0, 0.0, 0.0, 0.0)
    if h0 != 0.0 and q_is_zero == 0:
        if hm != 0.0 and hp == 0.0:
            # conj(Gamma) = exp(i (chim - chi0)) = A0 exp(i chim)
            w = cmul(A0, wp.vec2(wp.cos(chim), wp.sin(chim)))
            out = wp.vec4(w[0], w[1], CASE_MINUS, 1.0)
        elif hp != 0.0 and hm == 0.0:
            # conj(Gamma) = -exp(i (chi0 - chip)) = -conj(A0) exp(-i chip)
            w = cmul(cconj(A0), wp.vec2(wp.cos(chip), -wp.sin(chip)))
            out = wp.vec4(-w[0], -w[1], CASE_PLUS, 1.0)  # pyright: ignore[reportOperatorIssue]
        elif hp != 0.0 and hm != 0.0:
            s = 0.5 * (chip + chim)
            d = 0.5 * (chip - chim)
            # sigma = sin(s - chi0), with cos(chi0) = Re A0, sin(chi0) = -Im A0
            sigma = wp.sin(s) * A0[0] + wp.cos(s) * A0[1]  # pyright: ignore[reportOperatorIssue]
            if sigma != 0.0:
                g = 2.0 * wp.abs(sigma)  # pyright: ignore[reportOperatorIssue]
                # conj(Gamma) / |Gamma| = i sgn(sigma) exp(-i d)
                a = wp.float32(-1.0)
                if sigma > 0.0:
                    a = wp.float32(1.0)
                case = CASE_BOTH
                if g < FLOOR:
                    a = 2.0 * sigma / FLOOR
                    case = CASE_FLOOR
                out = wp.vec4(a * wp.sin(d), a * wp.cos(d), case, g)
    return out


@wp.func
def _ssb_apply(Gin: wp.vec2, t: wp.vec4) -> wp.vec2:
    """Corrected value ``Gin W``; exactly (0, 0) for a non-contributing pair."""
    out = wp.vec2(0.0, 0.0)
    if t[2] != CASE_NONE:
        out = cmul(Gin, wp.vec2(t[0], t[1]))
    return out


@wp.func
def _ssb_general(
    ky: wp.float32,
    kx: wp.float32,
    qr: wp.vec2,
    A0: wp.vec2,
    aberrations: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    semiconvergence_angle: wp.float32,
    wavelength: wp.float32,
) -> wp.vec4:
    """``_ssb_terms`` with the general ``chi`` and ``probe_cone_pass``."""
    kpy = ky + qr[0]
    kpx = kx + qr[1]
    kmy = ky - qr[0]
    kmx = kx - qr[1]
    chip = lens_wavefront_f32(aberrations, kpy, kpx, wavelength)
    chim = lens_wavefront_f32(aberrations, kmy, kmx, wavelength)
    h0 = probe_cone_pass(wavelength, semiconvergence_angle, ky, kx)
    hp = probe_cone_pass(wavelength, semiconvergence_angle, kpy, kpx)
    hm = probe_cone_pass(wavelength, semiconvergence_angle, kmy, kmx)
    qz = wp.int32(0)
    if qr[0] == 0.0 and qr[1] == 0.0:
        qz = 1
    return _ssb_terms(A0, h0, hp, hm, chip, chim, qz)  # pyright: ignore[reportArgumentType]


@wp.func
def _ssb_defocus(
    ky: wp.float32,
    kx: wp.float32,
    qr: wp.vec2,
    A0: wp.vec2,
    aberrations: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    semiconvergence_angle: wp.float32,
    wavelength: wp.float32,
) -> wp.vec4:
    """``_ssb_general`` with the defocus-only ``chi``."""
    kpy = ky + qr[0]
    kpx = kx + qr[1]
    kmy = ky - qr[0]
    kmx = kx - qr[1]
    chip = lens_wavefront_focus_f32(aberrations, kpy, kpx, wavelength)
    chim = lens_wavefront_focus_f32(aberrations, kmy, kmx, wavelength)
    h0 = probe_cone_pass(wavelength, semiconvergence_angle, ky, kx)
    hp = probe_cone_pass(wavelength, semiconvergence_angle, kpy, kpx)
    hm = probe_cone_pass(wavelength, semiconvergence_angle, kmy, kmx)
    qz = wp.int32(0)
    if qr[0] == 0.0 and qr[1] == 0.0:
        qz = 1
    return _ssb_terms(A0, h0, hp, hm, chip, chim, qz)  # pyright: ignore[reportArgumentType]


@wp.func
def _ssb_sq(
    ky: wp.float32,
    kx: wp.float32,
    qr: wp.vec2,
    A0: wp.vec2,
    aberrations: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    s_thresh: wp.float32,
    wavelength: wp.float32,
) -> wp.vec4:
    """``_ssb_general`` with the squared-radius aperture."""
    kpy = ky + qr[0]
    kpx = kx + qr[1]
    kmy = ky - qr[0]
    kmx = kx - qr[1]
    chip = lens_wavefront_f32(aberrations, kpy, kpx, wavelength)
    chim = lens_wavefront_f32(aberrations, kmy, kmx, wavelength)
    h0 = _aperture_sq(ky, kx, s_thresh)
    hp = _aperture_sq(kpy, kpx, s_thresh)
    hm = _aperture_sq(kmy, kmx, s_thresh)
    qz = wp.int32(0)
    if qr[0] == 0.0 and qr[1] == 0.0:
        qz = 1
    return _ssb_terms(A0, h0, hp, hm, chip, chim, qz)  # pyright: ignore[reportArgumentType]


@wp.func
def _mask_bits(h0: wp.float32, hp: wp.float32, hm: wp.float32) -> wp.uint8:
    """Pack the flags: bit 0 = H(K + Q), bit 1 = H(K - Q), bit 2 = H(K)."""
    b = wp.int32(0)
    if hp != 0.0:
        b = b + 1
    if hm != 0.0:
        b = b + 2
    if h0 != 0.0:
        b = b + 4
    return wp.uint8(b)


@wp.func
def _bit(bits: int, k: int) -> wp.float32:
    """Flag ``k`` of a mask byte as 0.0 or 1.0."""
    out = wp.float32(0.0)
    if ((bits >> k) & 1) != 0:
        out = wp.float32(1.0)
    return out


@wp.func
def _ssb_masked(
    ky: wp.float32,
    kx: wp.float32,
    qr: wp.vec2,
    A0: wp.vec2,
    aberrations: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    wavelength: wp.float32,
    bits: int,
) -> wp.vec4:
    """``_ssb_general`` with the aperture flags from a mask byte."""
    kpy = ky + qr[0]
    kpx = kx + qr[1]
    kmy = ky - qr[0]
    kmx = kx - qr[1]
    chip = lens_wavefront_f32(aberrations, kpy, kpx, wavelength)
    chim = lens_wavefront_f32(aberrations, kmy, kmx, wavelength)
    qz = wp.int32(0)
    if qr[0] == 0.0 and qr[1] == 0.0:
        qz = 1
    return _ssb_terms(A0, _bit(bits, 2), _bit(bits, 0), _bit(bits, 1), chip, chim, qz)  # pyright: ignore[reportArgumentType]


@wp.func
def _ssb_masked_defocus(
    ky: wp.float32,
    kx: wp.float32,
    qr: wp.vec2,
    A0: wp.vec2,
    aberrations: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    wavelength: wp.float32,
    bits: int,
) -> wp.vec4:
    """``_ssb_masked`` with the defocus-only ``chi``."""
    kpy = ky + qr[0]
    kpx = kx + qr[1]
    kmy = ky - qr[0]
    kmx = kx - qr[1]
    chip = lens_wavefront_focus_f32(aberrations, kpy, kpx, wavelength)
    chim = lens_wavefront_focus_f32(aberrations, kmy, kmx, wavelength)
    qz = wp.int32(0)
    if qr[0] == 0.0 and qr[1] == 0.0:
        qz = 1
    return _ssb_terms(A0, _bit(bits, 2), _bit(bits, 0), _bit(bits, 1), chip, chim, qz)  # pyright: ignore[reportArgumentType]


@wp.func
def _chi_zero_add(
    Gin: wp.vec2, h0: wp.float32, hp: wp.float32, hm: wp.float32, q_is_zero: int
) -> wp.vec2:
    """Corrected value for an all-zero aberration vector.

    Minus only gives ``Gin``, plus only gives ``-Gin``; both bands give 0
    (``sigma = sin(0) = 0``), as do the non-overlap cases.
    """
    out = wp.vec2(0.0, 0.0)
    if h0 != 0.0 and q_is_zero == 0:
        if hm != 0.0 and hp == 0.0:
            out = Gin
        elif hp != 0.0 and hm == 0.0:
            out = -Gin
    return out
