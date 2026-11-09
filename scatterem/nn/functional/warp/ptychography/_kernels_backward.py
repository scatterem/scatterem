"""Adjoint kernels of the correction with respect to the aberrations.

Above the floor the weight is ``W = exp(i psi)``, so a coefficient ``c_j``
changes the corrected value ``C = G W`` by ``dC = i (dpsi/dc_j) C dc_j``. With
``D_j(X) = dchi(X)/dc_j`` (``dchi_cartesian_aberrations``):

- minus only: ``psi = chi(K - Q) - chi(K)``, so ``phi_j = D_j(Km) - D_j(K)``;
- plus only: ``psi = pi + chi(K) - chi(K + Q)``, so ``phi_j = D_j(K) - D_j(Kp)``;
- both, ``|Gamma| >= FLOOR``: ``psi = pi/2 - d``, so
  ``phi_j = -(D_j(Kp) - D_j(Km)) / 2``;
- floor band and non-contributing pairs: 0.

The jump of ``sgn(sigma)`` is ignored. With the torch cotangent ``g``
(``dL = Re sum conj(g) dC``) this gives ``dL/dc_j = -sum phi_j Im(conj(g) C)``.
The kernels add the NEGATIVE, ``sum phi_j Im(conj(g) C)``, to ``out_grad[j]``.
No factor ``1 / |Gamma|`` appears, so the error near a zero of ``Gamma``
stays at the forward's own phase error.

Reduction: a block-wide ``tile_sum`` per coefficient and one
``tile_atomic_add`` per block. Every thread reaches the tile calls; a
non-contributing thread adds 0. The drivers launch with ``block_dim=256``.
"""

import warp as wp

from ._funcs import (
    CASE_BOTH,
    CASE_MINUS,
    CASE_PLUS,
    _bf_factor,
    _rot_q,
    _ssb_apply,
    _ssb_general,
    dchi_cartesian_aberrations,
)

# No FMA contraction: nvcc contracts the same expression differently in
# different kernels (a loop body, a second call of a helper), which breaks
# the bit identity between the spellings. With ``fuse_fp`` off every kernel
# rounds each operation the same way.
wp.set_module_options({"fuse_fp": False})


@wp.func
def _phase_rate(
    case: wp.float32,
    ky: wp.float32,
    kx: wp.float32,
    qr: wp.vec2,
    wavelength: wp.float32,
    j: int,
) -> wp.float32:
    """``phi_j = dpsi/dc_j`` of the weight phase for one pair."""
    out = wp.float32(0.0)
    if case == CASE_MINUS:
        dm = dchi_cartesian_aberrations(ky - qr[0], kx - qr[1], wavelength, j)
        out = dm - dchi_cartesian_aberrations(ky, kx, wavelength, j)
    elif case == CASE_PLUS:
        dp = dchi_cartesian_aberrations(ky + qr[0], kx + qr[1], wavelength, j)
        out = dchi_cartesian_aberrations(ky, kx, wavelength, j) - dp
    elif case == CASE_BOTH:
        dp = dchi_cartesian_aberrations(ky + qr[0], kx + qr[1], wavelength, j)
        dm = dchi_cartesian_aberrations(ky - qr[0], kx - qr[1], wavelength, j)
        out = -0.5 * (dp - dm)
    return out


@wp.func
def _pair_gradient_terms(
    gin: wp.vec2,
    cot: wp.vec2,
    qy: wp.float32,
    qx: wp.float32,
    ky: wp.float32,
    kx: wp.float32,
    aberrations: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    sin_rot: wp.float32,
    cos_rot: wp.float32,
    semiconvergence_angle: wp.float32,
    wavelength: wp.float32,
) -> wp.vec4:
    """``(case, Im(conj(g) C), qr_y, qr_x)`` of one pair."""
    qr = _rot_q(qy, qx, sin_rot, cos_rot)
    A0 = _bf_factor(ky, kx, aberrations, semiconvergence_angle, wavelength)
    t = _ssb_general(ky, kx, qr, A0, aberrations, semiconvergence_angle, wavelength)
    case = t[2]
    val = wp.float32(0.0)
    if case == CASE_MINUS or case == CASE_PLUS or case == CASE_BOTH:
        c = _ssb_apply(gin, t)
        val = cot[0] * c[1] - cot[1] * c[0]  # pyright: ignore[reportOperatorIssue]
    else:
        case = wp.float32(0.0)
    return wp.vec4(case, val, qr[0], qr[1])


@wp.func
def _reduce_into(
    terms: wp.vec4,
    ky: wp.float32,
    kx: wp.float32,
    wavelength: wp.float32,
    n_coeffs: int,
    out_grad: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
):
    """Block-sum ``phi_j * Im(conj(g) C)`` and add it to ``out_grad[j]``."""
    qr = wp.vec2(terms[2], terms[3])
    for j in range(n_coeffs):
        v = wp.float32(0.0)
        if terms[0] != 0.0:
            v = _phase_rate(terms[0], ky, kx, qr, wavelength, j) * terms[1]  # pyright: ignore[reportArgumentType]
        s = wp.tile_sum(wp.tile(v))
        wp.tile_atomic_add(out_grad, s, offset=j)  # pyright: ignore[reportArgumentType, reportCallIssue]


@wp.kernel(enable_backward=False)
def _direct_ptychography_backward_analytic(
    G: wp.array(dtype=wp.vec2, ndim=3),  # pyright: ignore[reportInvalidTypeForm]
    dL_dG: wp.array(dtype=wp.vec2, ndim=3),  # pyright: ignore[reportInvalidTypeForm]
    qx: wp.array(ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    qy: wp.array(ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    kx: wp.array(ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    ky: wp.array(ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    aberrations: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    sin_rot: wp.float32,
    cos_rot: wp.float32,
    semiconvergence_angle: wp.float32,
    wavelength: wp.float32,
    n_coeffs: int,
    out_grad: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
):
    """Accumulate the aberration gradient of the correction."""
    iqy, iqx, ik = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    kyv = wp.float32(ky[ik])
    kxv = wp.float32(kx[ik])
    terms = _pair_gradient_terms(
        G[iqy, iqx, ik], dL_dG[iqy, iqx, ik],
        wp.float32(qy[iqy]), wp.float32(qx[iqx]), kyv, kxv, aberrations,
        sin_rot, cos_rot, semiconvergence_angle, wavelength,
    )  # fmt: skip
    _reduce_into(terms, kyv, kxv, wavelength, n_coeffs, out_grad)


@wp.kernel(enable_backward=False)
def _direct_ptychography_backward_analytic_ksum(
    G: wp.array(dtype=wp.vec2, ndim=3),  # pyright: ignore[reportInvalidTypeForm]
    dL_dS: wp.array(dtype=wp.vec2, ndim=2),  # pyright: ignore[reportInvalidTypeForm]
    qx: wp.array(ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    qy: wp.array(ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    kx: wp.array(ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    ky: wp.array(ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    aberrations: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    sin_rot: wp.float32,
    cos_rot: wp.float32,
    semiconvergence_angle: wp.float32,
    wavelength: wp.float32,
    n_coeffs: int,
    out_grad: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
):
    """Accumulate the aberration gradient of the correct-and-sum."""
    iqy, iqx, ik = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    kyv = wp.float32(ky[ik])
    kxv = wp.float32(kx[ik])
    terms = _pair_gradient_terms(
        G[iqy, iqx, ik], dL_dS[iqy, iqx],
        wp.float32(qy[iqy]), wp.float32(qx[iqx]), kyv, kxv, aberrations,
        sin_rot, cos_rot, semiconvergence_angle, wavelength,
    )  # fmt: skip
    _reduce_into(terms, kyv, kxv, wavelength, n_coeffs, out_grad)
