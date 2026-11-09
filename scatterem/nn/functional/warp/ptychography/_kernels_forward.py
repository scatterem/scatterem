"""Direct-ptychography correction kernels and the bright-field factor.

The corrected value and its rules are in ``_funcs``. Every kernel evaluates
``chi(K + Q)`` and then ``chi(K - Q)`` before any aperture branch, and passes
them to the one helper ``_ssb_terms``, so the spellings stay bit-identical.
"""

import warp as wp

from ._funcs import (
    _bf_factor,
    _rot_q,
    _ssb_apply,
    _ssb_defocus,
    _ssb_general,
)

# No FMA contraction: nvcc contracts the same expression differently in
# different kernels (a loop body, a second call of a helper), which breaks
# the bit identity between the spellings. With ``fuse_fp`` off every kernel
# rounds each operation the same way.
wp.set_module_options({"fuse_fp": False})


@wp.kernel
def _direct_ptychography_forward(
    G: wp.array(dtype=wp.vec2, ndim=3),  # pyright: ignore[reportInvalidTypeForm]
    qx: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    qy: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    kx: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    ky: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    aberrations: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    sin_rot: wp.float32,
    cos_rot: wp.float32,
    semiconvergence_angle: wp.float32,
    eps: wp.float32,
    wavelength: wp.float32,
    G_out: wp.array(dtype=wp.vec2, ndim=3),  # pyright: ignore[reportInvalidTypeForm]
) -> None:
    """Phase-correct a [Qy, Qx, K] chunk of G."""
    iqy, iqx, ik = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    qr = _rot_q(qy[iqy], qx[iqx], sin_rot, cos_rot)
    kyv = ky[ik]
    kxv = kx[ik]
    A0 = _bf_factor(kyv, kxv, aberrations, semiconvergence_angle, wavelength)
    t = _ssb_general(kyv, kxv, qr, A0, aberrations, semiconvergence_angle, wavelength)
    G_out[iqy, iqx, ik] = _ssb_apply(G[iqy, iqx, ik], t)


@wp.kernel(enable_backward=False)
def _direct_ptychography_build_A(
    kx: wp.array(ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    ky: wp.array(ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    aberrations: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    semiconvergence_angle: wp.float32,
    wavelength: wp.float32,
    A_out: wp.array(dtype=wp.vec2, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
) -> None:
    """Build the per-pixel bright-field factor."""
    k = wp.tid()
    A_out[k] = _bf_factor(
        wp.float32(ky[k]),
        wp.float32(kx[k]),
        aberrations,
        semiconvergence_angle,
        wavelength,
    )


@wp.kernel(enable_backward=False)
def _direct_ptychography_build_A_planes(
    kx: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    ky: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    aberrations: wp.array(dtype=wp.float32, ndim=2),  # pyright: ignore[reportInvalidTypeForm]
    semiconvergence_angle: wp.float32,
    wavelength: wp.float32,
    A_out: wp.array(dtype=wp.vec2, ndim=2),  # pyright: ignore[reportInvalidTypeForm]
) -> None:
    """Build the bright-field factor for a stack of planes."""
    p, k = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    A_out[p, k] = _bf_factor(
        ky[k], kx[k], aberrations[p], semiconvergence_angle, wavelength
    )


@wp.kernel(enable_backward=False)
def _direct_ptychography_forward_precomputed_A(
    G: wp.array(dtype=wp.vec2, ndim=3),  # pyright: ignore[reportInvalidTypeForm]
    qx: wp.array(ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    qy: wp.array(ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    kx: wp.array(ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    ky: wp.array(ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    A_all: wp.array(dtype=wp.vec2, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    aberrations: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    sin_rot: wp.float32,
    cos_rot: wp.float32,
    semiconvergence_angle: wp.float32,
    eps: wp.float32,
    wavelength: wp.float32,
    G_out: wp.array(dtype=wp.vec2, ndim=3),  # pyright: ignore[reportInvalidTypeForm]
) -> None:
    """Phase-correct a [Qy, Qx, K] chunk with a given bright-field factor."""
    iqy, iqx, ik = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    gin = G[iqy, iqx, ik]
    qr = _rot_q(wp.float32(qy[iqy]), wp.float32(qx[iqx]), sin_rot, cos_rot)
    t = _ssb_general(
        wp.float32(ky[ik]),
        wp.float32(kx[ik]),
        qr,
        A_all[ik],
        aberrations,
        semiconvergence_angle,
        wavelength,
    )
    G_out[iqy, iqx, ik] = _ssb_apply(gin, t)


@wp.kernel(enable_backward=False)
def _direct_ptychography_chi_is_zero(
    aberrations: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    flag_out: wp.array(dtype=wp.int32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
) -> None:
    """Flag an aberration vector whose phase is identically zero."""
    flag = wp.int32(1)
    for i in range(aberrations.shape[0]):
        c = aberrations[i]
        if not (c == 0.0):
            flag = 0  # non-zero or NaN
        elif wp.float32(1.0) / c < 0.0:
            flag = 0  # -0.0: 1 / -0.0 is -inf
    flag_out[0] = flag


@wp.kernel(enable_backward=False)
def _direct_ptychography_forward_precomputed_A_kouter(
    G: wp.array(dtype=wp.vec2, ndim=3),  # pyright: ignore[reportInvalidTypeForm]
    qx: wp.array(ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    qy: wp.array(ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    kx: wp.array(ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    ky: wp.array(ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    A_all: wp.array(dtype=wp.vec2, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    aberrations: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    sin_rot: wp.float32,
    cos_rot: wp.float32,
    semiconvergence_angle: wp.float32,
    eps: wp.float32,
    wavelength: wp.float32,
    chi_is_zero: wp.array(dtype=wp.int32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    G_out: wp.array(dtype=wp.vec2, ndim=3),  # pyright: ignore[reportInvalidTypeForm]
) -> None:
    """Phase-correct a [K, Qy, Qx] chunk with a given bright-field factor."""
    ik, iqy, iqx = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    gin = G[ik, iqy % G.shape[1], iqx % G.shape[2]]
    qr = _rot_q(wp.float32(qy[iqy]), wp.float32(qx[iqx]), sin_rot, cos_rot)
    t = _ssb_general(
        wp.float32(ky[ik]),
        wp.float32(kx[ik]),
        qr,
        A_all[ik],
        aberrations,
        semiconvergence_angle,
        wavelength,
    )
    G_out[ik, iqy, iqx] = _ssb_apply(gin, t)


@wp.kernel(enable_backward=False)
def _direct_ptychography_forward_precomputed_A_kouter_defocus_only(
    G: wp.array(dtype=wp.vec2, ndim=3),  # pyright: ignore[reportInvalidTypeForm]
    qx: wp.array(ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    qy: wp.array(ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    kx: wp.array(ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    ky: wp.array(ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    A_all: wp.array(dtype=wp.vec2, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    aberrations: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    sin_rot: wp.float32,
    cos_rot: wp.float32,
    semiconvergence_angle: wp.float32,
    eps: wp.float32,
    wavelength: wp.float32,
    chi_is_zero: wp.array(dtype=wp.int32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    G_out: wp.array(dtype=wp.vec2, ndim=3),  # pyright: ignore[reportInvalidTypeForm]
) -> None:
    """Defocus-only form of the [K, Qy, Qx] correction."""
    ik, iqy, iqx = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    gin = G[ik, iqy % G.shape[1], iqx % G.shape[2]]
    qr = _rot_q(wp.float32(qy[iqy]), wp.float32(qx[iqx]), sin_rot, cos_rot)
    t = _ssb_defocus(
        wp.float32(ky[ik]),
        wp.float32(kx[ik]),
        qr,
        A_all[ik],
        aberrations,
        semiconvergence_angle,
        wavelength,
    )
    G_out[ik, iqy, iqx] = _ssb_apply(gin, t)
