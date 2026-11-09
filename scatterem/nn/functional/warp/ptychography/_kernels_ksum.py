"""Correct-and-sum kernels over the bright-field axis.

Each thread owns one ``(Qy, Qx)`` element and adds the corrected values of
the contributing pairs in ascending ``ik`` order, in float32. A pair that
does not contribute adds nothing, so a NaN there does not reach the sum.
The mask byte layout is bit 0 = H(K + Q), bit 1 = H(K - Q), bit 2 = H(K).
"""

import warp as wp

from scatterem.utils.warp.misc import probe_cone_pass

from ._funcs import (
    CASE_NONE,
    _chi_zero_add,
    _mask_bits,
    _rot_q,
    _ssb_apply,
    _ssb_general,
    _ssb_masked,
    _ssb_masked_defocus,
)

# No FMA contraction: nvcc contracts the same expression differently in
# different kernels (a loop body, a second call of a helper), which breaks
# the bit identity between the spellings. With ``fuse_fp`` off every kernel
# rounds each operation the same way.
wp.set_module_options({"fuse_fp": False})


@wp.kernel(enable_backward=False)
def _direct_ptychography_forward_ksum(
    G: wp.array(dtype=wp.vec2, ndim=3),  # pyright: ignore[reportInvalidTypeForm]
    qx: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    qy: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    kx: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    ky: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    A_all: wp.array(dtype=wp.vec2, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    aberrations: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    sin_rot: wp.float32,
    cos_rot: wp.float32,
    semiconvergence_angle: wp.float32,
    eps: wp.float32,
    wavelength: wp.float32,
    S_out: wp.array(dtype=wp.vec2, ndim=2),  # pyright: ignore[reportInvalidTypeForm]
) -> None:
    """Phase-correct a chunk and sum it over K."""
    iqy, iqx = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    qr = _rot_q(qy[iqy], qx[iqx], sin_rot, cos_rot)
    acc = wp.vec2(0.0, 0.0)
    for ik in range(G.shape[2]):
        t = _ssb_general(
            ky[ik], kx[ik], qr, A_all[ik], aberrations,
            semiconvergence_angle, wavelength,
        )  # fmt: skip
        if t[2] != CASE_NONE:
            acc = acc + _ssb_apply(G[iqy, iqx, ik], t)
    S_out[iqy, iqx] = acc


@wp.kernel(enable_backward=False)
def _direct_ptychography_forward_ksum_chi_zero(
    G: wp.array(dtype=wp.vec2, ndim=3),  # pyright: ignore[reportInvalidTypeForm]
    qx: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    qy: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    kx: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    ky: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    A_all: wp.array(dtype=wp.vec2, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    aberrations: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    sin_rot: wp.float32,
    cos_rot: wp.float32,
    semiconvergence_angle: wp.float32,
    eps: wp.float32,
    wavelength: wp.float32,
    S_out: wp.array(dtype=wp.vec2, ndim=2),  # pyright: ignore[reportInvalidTypeForm]
) -> None:
    """Correct-and-sum for an all-zero aberration vector."""
    iqy, iqx = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    qr = _rot_q(qy[iqy], qx[iqx], sin_rot, cos_rot)
    qz = wp.int32(0)
    if qr[0] == 0.0 and qr[1] == 0.0:
        qz = 1
    acc = wp.vec2(0.0, 0.0)
    for ik in range(G.shape[2]):
        kyv = ky[ik]
        kxv = kx[ik]
        h0 = probe_cone_pass(wavelength, semiconvergence_angle, kyv, kxv)
        hp = probe_cone_pass(
            wavelength, semiconvergence_angle, kyv + qr[0], kxv + qr[1]
        )
        hm = probe_cone_pass(
            wavelength, semiconvergence_angle, kyv - qr[0], kxv - qr[1]
        )
        if h0 != 0.0 and qz == 0 and hp + hm == 1.0:
            acc = acc + _chi_zero_add(G[iqy, iqx, ik], h0, hp, hm, qz)  # pyright: ignore[reportArgumentType]
    S_out[iqy, iqx] = acc


@wp.kernel(enable_backward=False)
def _direct_ptychography_aperture_mask_planes(
    qx: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    qy: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    kx: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    ky: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    sin_rot: wp.float32,
    cos_rot: wp.float32,
    semiconvergence_angle: wp.float32,
    wavelength: wp.float32,
    mask: wp.array(dtype=wp.uint8, ndim=3),  # pyright: ignore[reportInvalidTypeForm]
) -> None:
    """Pack the two shifted-aperture tests per (Q, K) into bits."""
    iqy, iqx, ik = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    qr = _rot_q(qy[iqy], qx[iqx], sin_rot, cos_rot)
    kyv = ky[ik]
    kxv = kx[ik]
    h0 = probe_cone_pass(wavelength, semiconvergence_angle, kyv, kxv)
    hp = probe_cone_pass(wavelength, semiconvergence_angle, kyv + qr[0], kxv + qr[1])
    hm = probe_cone_pass(wavelength, semiconvergence_angle, kyv - qr[0], kxv - qr[1])
    mask[iqy, iqx, ik] = _mask_bits(h0, hp, hm)


@wp.kernel(enable_backward=False)
def _direct_ptychography_forward_ksum_planes(
    G: wp.array(dtype=wp.vec2, ndim=3),  # pyright: ignore[reportInvalidTypeForm]
    qx: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    qy: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    kx: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    ky: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    A_all: wp.array(dtype=wp.vec2, ndim=2),  # pyright: ignore[reportInvalidTypeForm]
    aberrations: wp.array(dtype=wp.float32, ndim=2),  # pyright: ignore[reportInvalidTypeForm]
    sin_rot: wp.float32,
    cos_rot: wp.float32,
    semiconvergence_angle: wp.float32,
    eps: wp.float32,
    wavelength: wp.float32,
    aperture_mask: wp.array(dtype=wp.uint8, ndim=3),  # pyright: ignore[reportInvalidTypeForm]
    S_out: wp.array(dtype=wp.vec2, ndim=3),  # pyright: ignore[reportInvalidTypeForm]
) -> None:
    """Correct-and-sum for a stack of aberration vectors."""
    p, iqy, iqx = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    qr = _rot_q(qy[iqy], qx[iqx], sin_rot, cos_rot)
    ab = aberrations[p]
    acc = wp.vec2(0.0, 0.0)
    for ik in range(G.shape[2]):
        bits = int(aperture_mask[iqy, iqx, ik])
        t = _ssb_masked(ky[ik], kx[ik], qr, A_all[ik, p], ab, wavelength, bits)
        if t[2] != CASE_NONE:
            acc = acc + _ssb_apply(G[iqy, iqx, ik], t)
    S_out[p, iqy, iqx] = acc


@wp.kernel(enable_backward=False)
def _direct_ptychography_forward_ksum_planes_defocus_only(
    G: wp.array(dtype=wp.vec2, ndim=3),  # pyright: ignore[reportInvalidTypeForm]
    qx: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    qy: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    kx: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    ky: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    A_all: wp.array(dtype=wp.vec2, ndim=2),  # pyright: ignore[reportInvalidTypeForm]
    aberrations: wp.array(dtype=wp.float32, ndim=2),  # pyright: ignore[reportInvalidTypeForm]
    sin_rot: wp.float32,
    cos_rot: wp.float32,
    semiconvergence_angle: wp.float32,
    eps: wp.float32,
    wavelength: wp.float32,
    aperture_mask: wp.array(dtype=wp.uint8, ndim=3),  # pyright: ignore[reportInvalidTypeForm]
    S_out: wp.array(dtype=wp.vec2, ndim=3),  # pyright: ignore[reportInvalidTypeForm]
) -> None:
    """Defocus-only form of the plane-stack correct-and-sum."""
    p, iqy, iqx = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    qr = _rot_q(qy[iqy], qx[iqx], sin_rot, cos_rot)
    ab = aberrations[p]
    acc = wp.vec2(0.0, 0.0)
    for ik in range(G.shape[2]):
        bits = int(aperture_mask[iqy, iqx, ik])
        t = _ssb_masked_defocus(ky[ik], kx[ik], qr, A_all[ik, p], ab, wavelength, bits)
        if t[2] != CASE_NONE:
            acc = acc + _ssb_apply(G[iqy, iqx, ik], t)
    S_out[p, iqy, iqx] = acc
