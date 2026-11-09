"""Squared-radius-aperture forms of the k-outer correction and the mask.

``_aperture_sq(q, s_thresh)`` equals ``probe_cone_pass`` at ``q`` bit for bit when
``s_thresh`` comes from ``_aperture_sq_threshold``, so these kernels equal
their ``probe_cone_pass`` forms bit for bit.

Half-spectrum store: with ``C_k`` the full ``[Hq, W]`` corrected plane and
``m(q)`` the index mirror ``((Hq - iqy) % Hq, (W - iqx) % W)``, the spectrum
``H_k(q) = (C_k(q) - conj(C_k(m(q)))) / (2 i)`` is Hermitian, and its inverse
DFT is ``Im(ifft2(C_k))`` (Bracewell, Hermitian symmetry of real signals).
"""

import warp as wp

from ._aperture_sq import _aperture_sq
from ._funcs import _mask_bits, _rot_q, _ssb_apply, _ssb_sq

# No FMA contraction: nvcc contracts the same expression differently in
# different kernels (a loop body, a second call of a helper), which breaks
# the bit identity between the spellings. With ``fuse_fp`` off every kernel
# rounds each operation the same way.
wp.set_module_options({"fuse_fp": False})


@wp.func
def _kouter_sq_value(
    G: wp.array(dtype=wp.vec2, ndim=3),  # pyright: ignore[reportInvalidTypeForm]
    qx: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    qy: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    kx: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    ky: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    A_all: wp.array(dtype=wp.vec2, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    aberrations: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    sin_rot: wp.float32,
    cos_rot: wp.float32,
    s_thresh: wp.float32,
    wavelength: wp.float32,
    ik: int,
    iqy: int,
    iqx: int,
) -> wp.vec2:
    """Corrected value of the ``_sq`` k-outer kernel at ``(ik, iqy, iqx)``."""
    gin = G[ik, iqy % G.shape[1], iqx % G.shape[2]]
    qr = _rot_q(qy[iqy], qx[iqx], sin_rot, cos_rot)
    t = _ssb_sq(ky[ik], kx[ik], qr, A_all[ik], aberrations, s_thresh, wavelength)
    return _ssb_apply(gin, t)


@wp.func
def _kouter_sq_hspec(
    G: wp.array(dtype=wp.vec2, ndim=3),  # pyright: ignore[reportInvalidTypeForm]
    qx: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    qy: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    kx: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    ky: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    A_all: wp.array(dtype=wp.vec2, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    aberrations: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    sin_rot: wp.float32,
    cos_rot: wp.float32,
    s_thresh: wp.float32,
    wavelength: wp.float32,
    ik: int,
    iqy: int,
    iqx: int,
    hq: int,
    w: int,
) -> wp.vec2:
    """Hermitian spectrum ``H_k`` at ``(iqy, iqx)`` on the ``(hq, w)`` grid."""
    c = _kouter_sq_value(
        G, qx, qy, kx, ky, A_all, aberrations, sin_rot, cos_rot, s_thresh,
        wavelength, ik, iqy, iqx,
    )  # fmt: skip
    my = (hq - iqy) % hq
    mx = (w - iqx) % w
    cm = _kouter_sq_value(
        G, qx, qy, kx, ky, A_all, aberrations, sin_rot, cos_rot, s_thresh,
        wavelength, ik, my, mx,
    )  # fmt: skip
    # D = C(q) - conj(C(m(q))); H = D / (2 i) = (Im D / 2, -Re D / 2).
    # -Re D is formed as cm - c, so that an exact zero is +0.0.
    return wp.vec2((c[1] + cm[1]) * 0.5, (cm[0] - c[0]) * 0.5)  # pyright: ignore[reportOperatorIssue]


@wp.kernel(enable_backward=False)
def _direct_ptychography_forward_precomputed_A_kouter_sq(
    G: wp.array(dtype=wp.vec2, ndim=3),  # pyright: ignore[reportInvalidTypeForm]
    qx: wp.array(ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    qy: wp.array(ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    kx: wp.array(ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    ky: wp.array(ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    A_all: wp.array(dtype=wp.vec2, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    aberrations: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    sin_rot: wp.float32,
    cos_rot: wp.float32,
    s_thresh: wp.float32,
    eps: wp.float32,
    wavelength: wp.float32,
    chi_is_zero: wp.array(dtype=wp.int32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    G_out: wp.array(dtype=wp.vec2, ndim=3),  # pyright: ignore[reportInvalidTypeForm]
) -> None:
    """Squared-radius form of the [K, Qy, Qx] correction."""
    ik, iqy, iqx = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    gin = G[ik, iqy % G.shape[1], iqx % G.shape[2]]
    qr = _rot_q(wp.float32(qy[iqy]), wp.float32(qx[iqx]), sin_rot, cos_rot)
    t = _ssb_sq(
        wp.float32(ky[ik]),
        wp.float32(kx[ik]),
        qr,
        A_all[ik],
        aberrations,
        s_thresh,
        wavelength,
    )
    G_out[ik, iqy, iqx] = _ssb_apply(gin, t)


@wp.kernel(enable_backward=False)
def _direct_ptychography_forward_precomputed_A_kouter_sq_half(
    G: wp.array(dtype=wp.vec2, ndim=3),  # pyright: ignore[reportInvalidTypeForm]
    qx: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    qy: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    kx: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    ky: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    A_all: wp.array(dtype=wp.vec2, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    aberrations: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    sin_rot: wp.float32,
    cos_rot: wp.float32,
    s_thresh: wp.float32,
    eps: wp.float32,
    wavelength: wp.float32,
    chi_is_zero: wp.array(dtype=wp.int32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    G_out: wp.array(dtype=wp.vec2, ndim=3),  # pyright: ignore[reportInvalidTypeForm]
) -> None:
    """Half-spectrum store of the [K, Qy, Qx] correction."""
    ik, iqy, iqx = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    hq = G_out.shape[1]
    w = 2 * (G_out.shape[2] - 1)
    G_out[ik, iqy, iqx] = _kouter_sq_hspec(
        G, qx, qy, kx, ky, A_all, aberrations, sin_rot, cos_rot, s_thresh,
        wavelength, ik, iqy, iqx, hq, w,
    )  # fmt: skip


MASK_IK_BLOCK = wp.constant(8)


@wp.kernel(enable_backward=False)
def _direct_ptychography_aperture_mask_planes_sq(
    qx: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    qy: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    kx: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    ky: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    sin_rot: wp.float32,
    cos_rot: wp.float32,
    s_thresh: wp.float32,
    mask: wp.array(dtype=wp.uint8, ndim=3),  # pyright: ignore[reportInvalidTypeForm]
) -> None:
    """Squared-radius form of the aperture bit mask."""
    iqy, iqx, b = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    qr = _rot_q(qy[iqy], qx[iqx], sin_rot, cos_rot)
    nk = mask.shape[2]
    for i in range(MASK_IK_BLOCK):
        ik = b * MASK_IK_BLOCK + i
        if ik < nk:
            kyv = ky[ik]
            kxv = kx[ik]
            h0 = _aperture_sq(kyv, kxv, s_thresh)
            hp = _aperture_sq(kyv + qr[0], kxv + qr[1], s_thresh)
            hm = _aperture_sq(kyv - qr[0], kxv - qr[1], s_thresh)
            mask[iqy, iqx, ik] = _mask_bits(h0, hp, hm)
