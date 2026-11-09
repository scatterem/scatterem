"""Correct-and-sum kernels split over blocks of the bright-field axis.

Tile ``it`` of ``NT = S_part.shape[0]`` sums ``ik`` in
``[it * B, min((it + 1) * B, Nk))`` with ``B = ceil(Nk / NT)``. The ranges
cover every ``ik`` once; an empty range writes (0, 0). The aberrations are all
``+0.0``, so a pair with one side band adds ``+G`` (minus) or ``-G`` (plus) and
every other pair adds nothing (see ``_funcs._chi_zero_add``).
"""

import warp as wp

from scatterem.utils.warp.misc import probe_cone_pass

from ._aperture_sq import _aperture_sq
from ._funcs import _chi_zero_add, _rot_q

# No FMA contraction: nvcc contracts the same expression differently in
# different kernels (a loop body, a second call of a helper), which breaks
# the bit identity between the spellings. With ``fuse_fp`` off every kernel
# rounds each operation the same way.
wp.set_module_options({"fuse_fp": False})


@wp.kernel(enable_backward=False)
def _direct_ptychography_forward_ksum_chi_zero_tiled(
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
    S_part: wp.array(dtype=wp.vec2, ndim=3),  # pyright: ignore[reportInvalidTypeForm]
) -> None:
    """Zero-aberration correct-and-sum over one block of K."""
    it, iqy, iqx = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    nk = G.shape[2]
    nt = S_part.shape[0]
    blk = (nk + nt - 1) // nt
    k0 = it * blk
    k1 = wp.min(k0 + blk, nk)
    qr = _rot_q(qy[iqy], qx[iqx], sin_rot, cos_rot)
    qz = wp.int32(0)
    if qr[0] == 0.0 and qr[1] == 0.0:
        qz = 1
    acc = wp.vec2(0.0, 0.0)
    for ik in range(k0, k1):  # pyright: ignore[reportArgumentType]
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
    S_part[it, iqy, iqx] = acc


@wp.kernel(enable_backward=False)
def _direct_ptychography_forward_ksum_chi_zero_tiled_sq(
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
    S_part: wp.array(dtype=wp.vec2, ndim=3),  # pyright: ignore[reportInvalidTypeForm]
) -> None:
    """Squared-radius form of the blocked zero-aberration correct-and-sum."""
    it, iqy, iqx = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    nk = G.shape[2]
    nt = S_part.shape[0]
    blk = (nk + nt - 1) // nt
    k0 = it * blk
    k1 = wp.min(k0 + blk, nk)
    qr = _rot_q(qy[iqy], qx[iqx], sin_rot, cos_rot)
    qz = wp.int32(0)
    if qr[0] == 0.0 and qr[1] == 0.0:
        qz = 1
    acc = wp.vec2(0.0, 0.0)
    for ik in range(k0, k1):  # pyright: ignore[reportArgumentType]
        kyv = ky[ik]
        kxv = kx[ik]
        h0 = _aperture_sq(kyv, kxv, s_thresh)
        hp = _aperture_sq(kyv + qr[0], kxv + qr[1], s_thresh)
        hm = _aperture_sq(kyv - qr[0], kxv - qr[1], s_thresh)
        if h0 != 0.0 and qz == 0 and hp + hm == 1.0:
            acc = acc + _chi_zero_add(G[iqy, iqx, ik], h0, hp, hm, qz)  # pyright: ignore[reportArgumentType]
    S_part[it, iqy, iqx] = acc
