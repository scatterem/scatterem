"""Phase-contrast transfer function kernel.

Thread ``(ik, iqy, iqx)`` adds ``|Gamma(Q, K)|`` (see ``_funcs``) to
``pctf[iqy, iqx]``: 0 without a double overlap and at ``Q = 0``, 1 for one
side band and ``2 |sin(s - chi(K))|`` for both. For ``-Q`` the two side bands
swap and ``s`` is the same float, so each term is exactly even in ``Q``.
"""

import warp as wp

from ._funcs import _bf_factor, _rot_q, _ssb_general

# No FMA contraction: nvcc contracts the same expression differently in
# different kernels (a loop body, a second call of a helper), which breaks
# the bit identity between the spellings. With ``fuse_fp`` off every kernel
# rounds each operation the same way.
wp.set_module_options({"fuse_fp": False})


@wp.kernel
def _phase_contrast_transfer_function_forward(
    G: wp.array(dtype=wp.vec2, ndim=3),  # pyright: ignore[reportInvalidTypeForm]
    qx: wp.array(ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    qy: wp.array(ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    kx: wp.array(ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    ky: wp.array(ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    aberrations: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    sin_rot: wp.float32,
    cos_rot: wp.float32,
    semiconvergence_angle: wp.float32,
    wavelength: wp.float32,
    pctf: wp.array(dtype=wp.float32, ndim=2),  # pyright: ignore[reportInvalidTypeForm]
) -> None:
    """Accumulate the double-overlap transfer magnitude per Q."""
    ik, iqy, iqx = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    qr = _rot_q(wp.float32(qy[iqy]), wp.float32(qx[iqx]), sin_rot, cos_rot)
    kyv = wp.float32(ky[ik])
    kxv = wp.float32(kx[ik])
    A0 = _bf_factor(kyv, kxv, aberrations, semiconvergence_angle, wavelength)
    t = _ssb_general(kyv, kxv, qr, A0, aberrations, semiconvergence_angle, wavelength)
    if t[3] != 0.0:
        wp.atomic_add(pctf, iqy, iqx, t[3])
