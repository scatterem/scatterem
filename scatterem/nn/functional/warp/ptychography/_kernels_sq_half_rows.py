"""Row-loop launch of the half-spectrum k-outer store.

Thread ``(ik, iqy, b)`` writes columns ``b * KOUTER_HALF_ROWS`` up to
``min((b + 1) * KOUTER_HALF_ROWS, Wh) - 1`` of row ``iqy``, each exactly as
``_direct_ptychography_forward_precomputed_A_kouter_sq_half`` does.
"""

import warp as wp

from ._kernels_sq import _kouter_sq_hspec

# No FMA contraction: nvcc contracts the same expression differently in
# different kernels (a loop body, a second call of a helper), which breaks
# the bit identity between the spellings. With ``fuse_fp`` off every kernel
# rounds each operation the same way.
wp.set_module_options({"fuse_fp": False})

KOUTER_HALF_ROWS = wp.constant(8)


@wp.kernel(enable_backward=False)
def _direct_ptychography_forward_precomputed_A_kouter_sq_half_rows(
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
    """Half-spectrum store with each thread on several columns."""
    ik, iqy, b = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    hq = G_out.shape[1]
    wh = G_out.shape[2]
    w = 2 * (wh - 1)
    for i in range(KOUTER_HALF_ROWS):
        iqx = b * KOUTER_HALF_ROWS + i
        if iqx < wh:
            G_out[ik, iqy, iqx] = _kouter_sq_hspec(
                G, qx, qy, kx, ky, A_all, aberrations, sin_rot, cos_rot,
                s_thresh, wavelength, ik, iqy, iqx, hq, w,
            )  # fmt: skip
