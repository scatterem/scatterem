"""Two-planes-per-word store of the k-outer correction.

``G_out[j] = H_{2j} + i H_{2j+1}`` on the full grid, with ``H_k`` the
Hermitian spectrum of ``_kernels_sq``. Both ``ifft2(H_k)`` are real, so
``ifft2(G_out[j])`` holds image ``2j`` in its real part and image ``2j + 1``
in its imaginary part. For odd ``Nk`` the last plane is ``H_{Nk-1}`` alone.
"""

import warp as wp

from ._kernels_sq import _kouter_sq_hspec

# No FMA contraction: nvcc contracts the same expression differently in
# different kernels (a loop body, a second call of a helper), which breaks
# the bit identity between the spellings. With ``fuse_fp`` off every kernel
# rounds each operation the same way.
wp.set_module_options({"fuse_fp": False})


@wp.kernel(enable_backward=False)
def _direct_ptychography_forward_precomputed_A_kouter_sq_packed(
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
    """Pack two half-spectrum planes into one complex plane."""
    j, iqy, iqx = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    hq = G_out.shape[1]
    w = G_out.shape[2]
    k0 = 2 * j
    k1 = k0 + 1
    h0 = _kouter_sq_hspec(
        G, qx, qy, kx, ky, A_all, aberrations, sin_rot, cos_rot, s_thresh,
        wavelength, k0, iqy, iqx, hq, w,
    )  # fmt: skip
    out = h0
    if k1 < G.shape[0]:
        h1 = _kouter_sq_hspec(
            G, qx, qy, kx, ky, A_all, aberrations, sin_rot, cos_rot, s_thresh,
            wavelength, k1, iqy, iqx, hq, w,
        )  # fmt: skip
        out = wp.vec2(h0[0] - h1[1], h0[1] + h1[0])  # pyright: ignore[reportOperatorIssue]
    G_out[j, iqy, iqx] = out
