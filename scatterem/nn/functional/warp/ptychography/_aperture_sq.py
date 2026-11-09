"""Squared-radius form of the probe-aperture predicate and its threshold.

``probe_cone_pass()`` is a monotone function of the float32 sum of squares
``s = qx * qx + qy * qy`` (see ``scatterem.utils.warp.misc``). So there is one
smallest float32 ``S`` at which it is 0.0, and ``s < S`` is the same test as
``aperture() == 1`` for every float32 ``s``. ``_aperture_sq_threshold`` finds
``S`` by bisection on the device that later uses it. Both predicates form ``s``
with the same helper ``_q2``, so they round ``s`` the same way.
"""

import warp as wp

from scatterem.utils.warp.misc import _aperture_of_s, _q2

# No FMA contraction: nvcc contracts the same expression differently in
# different kernels (a loop body, a second call of a helper), which breaks
# the bit identity between the spellings. With ``fuse_fp`` off every kernel
# rounds each operation the same way.
wp.set_module_options({"fuse_fp": False})

#: Upper bound of bisection steps. A float bisection over ``[0, hi0]`` ends
#: after at most the exponent range plus the 24 mantissa bits (about 280).
_MAX_BISECT = wp.constant(400)


@wp.kernel(enable_backward=False)
def _aperture_sq_threshold(
    wavelength: wp.float32,
    semiconvergence_angle_max: wp.float32,
    hi0: wp.float32,
    out: wp.array(dtype=wp.float32, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
) -> None:
    """Find the squared-radius threshold of the aperture predicate.

    Writes the smallest float32 ``S`` with ``probe_cone_pass`` is 0 at ``S``, or -1.0 when
    the bracket ``[0, hi0]`` is not valid or the bisection does not close.
    """
    lo = wp.float32(0.0)
    hi = hi0
    result = wp.float32(-1.0)
    alpha = semiconvergence_angle_max
    ok_lo = _aperture_of_s(lo, wavelength, alpha) == wp.float32(1.0)
    ok_hi = _aperture_of_s(hi, wavelength, alpha) == wp.float32(0.0)
    if ok_lo and ok_hi:
        for _it in range(_MAX_BISECT):
            mid = lo + wp.float32(0.5) * (hi - lo)
            if mid == lo or mid == hi:
                # No float32 lies strictly between lo and hi.
                result = hi
                break
            if _aperture_of_s(mid, wavelength, alpha) == wp.float32(1.0):
                lo = mid
            else:
                hi = mid
    out[0] = result


@wp.func
def _aperture_sq(
    qy: wp.float32,
    qx: wp.float32,
    s_thresh: wp.float32,
) -> wp.float32:
    """Aperture predicate from a squared-radius threshold.

    Returns 1.0 if ``qx * qx + qy * qy < s_thresh``, else 0.0.
    """
    out = wp.float32(0.0)
    if _q2(qy, qx) < s_thresh:
        out = wp.float32(1.0)
    return out
