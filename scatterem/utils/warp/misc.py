"""The hard probe-forming aperture as a Warp device function.

The aperture passes a transverse wave vector ``q`` when its scattering angle
``theta`` is strictly below the semiconvergence angle ``alpha``. Elastic
scattering gives ``sin(theta) = wavelength * |q|`` (Kirkland, *Advanced
Computing in Electron Microscopy*, 3rd ed., 2020).

The predicate compares ``wavelength * |q|`` with ``sin(alpha)``. For angles in
``[0, pi/2]`` the sine is strictly increasing, so this is the same test as
``theta < alpha``. It needs no ``asin``, and it is exactly monotone in the
float32 sum of squares ``qx * qx + qy * qy``. The squared-radius predicate in
``nn.functional.warp.ptychography._aperture_sq`` depends on that.
"""

import warp as wp

__all__ = [
    "probe_cone_pass",
]


@wp.func
def _q2(qy: wp.float32, qx: wp.float32) -> wp.float32:
    """Float32 sum of squares ``qx * qx + qy * qy``, the one spelling all use."""
    return qx * qx + qy * qy  # pyright: ignore[reportOperatorIssue]


@wp.func
def _aperture_of_s(
    s: wp.float32,
    wavelength: wp.float32,
    cone_angle: wp.float32,
) -> wp.float32:
    """Aperture predicate as a function of the squared radius ``s``.

    Returns 1.0 when ``wavelength * sqrt(s) < sin(alpha)``, else 0.0. When
    ``alpha`` is above ``pi/2`` every real angle passes, so the test becomes
    ``wavelength * sqrt(s) <= 1``. NaN in any input gives 0.0.
    """
    x = wavelength * wp.sqrt(s)
    out = wp.float32(0.0)
    if cone_angle > wp.float32(wp.HALF_PI):
        if x <= wp.float32(1.0):
            out = wp.float32(1.0)
    else:
        if x < wp.sin(cone_angle):
            out = wp.float32(1.0)
    return out


@wp.func
def probe_cone_pass(
    wavelength: wp.float32,
    cone_angle: wp.float32,
    qy: wp.float32,
    qx: wp.float32,
) -> wp.float32:
    """Pass value of the hard probe-forming aperture: 1.0 inside the cone.

    Args:
        wavelength: electron wavelength in Angstrom.
        cone_angle: probe-forming semi-angle (semiconvergence angle) in rad.
        qy: transverse wave vector, y component, in 1/Angstrom.
        qx: transverse wave vector, x component, in 1/Angstrom.

    Returns:
        1.0 if the scattering angle of ``(qy, qx)`` is strictly below
        ``cone_angle``, else 0.0. ``wavelength * |q| > 1`` (no real angle)
        gives 0.0.
    """
    return _aperture_of_s(_q2(qy, qx), wavelength, cone_angle)
