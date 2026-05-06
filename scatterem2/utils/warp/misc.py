import warp as wp


@wp.func
def aperture(
    qy: wp.float32,
    qx: wp.float32,
    wavelength: wp.float32,
    semiconvergence_angle_max: wp.float32,
) -> wp.float32:
    """Aperture function"""
    qx2 = qx * qx
    qy2 = qy * qy
    q = wp.sqrt(qx2 + qy2)
    ktheta = wp.asin(q * wavelength)
    return wp.where(ktheta < semiconvergence_angle_max, 1.0, 0.0)
