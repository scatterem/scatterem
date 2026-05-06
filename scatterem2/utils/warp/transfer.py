import warp as wp


@wp.func
def aberration_function_polar(
    alpha: wp.float32,
    phi: wp.float32,
    wavelength: wp.float32,
    aberrations: wp.array(dtype=wp.float32),
) -> wp.float32:
    """
    Zernike polynomials in the polar coordinate system
    """
    chi = 0.0

    C10 = 0
    C12 = 1
    C21 = 2
    C23 = 3
    C30 = 4
    C32 = 5
    C34 = 6
    phi12 = 7
    phi21 = 8
    phi23 = 9
    phi32 = 10
    phi34 = 11

    if aberrations[C10] != 0 or aberrations[C12] != 0 or aberrations[phi12] != 0:
        chi = chi + (
            0.5
            * alpha**2.0
            * (
                aberrations[C10]
                + aberrations[C12] * wp.cos(2 * (phi - aberrations[phi12]))
            )
        )

    if (
        aberrations[C21] != 0
        or aberrations[phi21] != 0
        or aberrations[C23] != 0
        or aberrations[phi23] != 0
    ):
        chi = chi + (
            0.3333333333333333
            * alpha**3.0
            * (
                aberrations[C21] * wp.cos(phi - aberrations[phi21])
                + aberrations[C23] * wp.cos(3 * (phi - aberrations[phi23]))
            )
        )

    if (
        aberrations[C30] != 0
        or aberrations[C32] != 0
        or aberrations[phi32] != 0
        or aberrations[C34] != 0
        or aberrations[phi34] != 0
    ):
        chi = chi + (
            0.25
            * alpha**4.0
            * (
                aberrations[C30]
                + aberrations[C32] * wp.cos(2 * (phi - aberrations[phi32]))
                + aberrations[C34] * wp.cos(4 * (phi - aberrations[phi34]))
            )
        )
    chi *= 2 * wp.pi / wavelength
    return chi


@wp.func
def aberration_function_cartesian(
    qy: wp.float32,
    qx: wp.float32,
    wavelength: wp.float32,
    aberrations: wp.array(dtype=wp.float32),
) -> wp.float32:
    """
    Zernike polynomials in the cartesian coordinate system
    """
    u = qx * wavelength
    v = qy * wavelength
    u2 = u * u
    u3 = u2 * u
    u4 = u3 * u

    v2 = v * v
    v3 = v2 * v
    v4 = v3 * v

    # r^2
    chi1 = 0.5 * aberrations[0] * (u2 + v2)

    # r^2 cos(2 phi) + r^2 sin(2 phi)
    chi2 = 0.5 * (aberrations[1] * (u2 - v2) + 2.0 * aberrations[2] * u * v)

    # r^3 cos(3phi) + r^3 sin(3 phi)
    chi3 = (1.0 / 3.0) * (
        aberrations[5] * (u3 - 3.0 * u * v2) + aberrations[6] * (3.0 * u2 * v - v3)
    )

    # r^3 cos(phi) + r^3 sin(phi)
    chi4 = (1.0 / 3.0) * (
        aberrations[3] * (u3 + u * v2) + aberrations[4] * (v3 + u2 * v)
    )

    # r^4
    chi5 = 0.25 * aberrations[7] * (u4 + v4 + 2.0 * u2 * v2)

    # r^4 cos(4 phi)
    chi6 = 0.25 * aberrations[10] * (u4 - 6.0 * u2 * v2 + v4)

    # r^4 sin(4 phi)
    chi7 = 0.25 * aberrations[11] * (4.0 * u3 * v - 4.0 * u * v3)

    # r^4 cos(2 phi)
    chi8 = 0.25 * aberrations[8] * (u4 - v4)

    # r^4 sin(2 phi)
    chi9 = 0.25 * aberrations[9] * (2.0 * u3 * v + 2.0 * u * v3)

    result = chi1 + chi2 + chi3 + chi4 + chi5 + chi6 + chi7 + chi8 + chi9

    return result * 2.0 * wp.pi / wavelength

