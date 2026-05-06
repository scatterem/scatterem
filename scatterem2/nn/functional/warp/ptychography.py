"""
Warp kernels for direct ptychography operations.
"""
import warp as wp
import torch

from scatterem2.utils.warp import (
    aberration_function_cartesian,
    aperture,
    cabs,
    cconj,
    cexp,
    cmul,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

@wp.func
def minus_i(z: wp.vec2) -> wp.vec2:
    """
    Multiplication of a complex number by -i.
    """
    # (-i)*(x+iy) =  y + i*(-x)
    return wp.vec2(z[1], -z[0])


@wp.func
def ip_real_conj(a: wp.vec2, b: wp.vec2) -> wp.float32:
    """
    Inner product of the real part of the conjugate of a and b.
    """
    # Re{ conj(a) * b } for (ax+i ay)(bx+i by) = ax*bx + ay*by
    return a[0] * b[0] + a[1] * b[1]


@wp.func
def dchi_cartesian_aberrations(
    qy: wp.float32, qx: wp.float32, wavelength: wp.float32, j: int
) -> wp.float32:
    """
    Derivative of the aberration array with respect to the qy and qx.
    Args:
        qy: wp.float32 - qy coordinate
        qx: wp.float32 - qx coordinate
        wavelength: wp.float32 - wavelength
        j: int - index of the aberration

    Returns:
        wp.float32 - derivative of the aberration array with respect to the qy and qx
    """

    u = qx * wavelength
    v = qy * wavelength
    u2 = u * u
    v2 = v * v
    u3 = u2 * u
    v3 = v2 * v
    u4 = u3 * u
    v4 = v3 * v
    base = wp.float32(0.0)

    if j == 0:
        base = 0.5 * (u2 + v2)
    elif j == 1:
        base = 0.5 * (u2 - v2)
    elif j == 2:
        base = u * v
    elif j == 3:
        base = (1.0 / 3.0) * (u3 + u * v2)
    elif j == 4:
        base = (1.0 / 3.0) * (v3 + u * u * v)
    elif j == 5:
        base = (1.0 / 3.0) * (u3 - 3.0 * u * v2)
    elif j == 6:
        base = (1.0 / 3.0) * (3.0 * u * u * v - v3)
    elif j == 7:
        base = 0.25 * (u4 + v4 + 2.0 * u2 * v2)
    elif j == 8:
        base = 0.25 * (u4 - v4)
    elif j == 9:
        base = 0.25 * (2.0 * u3 * v + 2.0 * u * v3)
    elif j == 10:
        base = 0.25 * (u4 - 6.0 * u2 * v2 + v4)
    elif j == 11:
        base = 0.25 * (4.0 * u3 * v - 4.0 * u * v3)

    return base * (2.0 * wp.pi / wavelength)


@wp.kernel
def _direct_ptychography_forward(
    G: wp.array(dtype=wp.vec2, ndim=3),
    Qx_all: wp.array(ndim=1),
    Qy_all: wp.array(ndim=1),
    Kx_all: wp.array(ndim=1),
    Ky_all: wp.array(ndim=1),
    aberrations: wp.array(dtype=wp.float32, ndim=1),
    sin_rot: wp.float32,
    cos_rot: wp.float32,
    semiconvergence_angle: wp.float32,
    eps: wp.float32,
    wavelength: wp.float32,
    G_out: wp.array(dtype=wp.vec2, ndim=3),
) -> None:
    """
    Forward kernel for the direct ptychography forward pass.

    Args:
        G: wp.array(dtype=wp.vec2, ndim=3) - [Qy,Qx,ik]
        Qx_all: wp.array(ndim=1) - [Qx]
        Qy_all: wp.array(ndim=1) - [Qy]
        Kx_all: wp.array(ndim=1) - [Kx]
        Ky_all: wp.array(ndim=1) - [Ky]
        aberrations: wp.array(dtype=wp.float32, ndim=1)
        sin_rot: wp.float32 - sin(rotation)
        cos_rot: wp.float32 - cos(rotation)
        semiconvergence_angle: wp.float32 - semiconvergence angle
        eps: wp.float32 - epsilon
        wavelength: wp.float32 - wavelength
        G_out: wp.array(dtype=wp.vec2, ndim=3) - [Qy,Qx,ik]

    Returns:
        None (G_out is modified in place) - output is the corrected G
    """

    iqy, iqx, ik = wp.tid()

    Qx = Qx_all[iqx]
    Qy = Qy_all[iqy]
    Kx = Kx_all[ik]
    Ky = Ky_all[ik]

    Qx_rot = Qx * cos_rot - Qy * sin_rot
    Qy_rot = Qx * sin_rot + Qy * cos_rot

    Qx = Qx_rot
    Qy = Qy_rot

    chi1 = aberration_function_cartesian(Ky, Kx, wavelength, aberrations)
    apert1 = wp.vec2(
        aperture(Ky, Kx, wavelength, semiconvergence_angle), wp.float32(0.0)
    )
    expichi1 = cexp(1.0, -chi1)
    A = cmul(apert1, expichi1)

    chi2 = aberration_function_cartesian(Ky + Qy, Kx + Qx, wavelength, aberrations)
    apert2 = wp.vec2(
        aperture(Ky + Qy, Kx + Qx, wavelength, semiconvergence_angle), wp.float32(0.0)
    )
    expichi2 = cexp(1.0, -chi2)
    A_plus = cmul(apert2, expichi2)

    chi3 = aberration_function_cartesian(Ky - Qy, Kx - Qx, wavelength, aberrations)
    apert3 = wp.vec2(
        aperture(Ky - Qy, Kx - Qx, wavelength, semiconvergence_angle), wp.float32(0.0)
    )
    expichi3 = cexp(1.0, -chi3)
    Am = cmul(apert3, expichi3)

    gamma_complex = cmul(cconj(A), Am) - cmul(A, cconj(A_plus))

    gamma_abs = cabs(gamma_complex)
    gamma_abs = wp.where(gamma_abs < 1e-8, 1e-8, gamma_abs)
    gamma_phase = wp.vec2(gamma_complex[0] / gamma_abs, gamma_complex[1] / gamma_abs)
    gamma_conj = cconj(gamma_phase)
    G_out[iqy, iqx, ik] = cmul(G[iqy, iqx, ik], gamma_conj)


@wp.kernel
def _phase_contrast_transfer_function_forward(
    G: wp.array(dtype=wp.vec2, ndim=3),
    Qx_all: wp.array(ndim=1),
    Qy_all: wp.array(ndim=1),
    Kx_all: wp.array(ndim=1),
    Ky_all: wp.array(ndim=1),
    aberrations: wp.array(dtype=wp.float32, ndim=1),
    sin_rot: wp.float32,
    cos_rot: wp.float32,
    semiconvergence_angle: wp.float32,
    wavelength: wp.float32,
    pctf: wp.array(dtype=wp.float32, ndim=2),
) -> None:
    """
    Forward kernel for the phase contrast transfer function forward pass.

    Args:
        G: wp.array(dtype=wp.vec2, ndim=3) - [Qy,Qx,ik]
        Qx_all: wp.array(ndim=1) - [Qx]
        Qy_all: wp.array(ndim=1) - [Qy]
        Kx_all: wp.array(ndim=1) - [Kx]
        Ky_all: wp.array(ndim=1) - [Ky]
        aberrations: wp.array(dtype=wp.float32, ndim=1)
        sin_rot: wp.float32 - sin(rotation)
        cos_rot: wp.float32 - cos(rotation)
        semiconvergence_angle: wp.float32 - semiconvergence angle
        wavelength: wp.float32 - wavelength
        pctf: wp.array(dtype=wp.float32, ndim=2) - output phase contrast transfer function

    Returns:
        None (pctf is modified in place)
    """

    iqy, iqx, ik = wp.tid()

    Qx = Qx_all[iqx]
    Qy = Qy_all[iqy]
    Kx = Kx_all[ik]
    Ky = Ky_all[ik]

    Qx_rot = Qx * cos_rot - Qy * sin_rot
    Qy_rot = Qx * sin_rot + Qy * cos_rot

    Qx = Qx_rot
    Qy = Qy_rot

    chi1 = aberration_function_cartesian(Ky, Kx, wavelength, aberrations)
    apert1 = wp.vec2(
        aperture(Ky, Kx, wavelength, semiconvergence_angle), wp.float32(0.0)
    )
    expichi1 = cexp(1.0, -chi1)
    A = cmul(apert1, expichi1)

    chi2 = aberration_function_cartesian(Ky + Qy, Kx + Qx, wavelength, aberrations)
    apert2 = wp.vec2(
        aperture(Ky + Qy, Kx + Qx, wavelength, semiconvergence_angle), wp.float32(0.0)
    )
    expichi2 = cexp(1.0, -chi2)
    A_plus = cmul(apert2, expichi2)

    chi3 = aberration_function_cartesian(Ky - Qy, Kx - Qx, wavelength, aberrations)
    apert3 = wp.vec2(
        aperture(Ky - Qy, Kx - Qx, wavelength, semiconvergence_angle), wp.float32(0.0)
    )
    expichi3 = cexp(1.0, -chi3)
    Am = cmul(apert3, expichi3)

    gamma_complex = cmul(cconj(A), Am) - cmul(A, cconj(A_plus))
    gamma_abs = cabs(gamma_complex)  
    wp.atomic_add(pctf, iqy, iqx, gamma_abs)


@wp.kernel(enable_backward=False)
def _direct_ptychography_backward_analytic(
    G: wp.array(dtype=wp.vec2, ndim=3),  # [Qy,Qx,ik]
    dL_dG: wp.array(dtype=wp.vec2, ndim=3),  # same shape
    Qx_all: wp.array(ndim=1),
    Qy_all: wp.array(ndim=1),
    Kx_all: wp.array(ndim=1),
    Ky_all: wp.array(ndim=1),
    aberrations: wp.array(dtype=wp.float32, ndim=1),
    sin_rot: wp.float32,
    cos_rot: wp.float32,
    semiconvergence_angle: wp.float32,
    wavelength: wp.float32,
    n_coeffs: int,  # <= 12
    out_grad: wp.array(dtype=wp.float32, ndim=1),  # length >= n_coeffs
):
    """
    Analytic gradient for the direct ptychography backward pass.

    Args:
        G: wp.array(dtype=wp.vec2, ndim=3) - [Qy,Qx,ik]
        dL_dG: wp.array(dtype=wp.vec2, ndim=3) - same shape
        Qx_all: wp.array(ndim=1) - [Qx]
        Qy_all: wp.array(ndim=1) - [Qy]
        Kx_all: wp.array(ndim=1) - [Kx]
        Ky_all: wp.array(ndim=1) - [Ky]
        aberrations: wp.array(dtype=wp.float32, ndim=1)
        sin_rot: wp.float32 - sin(rotation)
        cos_rot: wp.float32 - cos(rotation)
        semiconvergence_angle: wp.float32 - semiconvergence angle
        wavelength: wp.float32 - wavelength
        n_coeffs: int - <= 12 - number of coefficients
        out_grad: wp.array(dtype=wp.float32, ndim=1) - length >= n_coeffs - output gradient

    Returns:
        out_grad: wp.array(dtype=wp.float32, ndim=1) - length >= n_coeffs

    Notes:
        This kernel computes the analytic gradient of the direct ptychography forward pass.
        It is used to compute the gradient of the aberrations with respect to the loss function.
        It is a per-block tile reduction kernel.
    """

    iqy, iqx, ik = wp.tid()

    # coords
    qx = Qx_all[iqx]
    qy = Qy_all[iqy]
    kx0 = Kx_all[ik]
    ky0 = Ky_all[ik]

    # rotate Q (don't clobber)
    qxr = qx * cos_rot - qy * sin_rot
    qyr = qx * sin_rot + qy * cos_rot

    kx_p = kx0 + qxr
    ky_p = ky0 + qyr
    kx_m = kx0 - qxr
    ky_m = ky0 - qyr

    # upstream adjoint present?
    adj = dL_dG[iqy, iqx, ik]
    has_adj = not ((adj[0] == 0.0) and (adj[1] == 0.0))

    # binary aperture tests (no sqrt)
    kcut = semiconvergence_angle / wavelength
    kcut2 = kcut * kcut

    inp = (kx_p * kx_p + ky_p * ky_p) <= kcut2
    inm = (kx_m * kx_m + ky_m * ky_m) <= kcut2
    active = has_adj

    g_in = G[iqy, iqx, ik]

    # forward terms only when active; unit amplitude (top-hat)
    A0 = wp.vec2(0.0, 0.0)
    Ap = wp.vec2(0.0, 0.0)
    Am = wp.vec2(0.0, 0.0)
    if active:
        chi0 = aberration_function_cartesian(ky0, kx0, wavelength, aberrations)
        A0 = cexp(1.0, -chi0)
        chip = aberration_function_cartesian(ky_p, kx_p, wavelength, aberrations)
        Ap = cexp(1.0, -chip)
        chim = aberration_function_cartesian(ky_m, kx_m, wavelength, aberrations)
        Am = cexp(1.0, -chim)

    C1 = cmul(cconj(A0), Am)  # A* * Am
    C2 = cmul(A0, cconj(Ap))  # A   * A+*

    # coefficient loop with per-block tile reduction
    for j in range(n_coeffs):
        contrib = wp.float32(0.0)

        if active:
            dchi0 = dchi_cartesian_aberrations(ky0, kx0, wavelength, j)
            dchip = (
                dchi_cartesian_aberrations(ky_p, kx_p, wavelength, j) if inp else 0.0
            )
            dchim = (
                dchi_cartesian_aberrations(ky_m, kx_m, wavelength, j) if inm else 0.0
            )

            # dγ/da = i [ C1*(dchi0 - dchim) + C2*(dchi0 - dchip) ]
            t1 = wp.vec2(C1[0] * (dchi0 - dchim), C1[1] * (dchi0 - dchim))
            t2 = wp.vec2(C2[0] * (dchi0 - dchip), C2[1] * (dchi0 - dchip))
            dgamma = minus_i(wp.vec2(t1[0] + t2[0], t1[1] + t2[1]))

            dGout = cmul(g_in, cconj(dgamma))
            contrib = ip_real_conj(adj, dGout)  # Re{ conj(adj) * dGout }

        # cooperative block sum → single global atomic per coeff per block
        t = wp.tile(contrib)  # one scalar per thread
        s = wp.tile_sum(t)  # block-wide sum
        wp.tile_atomic_add(out_grad, s, offset=(j,))