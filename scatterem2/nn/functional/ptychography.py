"""
PyTorch functional operations for direct ptychography.
"""
import torch
import warp as wp

from scatterem2.nn.functional.warp.ptychography import (
    _direct_ptychography_backward_analytic,
    _direct_ptychography_forward,
    _phase_contrast_transfer_function_forward,
)


@torch.no_grad()
def correct_aberrations_inplace(
    Gprime: torch.Tensor,
    aberrations: torch.Tensor,
    rotation: float,
    semiconvergence_angle: float,
    wavelength: float,
    Qx: torch.Tensor,
    Qy: torch.Tensor,
    Kx: torch.Tensor,
    Ky: torch.Tensor,
):
    """
    Correct aberrations in place using direct ptychography.

    Args:
        Gprime: torch.Tensor - input G tensor
        aberrations: torch.Tensor - aberrations array
        rotation: float - rotation in degrees
        semiconvergence_angle: float - semiconvergence angle
        wavelength: float - wavelength
        Qx: torch.Tensor - Qx coordinates
        Qy: torch.Tensor - Qy coordinates
        Kx: torch.Tensor - Kx coordinates
        Ky: torch.Tensor - Ky coordinates

    Returns:
        torch.Tensor - corrected G tensor
    """
    device = wp.device_from_torch(Gprime.device)

    # Warp views
    G_wp = wp.from_torch(torch.view_as_real(Gprime), dtype=wp.vec2, requires_grad=False)
    Qx_wp = wp.from_torch(Qx, requires_grad=False)
    Qy_wp = wp.from_torch(Qy, requires_grad=False)
    Kx_wp = wp.from_torch(Kx, requires_grad=False)
    Ky_wp = wp.from_torch(Ky, requires_grad=False)
    ab_wp = wp.from_torch(aberrations, requires_grad=True)

    sin_rot = wp.float32(float(torch.sin(torch.deg2rad(rotation))))
    cos_rot = wp.float32(float(torch.cos(torch.deg2rad(rotation))))
    semi_wp = wp.float32(semiconvergence_angle)
    lam_wp = wp.float32(wavelength)

    wp.launch(
        kernel=_direct_ptychography_forward,
        dim=Gprime.shape,
        inputs=[
            G_wp,
            Qx_wp,
            Qy_wp,
            Kx_wp,
            Ky_wp,
            ab_wp,
            sin_rot,
            cos_rot,
            semi_wp,
            wp.float32(1e-3),
            lam_wp,
        ],
        outputs=[G_wp],
        device=device,
        record_tape=False,
    )
    return torch.view_as_complex(wp.to_torch(G_wp))


@torch.no_grad()
def phase_contrast_transfer_function(
    G: torch.Tensor,
    aberrations: torch.Tensor,
    rotation: float,
    semiconvergence_angle: float,
    wavelength: float,
    Qx: torch.Tensor,
    Qy: torch.Tensor,
    Kx: torch.Tensor,
    Ky: torch.Tensor,
):
    """
    Compute the phase contrast transfer function.

    Args:
        G: torch.Tensor - input G tensor
        aberrations: torch.Tensor - aberrations array
        rotation: float - rotation in degrees
        semiconvergence_angle: float - semiconvergence angle
        wavelength: float - wavelength
        Qx: torch.Tensor - Qx coordinates
        Qy: torch.Tensor - Qy coordinates
        Kx: torch.Tensor - Kx coordinates
        Ky: torch.Tensor - Ky coordinates

    Returns:
        torch.Tensor - phase contrast transfer function
    """
    device = wp.device_from_torch(G.device)

    # Warp views
    G_wp = wp.from_torch(torch.view_as_real(G), dtype=wp.vec2, requires_grad=False)
    Qx_wp = wp.from_torch(Qx, requires_grad=False)
    Qy_wp = wp.from_torch(Qy, requires_grad=False)
    Kx_wp = wp.from_torch(Kx, requires_grad=False)
    Ky_wp = wp.from_torch(Ky, requires_grad=False)
    ab_wp = wp.from_torch(aberrations, requires_grad=True)

    sin_rot = wp.float32(float(torch.sin(torch.deg2rad(rotation))))
    cos_rot = wp.float32(float(torch.cos(torch.deg2rad(rotation))))
    semi_wp = wp.float32(semiconvergence_angle)
    lam_wp = wp.float32(wavelength)

    K = torch.sqrt(Ky[None, :] ** 2 + Kx[None, :] ** 2)
    A = K < semiconvergence_angle / wavelength
    pctf_denominator = 2 * A.sum()
    pctf_wp = wp.zeros(G.shape[:-1], dtype=wp.float32, requires_grad=False)

    wp.launch(
        kernel=_phase_contrast_transfer_function_forward,
        dim=G.shape,
        inputs=[
            G_wp,
            Qx_wp,
            Qy_wp,
            Kx_wp,
            Ky_wp,
            ab_wp,
            sin_rot,
            cos_rot,
            semi_wp,
            lam_wp,
        ],
        outputs=[pctf_wp],
        device=device,
        record_tape=False,
    )
    pctf = wp.to_torch(pctf_wp) / pctf_denominator
    return pctf


class CorrectAberrations(torch.autograd.Function):
    """
    Direct ptychography with analytic backward gradient.
    """

    @staticmethod
    def forward(
        ctx,
        Gprime: torch.Tensor,
        aberrations: torch.Tensor,
        rotation: float,
        semiconvergence_angle: float,
        wavelength: float,
        Qx: torch.Tensor,
        Qy: torch.Tensor,
        Kx: torch.Tensor,
        Ky: torch.Tensor,
    ):
        device = wp.device_from_torch(Gprime.device)

        # Warp views
        G_wp = wp.from_torch(
            torch.view_as_real(Gprime), dtype=wp.vec2, requires_grad=False
        )
        Qx_wp = wp.from_torch(Qx, requires_grad=False)
        Qy_wp = wp.from_torch(Qy, requires_grad=False)
        Kx_wp = wp.from_torch(Kx, requires_grad=False)
        Ky_wp = wp.from_torch(Ky, requires_grad=False)
        ab_wp = wp.from_torch(aberrations, requires_grad=True)

        sin_rot = wp.float32(float(torch.sin(torch.deg2rad(rotation))))
        cos_rot = wp.float32(float(torch.cos(torch.deg2rad(rotation))))
        semi_wp = wp.float32(semiconvergence_angle)
        lam_wp = wp.float32(wavelength)

        G_out_wp = wp.zeros(
            torch.view_as_real(Gprime).shape[:-1], dtype=wp.vec2, requires_grad=False
        )

        wp.launch(
            kernel=_direct_ptychography_forward,
            dim=Gprime.shape,
            inputs=[
                G_wp,
                Qx_wp,
                Qy_wp,
                Kx_wp,
                Ky_wp,
                ab_wp,
                sin_rot,
                cos_rot,
                semi_wp,
                wp.float32(1e-3),
                lam_wp,
            ],
            outputs=[G_out_wp],
            device=device,
            record_tape=False,
        )

        # Save what we need for backward
        ctx.G_wp = G_wp
        ctx.G_out_wp = G_out_wp
        ctx.Qx_wp, ctx.Qy_wp, ctx.Kx_wp, ctx.Ky_wp = Qx_wp, Qy_wp, Kx_wp, Ky_wp
        ctx.ab_wp = ab_wp
        ctx.sin_rot, ctx.cos_rot, ctx.semi_wp, ctx.lam_wp = (
            sin_rot,
            cos_rot,
            semi_wp,
            lam_wp,
        )
        ctx.n_coeffs = int(aberrations.shape[0])

        return torch.view_as_complex(wp.to_torch(G_out_wp))

    @staticmethod
    def backward(ctx, adj_G):
        # Map incoming gradient from Torch to Warp
        adj_wp = wp.from_torch(
            torch.view_as_real(adj_G), dtype=wp.vec2, requires_grad=False
        )

        n_coeffs = int(min(12, ctx.n_coeffs))  # or aberrations.shape[0]
        out_grad_wp = wp.zeros(n_coeffs, dtype=wp.float32, requires_grad=False)

        wp.launch(
            kernel=_direct_ptychography_backward_analytic,
            dim=ctx.G_out_wp.shape,
            inputs=[
                ctx.G_wp,
                adj_wp,
                ctx.Qx_wp,
                ctx.Qy_wp,
                ctx.Kx_wp,
                ctx.Ky_wp,
                ctx.ab_wp,
                ctx.sin_rot,
                ctx.cos_rot,
                ctx.semi_wp,
                ctx.lam_wp,
                n_coeffs,
                out_grad_wp,
            ],
            device=ctx.G_wp.device,
            record_tape=False,
            block_dim=256,
        )

        # Convert to Torch
        ab_grad = wp.to_torch(out_grad_wp)

        # Return grads in the order of forward() args
        return (
            None,  # Gprime
            -ab_grad,  # aberrations
            None,
            None,
            None,
            None,  # rotation, semiconv, wavelength, sampling
            None,
            None,
            None,
            None,  # Qx, Qy, Kx, Ky
        )