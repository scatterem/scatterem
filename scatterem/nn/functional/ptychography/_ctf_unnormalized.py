"""The public PCTF wrapper split at its bright-field normalisation.

:func:`.ctf.phase_contrast_transfer_function` is
``normalize_pctf(pctf_unnormalized(...), ...)``.  The analytical SSNR takes the
two halves separately so its fused Warp epilogue can do the normalisation
divide as well (``reconstruction/ptychography/direct/_ssnr_analytical_warp``);
its torch fallback calls :func:`normalize_pctf`, the same ops on the same
operands as the public function.
"""

import math

import torch


def pctf_unnormalized(
    G: torch.Tensor | None,
    aberrations: torch.Tensor,
    rotation: float | torch.Tensor,
    semiconvergence_angle: float,
    wavelength: float,
    Qx: torch.Tensor,
    Qy: torch.Tensor,
    Kx: torch.Tensor,
    Ky: torch.Tensor,
) -> torch.Tensor:
    """:func:`.ctf.phase_contrast_transfer_function` BEFORE ``/ (2 * A.sum())``.

    Same arguments, same checks, same rotation conversion and the same op.
    """
    if torch.is_tensor(rotation):
        # OK in eager; graph-breaks under torch.compile(fullgraph=True).
        # Compile-safe callers must pass a Python float (see docstring).
        rotation = float(rotation.item())
    sin_rot = math.sin(math.radians(rotation))
    cos_rot = math.cos(math.radians(rotation))
    if G is not None:
        expected = (Qy.shape[0], Qx.shape[0], Kx.shape[0])
        if tuple(G.shape) != expected:
            raise ValueError(
                f"G has shape {tuple(G.shape)}, expected {expected} = "
                "(len(Qy), len(Qx), len(Kx)); the kernel used to take its "
                "launch grid from G, so a mismatch read the coordinate "
                "arrays out of bounds"
            )
    return torch.ops.scatterem.phase_contrast_transfer_function_fwd(
        aberrations.contiguous(),
        Qx.contiguous(),
        Qy.contiguous(),
        Kx.contiguous(),
        Ky.contiguous(),
        sin_rot,
        cos_rot,
        float(semiconvergence_angle),
        float(wavelength),
    )


def normalize_pctf(
    pctf: torch.Tensor,
    Kx: torch.Tensor,
    Ky: torch.Tensor,
    semiconvergence_angle: float,
    wavelength: float,
) -> torch.Tensor:
    """``pctf / (2 * #bright-field pixels)``, the public PCTF's normalisation."""
    K = torch.sqrt(Ky[None, :] ** 2 + Kx[None, :] ** 2)
    A = K < semiconvergence_angle / wavelength
    pctf_denominator = 2 * A.sum()
    return pctf / pctf_denominator
