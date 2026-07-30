"""
PyTorch functional operations for direct ptychography.
"""
import math

import torch
from torch import Tensor
import warp as wp

from scatterem.nn.functional.warp.ptychography import (
    _direct_ptychography_backward_analytic,
    _direct_ptychography_forward,
    _phase_contrast_transfer_function_forward,
)


@torch.library.custom_op(
    "scatterem::correct_aberrations_inplace",
    mutates_args=("Gprime_real",),
)
def _correct_aberrations_inplace_op(
    Gprime_real: Tensor,
    aberrations: Tensor,
    sin_cos_rot: Tensor,
    Qx: Tensor,
    Qy: Tensor,
    Kx: Tensor,
    Ky: Tensor,
    semiconvergence_angle: float,
    eps: float,
    wavelength: float,
) -> None:
    # ``sin_cos_rot`` is a length-2 float32 tensor [sin(theta), cos(theta)]
    # carried as a tensor so the wrapper does not need ``.item()`` (which
    # graph-breaks under ``torch.compile(fullgraph=True)``). The conversion
    # happens here, inside the op body, where Dynamo does not trace.
    sin_rot = float(sin_cos_rot[0].item())
    cos_rot = float(sin_cos_rot[1].item())
    device = wp.device_from_torch(Gprime_real.device)
    G_wp = wp.from_torch(Gprime_real, dtype=wp.vec2)
    Qx_wp = wp.from_torch(Qx)
    Qy_wp = wp.from_torch(Qy)
    Kx_wp = wp.from_torch(Kx)
    Ky_wp = wp.from_torch(Ky)
    ab_wp = wp.from_torch(aberrations)
    wp.launch(
        kernel=_direct_ptychography_forward,
        dim=Gprime_real.shape[:-1],
        inputs=[
            G_wp,
            Qx_wp,
            Qy_wp,
            Kx_wp,
            Ky_wp,
            ab_wp,
            sin_rot,
            cos_rot,
            semiconvergence_angle,
            eps,
            wavelength,
        ],
        outputs=[G_wp],
        device=device,
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
    if not isinstance(rotation, torch.Tensor):
        rotation = torch.as_tensor(
            rotation, dtype=torch.float32, device=Gprime.device,
        )
    rotation_rad = torch.deg2rad(rotation)
    sin_cos_rot = torch.stack(
        [torch.sin(rotation_rad), torch.cos(rotation_rad)],
    ).to(dtype=torch.float32).contiguous()
    # In eager mode, alias ``Gprime``'s storage so the kernel mutates the
    # caller's complex tensor in place (preserved API behavior).
    # Under ``torch.compile``, that aliasing path trips inductor's complex
    # codegen + auto_functionalize interaction (verified: scheduler
    # ``get_buf_bytes`` AssertionError when a mutating custom op writes
    # through a real view of a complex base tensor). We fall back to an
    # out-of-place real buffer in the compile path; the returned complex
    # tensor still carries the corrected values, but the caller's
    # ``Gprime`` is not updated under compile. All real callers consume
    # only the return value, so this is observable only by tests that
    # explicitly check post-call ``Gprime`` -- those tests run eager.
    if torch.compiler.is_compiling():
        Gprime_real = torch.view_as_real(Gprime).contiguous().clone()
    else:
        Gprime_real = torch.view_as_real(Gprime).contiguous()
    torch.ops.scatterem.correct_aberrations_inplace(
        Gprime_real,
        aberrations.contiguous(),
        sin_cos_rot,
        Qx.contiguous(),
        Qy.contiguous(),
        Kx.contiguous(),
        Ky.contiguous(),
        float(semiconvergence_angle),
        1e-3,
        float(wavelength),
    )
    return torch.view_as_complex(Gprime_real)


@torch.library.custom_op(
    "scatterem::phase_contrast_transfer_function_fwd",
    mutates_args=(),
)
def _phase_contrast_transfer_function_fwd(
    G_real: Tensor,
    aberrations: Tensor,
    Qx: Tensor,
    Qy: Tensor,
    Kx: Tensor,
    Ky: Tensor,
    sin_rot: float,
    cos_rot: float,
    semiconvergence_angle: float,
    wavelength: float,
) -> Tensor:
    """Compute the un-normalized phase contrast transfer function.

    ``G_real`` is the ``view_as_real`` projection of the complex ``G``
    tensor with shape ``[Nqy, Nqx, Nk, 2]`` (float32, contiguous). The
    returned tensor has shape ``[Nqy, Nqx]`` (float32) and is the
    per-(Qy, Qx) sum of ``|gamma_complex|`` over ``ik``. The bright-field
    normalization ``2 * A.sum()`` is applied by the public wrapper so the
    op body stays pure Warp + a single output allocation.
    """
    device = wp.device_from_torch(G_real.device)
    # G_real shape == [Nqy, Nqx, Nk, 2]; pctf shape == [Nqy, Nqx].
    pctf = torch.zeros(
        G_real.shape[:-2], dtype=torch.float32, device=G_real.device,
    )
    G_wp = wp.from_torch(G_real, dtype=wp.vec2)
    Qx_wp = wp.from_torch(Qx)
    Qy_wp = wp.from_torch(Qy)
    Kx_wp = wp.from_torch(Kx)
    Ky_wp = wp.from_torch(Ky)
    ab_wp = wp.from_torch(aberrations)
    pctf_wp = wp.from_torch(pctf)
    wp.launch(
        kernel=_phase_contrast_transfer_function_forward,
        dim=G_real.shape[:-1],
        inputs=[
            G_wp,
            Qx_wp,
            Qy_wp,
            Kx_wp,
            Ky_wp,
            ab_wp,
            sin_rot,
            cos_rot,
            semiconvergence_angle,
            wavelength,
        ],
        outputs=[pctf_wp],
        device=device,
    )
    return pctf


@_phase_contrast_transfer_function_fwd.register_fake
def _(
    G_real,
    aberrations,
    Qx,
    Qy,
    Kx,
    Ky,
    sin_rot,
    cos_rot,
    semiconvergence_angle,
    wavelength,
):
    return torch.empty(
        G_real.shape[:-2], dtype=torch.float32, device=G_real.device,
    )


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
        G: torch.Tensor - input G tensor (complex64)
        aberrations: torch.Tensor - aberrations array
        rotation: float | 0-d torch.Tensor - rotation in degrees. NOTE:
            passing a tensor here will cause a graph-break under
            ``torch.compile(fullgraph=True)`` because the wrapper calls
            ``rotation.item()`` to materialize a Python float for the
            ``math.sin/cos`` call. Pass a Python ``float`` if you intend
            to compile this function.
        semiconvergence_angle: float - semiconvergence angle
        wavelength: float - wavelength
        Qx: torch.Tensor - Qx coordinates
        Qy: torch.Tensor - Qy coordinates
        Kx: torch.Tensor - Kx coordinates
        Ky: torch.Tensor - Ky coordinates

    Returns:
        torch.Tensor - phase contrast transfer function (float32, [Nqy, Nqx])
    """
    if torch.is_tensor(rotation):
        # OK in eager; graph-breaks under torch.compile(fullgraph=True).
        # Compile-safe callers must pass a Python float (see docstring).
        rotation = float(rotation.item())
    sin_rot = math.sin(math.radians(rotation))
    cos_rot = math.cos(math.radians(rotation))
    G_real = torch.view_as_real(G).contiguous()
    pctf = torch.ops.scatterem.phase_contrast_transfer_function_fwd(
        G_real,
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
    K = torch.sqrt(Ky[None, :] ** 2 + Kx[None, :] ** 2)
    A = K < semiconvergence_angle / wavelength
    pctf_denominator = 2 * A.sum()
    return pctf / pctf_denominator


# ---------------------------------------------------------------------------
# CorrectAberrations -- Pattern H (autograd via hand-written backward kernel)
# ---------------------------------------------------------------------------
#
# The forward op evaluates the production direct-ptychography forward
# (``out = G * conj(gamma_phase)`` with unit-magnitude ``gamma_phase``). The
# backward op invokes ``_direct_ptychography_backward_analytic`` -- the same
# hand-written analytic kernel the legacy ``torch.autograd.Function`` used --
# which differentiates the *unnormalized* effective loss
# (``out_eff = G * conj(gamma_complex)``). This documented forward-vs-backward
# semantic mismatch is preserved verbatim from the original implementation
# (see ``TestCorrectAberrationsAutograd`` for the FD pin against the matched
# effective loss).


@torch.library.custom_op(
    "scatterem::correct_aberrations_fwd",
    mutates_args=(),
)
def _correct_aberrations_fwd(
    Gprime_real: Tensor,  # (Nqy, Nqx, Nk, 2) float32 -- view_as_real(Gprime)
    aberrations: Tensor,
    sin_cos_rot: Tensor,  # length-2 float32 [sin, cos]
    Qx: Tensor,
    Qy: Tensor,
    Kx: Tensor,
    Ky: Tensor,
    semiconvergence_angle: float,
    eps: float,
    wavelength: float,
) -> Tensor:
    """Forward of ``CorrectAberrations``.

    Returns a freshly-allocated ``(Nqy, Nqx, Nk, 2)`` float32 tensor; the
    public wrapper applies ``view_as_complex`` on the way out.
    """
    sin_rot = float(sin_cos_rot[0].item())
    cos_rot = float(sin_cos_rot[1].item())
    device = wp.device_from_torch(Gprime_real.device)
    out_real = torch.zeros_like(Gprime_real)
    G_wp = wp.from_torch(Gprime_real, dtype=wp.vec2)
    out_wp = wp.from_torch(out_real, dtype=wp.vec2)
    Qx_wp = wp.from_torch(Qx)
    Qy_wp = wp.from_torch(Qy)
    Kx_wp = wp.from_torch(Kx)
    Ky_wp = wp.from_torch(Ky)
    ab_wp = wp.from_torch(aberrations)
    wp.launch(
        kernel=_direct_ptychography_forward,
        dim=Gprime_real.shape[:-1],
        inputs=[
            G_wp,
            Qx_wp,
            Qy_wp,
            Kx_wp,
            Ky_wp,
            ab_wp,
            sin_rot,
            cos_rot,
            semiconvergence_angle,
            eps,
            wavelength,
        ],
        outputs=[out_wp],
        device=device,
    )
    return out_real


@_correct_aberrations_fwd.register_fake
def _(
    Gprime_real,
    aberrations,
    sin_cos_rot,
    Qx,
    Qy,
    Kx,
    Ky,
    semiconvergence_angle,
    eps,
    wavelength,
):
    return torch.empty_like(Gprime_real)


@torch.library.custom_op(
    "scatterem::correct_aberrations_bwd",
    mutates_args=(),
)
def _correct_aberrations_bwd(
    grad_out_real: Tensor,  # (Nqy, Nqx, Nk, 2) float32 -- view_as_real(adj_G)
    Gprime_real: Tensor,  # (Nqy, Nqx, Nk, 2) float32 -- saved input
    aberrations: Tensor,
    sin_cos_rot: Tensor,
    Qx: Tensor,
    Qy: Tensor,
    Kx: Tensor,
    Ky: Tensor,
    semiconvergence_angle: float,
    wavelength: float,
    n_coeffs: int,
) -> Tensor:
    """Analytic gradient w.r.t. ``aberrations``.

    Mirrors the legacy wrapper's launch -- crucially with ``block_dim=256``
    so the cooperative ``wp.tile`` / ``wp.tile_atomic_add`` reductions
    inside ``_direct_ptychography_backward_analytic`` work correctly.
    """
    sin_rot = float(sin_cos_rot[0].item())
    cos_rot = float(sin_cos_rot[1].item())
    device = wp.device_from_torch(grad_out_real.device)
    out_grad = torch.zeros((n_coeffs,), dtype=torch.float32, device=grad_out_real.device)
    G_wp = wp.from_torch(Gprime_real, dtype=wp.vec2)
    adj_wp = wp.from_torch(grad_out_real, dtype=wp.vec2)
    Qx_wp = wp.from_torch(Qx)
    Qy_wp = wp.from_torch(Qy)
    Kx_wp = wp.from_torch(Kx)
    Ky_wp = wp.from_torch(Ky)
    ab_wp = wp.from_torch(aberrations)
    out_grad_wp = wp.from_torch(out_grad)
    wp.launch(
        kernel=_direct_ptychography_backward_analytic,
        dim=grad_out_real.shape[:-1],
        inputs=[
            G_wp,
            adj_wp,
            Qx_wp,
            Qy_wp,
            Kx_wp,
            Ky_wp,
            ab_wp,
            sin_rot,
            cos_rot,
            semiconvergence_angle,
            wavelength,
            n_coeffs,
            out_grad_wp,
        ],
        device=device,
        block_dim=256,
    )
    return out_grad


@_correct_aberrations_bwd.register_fake
def _(
    grad_out_real,
    Gprime_real,
    aberrations,
    sin_cos_rot,
    Qx,
    Qy,
    Kx,
    Ky,
    semiconvergence_angle,
    wavelength,
    n_coeffs,
):
    return torch.empty(
        (n_coeffs,), dtype=torch.float32, device=grad_out_real.device,
    )


def _correct_aberrations_setup_context(ctx, inputs, output):
    (
        Gprime_real,
        aberrations,
        sin_cos_rot,
        Qx,
        Qy,
        Kx,
        Ky,
        semiconvergence_angle,
        eps,
        wavelength,
    ) = inputs
    ctx.save_for_backward(
        Gprime_real, aberrations, sin_cos_rot, Qx, Qy, Kx, Ky,
    )
    ctx.semiconvergence_angle = float(semiconvergence_angle)
    ctx.wavelength = float(wavelength)
    ctx.n_coeffs = int(min(12, aberrations.shape[0]))


def _correct_aberrations_backward(ctx, grad_out_real):
    (
        Gprime_real,
        aberrations,
        sin_cos_rot,
        Qx,
        Qy,
        Kx,
        Ky,
    ) = ctx.saved_tensors
    ab_grad = torch.ops.scatterem.correct_aberrations_bwd(
        grad_out_real.contiguous(),
        Gprime_real,
        aberrations,
        sin_cos_rot,
        Qx,
        Qy,
        Kx,
        Ky,
        ctx.semiconvergence_angle,
        ctx.wavelength,
        ctx.n_coeffs,
    )
    # Preserve original-wrapper sign convention: returned ``-ab_grad``.
    # The 10 returns line up with the 10 forward inputs of the custom op
    # (Gprime_real, aberrations, sin_cos_rot, Qx, Qy, Kx, Ky, semi, eps,
    # wavelength). Only ``aberrations`` is differentiable.
    return (None, -ab_grad, None, None, None, None, None, None, None, None)


torch.library.register_autograd(
    "scatterem::correct_aberrations_fwd",
    _correct_aberrations_backward,
    setup_context=_correct_aberrations_setup_context,
)


def correct_aberrations(
    Gprime: torch.Tensor,
    aberrations: torch.Tensor,
    rotation,
    semiconvergence_angle: float,
    wavelength: float,
    Qx: torch.Tensor,
    Qy: torch.Tensor,
    Kx: torch.Tensor,
    Ky: torch.Tensor,
) -> torch.Tensor:
    """Direct-ptychography aberration correction with analytic backward.

    Differentiable input: ``aberrations``. The backward uses the analytic
    ``_direct_ptychography_backward_analytic`` kernel (Pattern H). See the
    forward-vs-backward semantic note at the top of this op block.
    """
    if not isinstance(rotation, torch.Tensor):
        rotation = torch.as_tensor(
            rotation, dtype=torch.float32, device=Gprime.device,
        )
    rotation_rad = torch.deg2rad(rotation)
    sin_cos_rot = torch.stack(
        [torch.sin(rotation_rad), torch.cos(rotation_rad)],
    ).to(dtype=torch.float32).contiguous()
    Gprime_real = torch.view_as_real(Gprime).contiguous()
    out_real = torch.ops.scatterem.correct_aberrations_fwd(
        Gprime_real,
        aberrations.contiguous(),
        sin_cos_rot,
        Qx.contiguous(),
        Qy.contiguous(),
        Kx.contiguous(),
        Ky.contiguous(),
        float(semiconvergence_angle),
        1e-3,
        float(wavelength),
    )
    return torch.view_as_complex(out_real)


# Preserve the legacy ``CorrectAberrations.apply(...)`` API so existing
# callers (``scatterem/reconstruction/direct_ptychography.py``, downstream
# scripts) keep working without edits.
class CorrectAberrations:
    """Compatibility shim exposing the legacy ``.apply`` entry point.

    The original ``torch.autograd.Function`` was replaced by a pair of
    ``torch.library.custom_op`` ops (``scatterem::correct_aberrations_fwd``
    and ``scatterem::correct_aberrations_bwd``) wired together via
    ``torch.library.register_autograd``. ``CorrectAberrations.apply`` now
    forwards to the public :func:`correct_aberrations` callable, which is
    compile-safe under ``torch.compile(fullgraph=True, dynamic=False)``.
    """

    apply = staticmethod(correct_aberrations)




