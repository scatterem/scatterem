"""Differentiable aberration correction (Pattern H)."""

import torch
import warp as wp
from torch import Tensor

from scatterem.nn.functional.warp.ptychography import (
    _direct_ptychography_backward_analytic,
    _direct_ptychography_forward,
)
from scatterem.physics import _warp_interop as wi

from ._caches import (
    _sin_cos_floats,
    _sin_cos_rot,
)

# ---------------------------------------------------------------------------
# correct_aberrations -- Pattern H (autograd via hand-written backward kernel)
# ---------------------------------------------------------------------------
#
# The forward op evaluates the production direct-ptychography forward
# (``out = G * conj(gamma_phase)`` with unit-magnitude ``gamma_phase``). The
# backward op invokes ``_direct_ptychography_backward_analytic``, a
# hand-written kernel that computes the exact derivative of that forward
# almost everywhere: its apertures and guard, and the derivative of the
# ``gamma / |gamma|`` normalisation in closed form, without a ``1 / |gamma|``
# factor, so its float32 error near a PCTF zero stays at the forward's own
# (an earlier kernel differentiated a unit-amplitude, unnormalised
# surrogate). The forward jumps where ``gamma`` crosses 0 on a
# two-sided pair; no gradient sees that. A test
# checks it against float64 autograd of an independent transcription. The
# weight has unit modulus, so a loss of ``|out|`` alone has zero gradient.


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
    n_grad_coeffs: int,
) -> Tensor:
    """Forward of :func:`correct_aberrations`.

    Returns a freshly-allocated ``(Nqy, Nqx, Nk, 2)`` float32 tensor; the
    public wrapper applies ``view_as_complex`` on the way out.

    ``n_grad_coeffs`` is unused by the forward -- it is carried here only so
    ``_correct_aberrations_setup_context`` can see it, because the backward's
    cost is linear in it (see the comment on that function).  Negative means
    "all of them", which is what every caller got before it existed.
    """
    sin_rot, cos_rot = _sin_cos_floats(sin_cos_rot)
    device = wi.device_from_torch(Gprime_real.device)
    # ``empty_like``, NOT ``zeros_like``: the launch below covers the destination
    # EXACTLY once and stores unconditionally.  ``dim=Gprime_real.shape[:-1]``
    # gives one thread per ``(iqy, iqx, ik)`` and the last line of
    # ``_direct_ptychography_forward`` is a plain
    # ``G_out[iqy, iqx, ik] = cmul(...)`` OUTSIDE that kernel's aperture guard --
    # the guard skips the aberration polynomials, never the store -- so every
    # ``vec2`` (both floats) is written before anything can read it.  A zero-fill
    # is therefore dead stores, and they are not cheap: the destination is the
    # same size as the input, which at the published ``Fig1_Gd2O3`` fit config is
    # 86.0 MiB per bright-field chunk, and this op runs once per (objective
    # evaluation x chunk) -- 1775-2625 times in ONE ``determine_aberrations``.
    # Measured at that shape: the memset alone is 0.128 ms, and the op goes
    # 0.927 -> 0.796 ms = 1.165x, worth ~4.8 % of the fit stage.
    #
    # Do NOT copy this to a kernel that skips stores.  The sibling
    # ``_phase_contrast_transfer_function_forward`` DOES early-out (R9c) and
    # accumulates with ``wp.atomic_add``, so its destination must stay zeroed;
    # its caller allocates separately and is untouched here.  If
    # ``_direct_ptychography_forward`` ever gains a guarded store, this must go
    # back to ``zeros_like`` in the same commit.
    out_real = torch.empty_like(Gprime_real)
    # ``return_ctype=True`` hands Warp the low-level ``array_t`` descriptor
    # (ptr/grad_ptr/ndim/shape/strides) instead of a ``wp.array`` wrapper.
    # ``pack_arg`` accepts a descriptor verbatim, and it is the SAME BYTES the
    # wrapper would have been packed into -- ``wp.array.__ctype__()`` builds the
    # identical struct, field for field, including the gradient pointer Warp
    # allocates for a ``requires_grad`` input (verified over every array this op
    # wraps in a benchmark script). What it saves is the
    # wrapper construction: 6.1 us -> 2.0 us per 1-D array, 7.2 -> 2.7 for the
    # vec2 views, i.e. ~30 us of the ~100 us this op spends on the host per call
    # -- and this op runs 1800 times in one aberration fit, at 9 % GPU
    # utilisation, so host time is the critical path.
    #
    # This requires the launched kernel to be non-generic (see the dtype
    # annotations on ``_direct_ptychography_forward``): Warp cannot infer an
    # argument type from a bare descriptor. It also means ``pack_arg`` does no
    # dtype check, so anything but float32 falls back to the wrapper, which the
    # now-typed kernel rejects with a clear error rather than misreading the
    # buffer. Non-float32 never worked here -- it failed at Warp codegen before
    # this change -- so the fallback is a guard, not a supported path.
    # Warp launches on the data device; the aberrations vector frequently comes
    # from ``meta.aberrations.array`` on the CPU (metadata is device-agnostic).
    # Align it to the launch device here, at the single torch->Warp boundary, so
    # every caller (direct ptychography, SSNR/PCTF, gradients) is device-safe.
    aberrations = aberrations.to(wp.device_to_torch(device))
    ct = (
        Gprime_real.dtype is torch.float32
        and aberrations.dtype is torch.float32
        and Qx.dtype is torch.float32
        and Qy.dtype is torch.float32
        and Kx.dtype is torch.float32
        and Ky.dtype is torch.float32
    )
    G_wp = wi.from_torch(Gprime_real, dtype=wp.vec2, return_ctype=ct)
    out_wp = wi.from_torch(out_real, dtype=wp.vec2, return_ctype=ct)
    Qx_wp = wi.from_torch(Qx, return_ctype=ct)
    Qy_wp = wi.from_torch(Qy, return_ctype=ct)
    Kx_wp = wi.from_torch(Kx, return_ctype=ct)
    Ky_wp = wi.from_torch(Ky, return_ctype=ct)
    ab_wp = wi.from_torch(aberrations, return_ctype=ct)
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
    n_grad_coeffs,
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

    ``n_coeffs`` is the number of LEADING coefficients the kernel's
    ``for j in range(n_coeffs)`` loop covers; the returned gradient always has
    ``aberrations``' own length, with the coefficients beyond ``n_coeffs``
    left at the zero they are initialised to.  Those two used to be the same
    number, and separating them is what lets a caller that has frozen the
    high-order coefficients stop paying for them -- see
    ``_correct_aberrations_setup_context``.
    """
    sin_rot, cos_rot = _sin_cos_floats(sin_cos_rot)
    device = wi.device_from_torch(grad_out_real.device)
    out_grad = torch.zeros(
        (aberrations.shape[0],), dtype=torch.float32, device=grad_out_real.device
    )
    G_wp = wi.from_torch(Gprime_real, dtype=wp.vec2)
    adj_wp = wi.from_torch(grad_out_real, dtype=wp.vec2)
    Qx_wp = wi.from_torch(Qx)
    Qy_wp = wi.from_torch(Qy)
    Kx_wp = wi.from_torch(Kx)
    Ky_wp = wi.from_torch(Ky)
    # Warp launches on the data device; the aberrations vector frequently comes
    # from ``meta.aberrations.array`` on the CPU (metadata is device-agnostic).
    # Align it to the launch device here, at the single torch->Warp boundary, so
    # every caller (direct ptychography, SSNR/PCTF, gradients) is device-safe.
    aberrations = aberrations.to(wp.device_to_torch(device))
    ab_wp = wi.from_torch(aberrations)
    out_grad_wp = wi.from_torch(out_grad)
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
def _correct_aberrations_bwd_fake(
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
        (aberrations.shape[0],),
        dtype=torch.float32,
        device=grad_out_real.device,
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
        n_grad_coeffs,
    ) = inputs
    ctx.save_for_backward(
        Gprime_real,
        aberrations,
        sin_cos_rot,
        Qx,
        Qy,
        Kx,
        Ky,
    )
    ctx.semiconvergence_angle = float(semiconvergence_angle)
    ctx.wavelength = float(wavelength)
    # ``_direct_ptychography_backward_analytic``'s cost is LINEAR in this
    # number: per coefficient it evaluates ``dchi_cartesian_aberrations``
    # three times and then does a block-wide ``wp.tile_sum`` plus a
    # ``wp.tile_atomic_add``.  Measured with the earlier surrogate adjoint on the published
    # ``Fig1_Gd2O3`` fit chunk (512x512x43): 1.098 ms at ``n_coeffs=1``, 3.530
    # ms at 12, i.e. a ~0.93 ms floor (the 172 MiB of ``G``/``dL_dG`` loads and
    # the three chi evaluations) plus ~0.216 ms per coefficient.  The exact
    # adjoint skips the pairs outside the forward's guard, so it is faster
    # than that kernel on a synthetic 512x512x43 chunk with 34 % of the pairs
    # active (loaded H200).
    #
    # Callers that have FROZEN the high-order coefficients multiply those
    # gradients by exactly zero, so computing them is dead work -- and it is
    # the majority of the kernel: ``determine_aberrations`` at
    # ``correct_order=1`` frees three of twelve.  ``n_grad_coeffs`` lets such
    # a caller say so; it is a LEADING count rather than a mask because
    # ``_build_gradient_mask`` freezes a suffix, and a leading count stays
    # CORRECT (merely less aggressive) under an arbitrary user mask.
    #
    # Skipping a coefficient cannot perturb the ones that remain: each ``j``
    # accumulates into its own ``out_grad[j]`` slot, so this is dead-work
    # removal, not a reassociation.
    n_all = int(min(12, aberrations.shape[0]))
    ctx.n_coeffs = n_all if int(n_grad_coeffs) < 0 else min(n_all, int(n_grad_coeffs))


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
    # The 11 returns line up with the 11 forward inputs of the custom op
    # (Gprime_real, aberrations, sin_cos_rot, Qx, Qy, Kx, Ky, semi, eps,
    # wavelength, n_grad_coeffs). Only ``aberrations`` is differentiable.
    #
    # The op allocates its output on the measurement device (see its
    # register_fake), while ``aberrations`` may live on the CPU -- the kernel
    # wrappers now move it to the launch device instead of raising. Autograd
    # still requires the gradient on the input's own device, so map it back.
    return (
        None,
        -ab_grad.to(aberrations.device),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


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
    n_grad_coeffs: int | None = None,
) -> torch.Tensor:
    """Direct-ptychography aberration correction with analytic backward.

    Differentiable input: ``aberrations``. The backward uses the analytic
    ``_direct_ptychography_backward_analytic`` kernel (Pattern H), the exact
    derivative of this forward; see the note at the top of this op block.

    ``n_grad_coeffs`` (default: all of them) is the number of LEADING
    aberration coefficients the backward computes a gradient for; the rest
    come back as exact zeros. A caller that has frozen the high-order
    coefficients should pass its own count -- the adjoint's cost is linear in
    it. It cannot change the gradients that are computed; see
    ``_correct_aberrations_setup_context``.

    Parameters
    ----------
    Gprime : torch.Tensor
        Complex-valued input of shape ``(Nqy, Nqx, Nk)``.
    aberrations : torch.Tensor
        Aberration coefficients; the only differentiable input.
    rotation : Any
        Scan-to-detector rotation, forwarded to `_sin_cos_rot`.
    semiconvergence_angle : float
        Probe-forming semiconvergence angle.
    wavelength : float
        Electron wavelength.
    Qx : torch.Tensor
        Detector-plane spatial frequencies along x.
    Qy : torch.Tensor
        Detector-plane spatial frequencies along y.
    Kx : torch.Tensor
        Probe-plane spatial frequencies along x.
    Ky : torch.Tensor
        Probe-plane spatial frequencies along y.
    n_grad_coeffs : int or None, optional
        Number of leading aberration coefficients the backward computes a
        gradient for; the rest come back as exact zeros. `None` (default)
        means all of them.

    Returns
    -------
    torch.Tensor
        Complex-valued, aberration-corrected tensor of shape ``(Nqy, Nqx, Nk)``.
    """
    sin_cos_rot = _sin_cos_rot(rotation, Gprime.device)
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
        -1 if n_grad_coeffs is None else int(n_grad_coeffs),
    )
    return torch.view_as_complex(out_real)


# ---------------------------------------------------------------------------
# Fused correct-and-reduce.  Same op pair, same analytic-adjoint contract, but
# the ``ik`` axis is summed inside the kernel.  See the block comment above
# ``_direct_ptychography_forward_ksum`` for what changes numerically and why the
# ``G`` load may move inside the aperture guard.
