"""In-place (and out-of-place) aberration correction: one chunk, one launch."""

import torch
import warp as wp
from torch import Tensor

from scatterem.nn.functional.warp.ptychography import (
    _direct_ptychography_build_A,
    _direct_ptychography_forward_precomputed_A,
)
from scatterem.physics import _warp_interop as wi

from ._caches import (
    _sin_cos_floats,
    _sin_cos_rot,
)


@torch.library.custom_op(
    "scatterem::correct_aberrations_inplace",
    mutates_args=("Gprime_real", "out_real"),
)
def _correct_aberrations_inplace_op(
    Gprime_real: Tensor,
    aberrations: Tensor,
    sin_cos_rot: Tensor,
    Qx: Tensor,
    Qy: Tensor,
    Kx: Tensor,
    Ky: Tensor,
    A: Tensor,
    out_real: Tensor,
    semiconvergence_angle: float,
    eps: float,
    wavelength: float,
) -> None:
    # ``sin_cos_rot`` is a length-2 float32 tensor [sin(theta), cos(theta)]
    # carried as a tensor so the wrapper does not need ``.item()`` (which
    # graph-breaks under ``torch.compile(fullgraph=True)``). The conversion
    # happens here, inside the op body, where Dynamo does not trace -- memoised,
    # because it is a device sync; see ``_sin_cos_floats``.
    sin_rot, cos_rot = _sin_cos_floats(sin_cos_rot)
    device = wi.device_from_torch(Gprime_real.device)
    G_wp = wi.from_torch(Gprime_real, dtype=wp.vec2)
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
    # ``A`` is the ``ik``-only aperture-times-aberration factor. An empty tensor
    # is the sentinel for "not hoisted": build it here, one launch for the whole
    # chunk. Callers that drive this in a loop over bright-field chunks pass a
    # slice of one precomputed ``[n_bright_field, 2]`` buffer instead, which is
    # what makes the hoist pay -- see ``build_aberration_bf_factor``. The test is
    # on ``numel``, inside the op body, where Dynamo does not trace.
    if A.numel() == 0:
        A = torch.empty(
            (Kx.shape[0], 2),
            dtype=torch.float32,
            device=Gprime_real.device,
        )
        wp.launch(
            kernel=_direct_ptychography_build_A,
            dim=(Kx.shape[0],),
            inputs=[
                Kx_wp,
                Ky_wp,
                ab_wp,
                semiconvergence_angle,
                wavelength,
            ],
            outputs=[wi.from_torch(A, dtype=wp.vec2)],
            device=device,
        )
    A_wp = wi.from_torch(A, dtype=wp.vec2)
    # ``out_real`` is the out-of-place destination; an empty tensor is the sentinel
    # for "correct in place", which aims the kernel's output at its own input as
    # before. The kernel already reads ``G`` and writes ``G_out`` as separate
    # arrays, so the only difference between the two modes is which array this
    # launch is pointed at. Out-of-place spares a caller that must preserve its
    # input the separate ``clone()`` pass -- see ``correct_aberrations_inplace``.
    out_wp = G_wp if out_real.numel() == 0 else wi.from_torch(out_real, dtype=wp.vec2)
    wp.launch(
        kernel=_direct_ptychography_forward_precomputed_A,
        dim=Gprime_real.shape[:-1],
        inputs=[
            G_wp,
            Qx_wp,
            Qy_wp,
            Kx_wp,
            Ky_wp,
            A_wp,
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
    A: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
):
    """Correct aberrations in place using direct ptychography.

    Parameters
    ----------
    Gprime : torch.Tensor
        Input G tensor (complex).
    aberrations : torch.Tensor
        Aberration coefficients array.
    rotation : float
        Scan/detector rotation, in degrees.
    semiconvergence_angle : float
        Probe semiconvergence angle.
    wavelength : float
        Electron wavelength.
    Qx : torch.Tensor
        Qx coordinates.
    Qy : torch.Tensor
        Qy coordinates.
    Kx : torch.Tensor
        Kx coordinates.
    Ky : torch.Tensor
        Ky coordinates.
    A : torch.Tensor, optional
        `[len(Kx), 2]` float32 tensor from
        :func:`build_aberration_bf_factor` -- the ``ik``-only
        ``aperture(K) * exp(-i chi(K))`` factor, hoisted out of a loop over
        bright-field chunks. Default is None: the kernel's own launch
        builds it; the result is identical either way (the hoisted buffer
        is a memoisation of the same float32 arithmetic), only the number
        of launches differs.
    out : torch.Tensor, optional
        Pre-allocated contiguous complex tensor shaped like `Gprime`.
        Default is None (correct in place). Given one, the correction is
        computed OUT OF PLACE: `Gprime` is left untouched and the
        corrected values land in `out`, which is what is returned. A
        caller that must preserve `Gprime` (because it corrects the same
        chunk again under different aberrations) would otherwise have to
        `clone()` first, and that clone is a whole extra read+write pass
        over the chunk on top of the kernel's own -- four passes where two
        will do. The arithmetic is identical: the kernel already reads `G`
        and writes `G_out` as separate arrays, so only the destination
        changes.

    Returns
    -------
    torch.Tensor
        Corrected G tensor.
    """
    sin_cos_rot = _sin_cos_rot(rotation, Gprime.device)
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
    if A is None:
        # Empty sentinel: the op body builds the factor itself. Kept as a tensor
        # rather than an Optional so the custom-op schema stays a plain Tensor
        # argument under ``torch.compile(fullgraph=True)``.
        A = torch.empty((0, 2), dtype=torch.float32, device=Gprime.device)
    if out is None:
        # Same empty-sentinel convention as ``A``: "no destination" means correct
        # in place, and the schema stays a plain Tensor argument.
        out_real = torch.empty((0, 2), dtype=torch.float32, device=Gprime.device)
    else:
        if out.shape != Gprime.shape or out.dtype != Gprime.dtype:
            raise ValueError(
                f"out must match Gprime in shape and dtype; got {tuple(out.shape)}/"
                f"{out.dtype} for {tuple(Gprime.shape)}/{Gprime.dtype}"
            )
        out_real = torch.view_as_real(out)
        if not out_real.is_contiguous():
            raise ValueError("out must be contiguous")
    torch.ops.scatterem.correct_aberrations_inplace(
        Gprime_real,
        aberrations.contiguous(),
        sin_cos_rot,
        Qx.contiguous(),
        Qy.contiguous(),
        Kx.contiguous(),
        Ky.contiguous(),
        A.contiguous(),
        out_real,
        float(semiconvergence_angle),
        1e-3,
        float(wavelength),
    )
    return torch.view_as_complex(Gprime_real if out is None else out_real)
