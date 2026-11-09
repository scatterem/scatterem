"""The hoisted ``ik``-only aperture-times-aberration factor ``A``."""

import torch
import warp as wp

from scatterem.nn.functional.warp.ptychography import (
    _direct_ptychography_build_A,
    _direct_ptychography_build_A_planes,
)
from scatterem.physics import _warp_interop as wi


@torch.no_grad()
def build_aberration_bf_factor(
    aberrations: torch.Tensor,
    semiconvergence_angle: float,
    wavelength: float,
    Kx: torch.Tensor,
    Ky: torch.Tensor,
) -> torch.Tensor:
    """Precompute the ``ik``-only factor ``A = aperture(K) * exp(-i chi(K))``.

    ``_direct_ptychography_forward`` evaluates this per ``(Qy, Qx, ik)`` even
    though it varies only with the bright-field pixel. Building it once for all
    bright-field pixels and handing chunk slices to
    :func:`correct_aberrations_inplace` removes one of the kernel's three
    aberration-polynomial evaluations per element.

    Hoist it ABOVE the loop over bright-field chunks. Called per chunk it is a
    LOSS: the launch costs ~30 us of host time for ~2 us of device work, which
    is more than the device saving it buys back (measured: a 2.4 % end-to-end
    regression when built per chunk, a 1.15x win when built once).

    Args:
        aberrations: aberration coefficients, on the same device as ``Kx``.
        semiconvergence_angle: semiconvergence angle.
        wavelength: wavelength.
        Kx: ``[n_bright_field]`` bright-field pixel x coordinates.
        Ky: ``[n_bright_field]`` bright-field pixel y coordinates.

    Returns:
        torch.Tensor - ``[n_bright_field, 2]`` float32 (real, imag), sliceable
        along dim 0 in step with ``Kx``/``Ky``.
    """
    Kx = Kx.contiguous()
    Ky = Ky.contiguous()
    A = torch.empty((Kx.shape[0], 2), dtype=torch.float32, device=Kx.device)
    wp.launch(
        kernel=_direct_ptychography_build_A,
        dim=(Kx.shape[0],),
        inputs=[
            wi.from_torch(Kx),
            wi.from_torch(Ky),
            wi.from_torch(aberrations.contiguous()),
            float(semiconvergence_angle),
            float(wavelength),
        ],
        outputs=[wi.from_torch(A, dtype=wp.vec2)],
        device=wi.device_from_torch(Kx.device),
    )
    return A


@torch.no_grad()
def build_aberration_bf_factor_planes(
    aberrations: torch.Tensor,
    semiconvergence_angle: float,
    wavelength: float,
    Kx: torch.Tensor,
    Ky: torch.Tensor,
) -> torch.Tensor:
    """:func:`build_aberration_bf_factor` for ``n_planes`` aberration vectors AT ONCE.

    A former per-plane correction sweep called the per-plane form once
    per depth plane above its chunk loop (the TV autofocus uses this one now).
    That kernel is ``len(Kx)`` threads -- 729 at the registered benchmark
    geometry -- so its device half is ~2 us and the
    108 us/call it costs is ALL HOST: ``wp.launch`` 49.7 us, the four
    ``wi.from_torch`` views 30.1, ``torch.empty`` 7.8, and 18.7 in the call plus
    the ``no_grad`` guard (a benchmark script, in situ on the
    row's own arguments).  None of it shrinks with problem size, and the sweep
    paid it ``n_depths`` times for what is one launch's worth of device work.

    So the lever here is LAUNCH COUNT, not arithmetic: an ablation that deletes
    six of the row's seven calls outright bounds the row at **1.0836x**
    (a benchmark script, in-process alternating arms), while
    hoisting every host-side operand and keeping seven launches bounds it at only
    1.0603x.  A form that exploits the fact that the planes differ ONLY in
    ``aberrations[0]`` -- factoring ``A_plane = A_base * exp(-i pi lambda dz |K|^2)``
    -- would buy the ~2 us of device work, forfeit bit-exactness, and is the
    reason this is spelled as a batch and not as a factorisation.

    Args:
        aberrations: ``[n_planes, n_ab]`` float32 coefficients, one row per plane,
            on the same device as ``Kx``.
        semiconvergence_angle: semiconvergence angle.
        wavelength: wavelength.
        Kx: ``[n_bright_field]`` bright-field pixel x coordinates.
        Ky: ``[n_bright_field]`` bright-field pixel y coordinates.

    Returns:
        torch.Tensor - ``[n_planes, n_bright_field, 2]`` float32 (real, imag),
        plane OUTERMOST so that ``out[ip]`` is a contiguous ``[n_bf, 2]`` tensor
        interchangeable with :func:`build_aberration_bf_factor`'s return value.
        Bit-identical to it row by row -- the kernel is the same six lines with a
        leading plane axis on its ``wp.tid()`` and no reduction.
    """
    Kx = Kx.contiguous()
    Ky = Ky.contiguous()
    aberrations = aberrations.contiguous()
    A = torch.empty(
        (aberrations.shape[0], Kx.shape[0], 2),
        dtype=torch.float32,
        device=Kx.device,
    )
    wp.launch(
        kernel=_direct_ptychography_build_A_planes,
        dim=(aberrations.shape[0], Kx.shape[0]),
        inputs=[
            wi.from_torch(Kx),
            wi.from_torch(Ky),
            wi.from_torch(aberrations),
            float(semiconvergence_angle),
            float(wavelength),
        ],
        outputs=[wi.from_torch(A, dtype=wp.vec2)],
        device=wi.device_from_torch(Kx.device),
    )
    return A
