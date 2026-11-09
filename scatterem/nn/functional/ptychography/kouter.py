"""k-outer aberration correction -- the ``[.., Nk, ..]``-preserving forward.

Includes the recorded-``wp.launch`` fast path (``_KOUTER_CMD``) that replays a
bound command with only the operands that actually moved rebound.
"""

import torch
import warp as wp
from torch import Tensor

from scatterem.nn.functional.warp.ptychography import _direct_ptychography_build_A
from scatterem.physics import _warp_interop as wi

from ._caches import (
    _chi_is_zero_flag,
    _contiguous_memo,
    _sin_cos_floats,
    _sin_cos_rot,
)
from ._sq_aperture import _half_mode, _kouter_kernel, _launch_dim

# A RECORDED ``wp.launch`` for the k-outer correction kernel, replayed with only the
# operands that actually moved rebound.  A measurement put ``wp.launch`` at a FIXED
# 32-124 us of host time per call that does not shrink with problem size, and
# a benchmark row (a former per-plane correction
# sweep) paid it 45 times inside an 18 ms wall that is 91 % GPU-idle.  Warp's own fast
# path for that shape is ``wp.launch(record_cmd=True)`` once plus ``set_param_at_index``
# + ``.launch()`` per call -- an arrangement that was measured
# bit-exact over 60 calls.
#
# Keyed on the launch grid and device, because those are what the recorded command
# fixes; every other operand is compared and rebound.
#
# WHY THE ENTRY HOLDS THE TENSORS.  ``set_param_at_index`` packs a raw pointer into
# the recorded parameter block.  If the tensor behind that pointer is freed, the
# block dangles and the kernel silently reads whatever the caching allocator handed
# out next -- which is not a crash and not a wrong-looking number, it is a
# reproducibility bug that appears only when the allocator happens to reuse the
# block.  (Measured: a first draft of this that let ``Qx.contiguous()``'s temporary
# die produced a different image digest in one probe arm ordering and the correct
# one in another.)  So the entry keeps a strong reference to every bound tensor,
# which both keeps the pointer valid and makes ``is``-identity a sound cheap test.
_KOUTER_CMD: dict = {}
_KOUTER_CMD_MAX = 4

#: ``(param index, wp dtype or None)`` for every array argument of
#: ``_direct_ptychography_forward_precomputed_A_kouter``.  Inputs occupy 0..12 --
#: 0..6 are the arrays, 7..11 the scalars and 12 the one-element ``chi_is_zero``
#: predicate -- and the single output is 13, which is the order
#: ``wp.launch`` packs them in.
_KOUTER_ARRAY_SLOTS = (0, 1, 2, 3, 4, 5, 6, 12, 13)

#: The subset of ``_KOUTER_ARRAY_SLOTS`` whose kernel argument is ``wp.vec2`` --
#: the complex ``[.., 2]`` operands ``G``, ``A`` and the output.
_KOUTER_VEC2_SLOTS = frozenset((0, 5, 13))


def _launch_recorded_command(cmd: wp.Launch | None) -> None:
    """Launch a recorded Warp command, rejecting a missing command."""
    if cmd is None:
        raise ValueError("record_cmd=True must return a Warp launch command")
    cmd.launch()


def _wp_ndim(t: Tensor, vec2: bool) -> int:
    """``wp.from_torch(t, dtype=...)``'s resulting ``ndim``, without building it.

    ``from_torch`` trims one trailing axis for a ``vec2`` target (and falls back
    to a 1-D shape when nothing is left), so the rank of the ``array_t`` it packs
    is a pure function of the torch rank.  Kept in ``_KOUTER_CMD``'s bound record
    so the rebind below can check the one thing ``set_param_at_index_from_ctype``
    does not: that the recorded slot and the new descriptor have the same rank.
    """
    return max(t.dim() - 1, 1) if vec2 else t.dim()


def _correct_aberrations_kouter_impl(
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
    defocus_only: bool = False,
    half: int = 0,
) -> None:
    """``scatterem::correct_aberrations_inplace`` on a ``[Nk, Nqy, Nqx]`` chunk.

    Always OUT OF PLACE -- the caller of the k-outer layout is by construction one
    that corrects the same chunk repeatedly under different aberrations, so the
    in-place sentinel of the ``[Nqy, Nqx, Nk]`` op has no user here and is not
    carried over.

    ``defocus_only`` selects the specialised kernel
    ``_direct_ptychography_forward_precomputed_A_kouter_defocus_only``, which is
    BIT-IDENTICAL whenever ``aberrations[1:]`` is entirely ``+-0.0``.  It is a
    plain ``bool`` and NOT derived here on purpose: deriving it costs a host sync
    on ``aberrations``, which only a caller looping over chunks amortises (the
    former per-plane sweep did).  Defaults to ``False`` = the shipped kernel.

    A non-``defocus_only`` call takes the squared-radius aperture twin
    (BIT-IDENTICAL; see ``_kernels_sq``).  ``half=True`` (:mod:`.kouter_half`)
    writes ``H = -i (C(q) - conj C(-q)) / 2`` onto ``(Nk, Hq, W//2+1, 2)``, and
    ``half=HALF_PACKED`` (:mod:`.kouter_packed`) writes the packed
    ``H_{2j} + i H_{2j+1}`` onto ``(ceil(Nk/2), Hq, W, 2)``.  The threshold is
    derived on the RECORD path only: it is a pure function of ``(wavelength,
    semiconvergence_angle, device)``, all in the ``(key, scalars)`` pair a replay
    must match, so the RAW angle is recorded and the replay path never derives it.
    """
    sin_rot, cos_rot = _sin_cos_floats(sin_cos_rot)
    device = wi.device_from_torch(Gprime_real.device)
    # NOTHING is wrapped in a ``wp.array`` before the memoised replay path below
    # has had its say.  This prologue used to build four of them -- ``G``, ``Kx``,
    # ``Ky``, ``aberrations`` -- unconditionally, and on a warm reconstruction all
    # four are DEAD: ``A`` arrives precomputed, ``_chi_is_zero_flag`` hits, and the
    # replay rebinds from the torch tensors directly.  Measured on
    # ``r14_ffstem/carbon/ptycho``: 25 entries x 25.4 us = 0.634 ms of a 14.0 ms
    # row, on a path that is 98.4 % host-bound.  Each handle is now built by the
    # branch that actually uses it.
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
                wi.from_torch(Kx),
                wi.from_torch(Ky),
                wi.from_torch(aberrations),
                semiconvergence_angle,
                wavelength,
            ],
            outputs=[wi.from_torch(A, dtype=wp.vec2)],
            device=device,
        )
    chi_is_zero = _chi_is_zero_flag(aberrations, device)
    # THE GRID COMES FROM ``out_real``, NOT FROM ``Gprime_real``, and the two may
    # now differ by an integer tile factor on the two Q axes.  The kernel indexes
    # ``G`` modulo its own Q extents, so a caller that upsamples hands in the
    # UN-TILED chunk and this covers the upsampled output exactly once -- see the
    # comment on ``_direct_ptychography_forward_precomputed_A_kouter``'s store.
    # For every non-upsampling caller the two shapes are equal and this is the
    # expression it always was.
    dim = tuple(int(v) for v in out_real.shape[:-1])
    # The five scalars are recorded INTO the command, so a change in any of them is
    # a re-record and not a rebind -- cheaper to compare than to set, and it keeps
    # the scalar path impossible to get wrong.
    scalars = (
        float(sin_rot),
        float(cos_rot),
        float(semiconvergence_angle),
        float(eps),
        float(wavelength),
    )
    arrays = (Gprime_real, Qx, Qy, Kx, Ky, A, aberrations, chi_is_zero, out_real)
    sq = bool(half) or not defocus_only
    # ``defocus_only``, ``sq`` and ``half`` are part of the KEY, not of ``scalars``:
    # each selects a different kernel, and a recorded command carries the kernel it
    # was recorded against, so replaying one across a flag would launch the wrong body.
    key = (dim, str(device), bool(defocus_only), sq, _half_mode(half))
    entry = _KOUTER_CMD.get(key)
    if entry is not None and entry[1] == scalars:
        cmd, _, bound = entry
        for slot, t in zip(_KOUTER_ARRAY_SLOTS, arrays, strict=True):
            prev = bound[slot]
            if prev[0] is t and prev[1] == t.data_ptr() and prev[2] == t.shape:
                continue
            # A rebind wants exactly one thing: the ``array_t`` descriptor for the
            # new operand.  ``wi.from_torch`` builds a whole ``wp.array`` PYTHON
            # OBJECT, which ``set_param_at_index`` then converts to that descriptor
            # and throws away -- so the wrapper is pure overhead on this path.
            # ``return_ctype=True`` packs the descriptor directly (warp's own
            # docstring: "faster") and it is BYTE-IDENTICAL to the discarded route:
            # 7.4 -> 2.9 us for a vec2 operand, 6.2 -> 2.1 us for a plain one,
            # 125 rebinds per ``r14_ffstem/carbon/ptycho`` row.
            vec2 = slot in _KOUTER_VEC2_SLOTS
            ct = (
                wi.from_torch(t, dtype=wp.vec2, return_ctype=True)
                if vec2
                else wi.from_torch(t, return_ctype=True)
            )
            # ``set_param_at_index_from_ctype`` writes the struct with NO type
            # conversion, so it is also the one spelling that does not re-check the
            # operand against the kernel's declared argument.  ``from_torch`` still
            # validates dtype, inner shape and inner strides; RANK is the one thing
            # it cannot know, so it is checked here against the rank recorded when
            # this command was built, and a mismatch falls back to the checking
            # setter rather than writing a wrong descriptor.
            if int(ct.ndim) == prev[3]:
                cmd.set_param_at_index_from_ctype(slot, ct)
            else:
                cmd.set_param_at_index(
                    slot,
                    wi.from_torch(t, dtype=wp.vec2) if vec2 else wi.from_torch(t),
                )
            bound[slot] = (t, t.data_ptr(), t.shape, int(ct.ndim))
        cmd.launch()
        return
    kern, angle_arg = _kouter_kernel(
        defocus_only, sq, key[4], wavelength, semiconvergence_angle, Kx.device
    )
    cmd = wp.launch(
        kernel=kern,
        dim=_launch_dim(kern, dim),
        inputs=[
            wi.from_torch(Gprime_real, dtype=wp.vec2),
            wi.from_torch(Qx),
            wi.from_torch(Qy),
            wi.from_torch(Kx),
            wi.from_torch(Ky),
            wi.from_torch(A, dtype=wp.vec2),
            wi.from_torch(aberrations),
            sin_rot,
            cos_rot,
            angle_arg,
            eps,
            wavelength,
            wi.from_torch(chi_is_zero),
        ],
        outputs=[wi.from_torch(out_real, dtype=wp.vec2)],
        device=device,
        record_cmd=True,
    )
    # ``record_cmd=True`` RECORDS and does NOT launch.
    _launch_recorded_command(cmd)
    if len(_KOUTER_CMD) >= _KOUTER_CMD_MAX:
        _KOUTER_CMD.clear()
    _KOUTER_CMD[key] = (
        cmd,
        scalars,
        {
            slot: (t, t.data_ptr(), t.shape, _wp_ndim(t, slot in _KOUTER_VEC2_SLOTS))
            for slot, t in zip(_KOUTER_ARRAY_SLOTS, arrays, strict=True)
        },
    )


# The op stays REGISTERED and unchanged -- ``torch.ops.scatterem
# .correct_aberrations_kouter`` still exists, still declares ``out_real`` mutated,
# and is still what Dynamo sees.  The body is simply named separately so the eager
# wrapper below can reach it without the dispatcher; the two are the same function
# on the same tensors, so every output word is identical by construction.
@torch.library.custom_op(
    "scatterem::correct_aberrations_kouter",
    mutates_args=("out_real",),
)
def _correct_aberrations_kouter_op(
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
    defocus_only: bool = False,
) -> None:
    _correct_aberrations_kouter_impl(
        Gprime_real,
        aberrations,
        sin_cos_rot,
        Qx,
        Qy,
        Kx,
        Ky,
        A,
        out_real,
        semiconvergence_angle,
        eps,
        wavelength,
        defocus_only,
    )


@torch.no_grad()
def correct_aberrations_kouter(
    Gprime: torch.Tensor,
    aberrations: torch.Tensor,
    rotation: float,
    semiconvergence_angle: float,
    wavelength: float,
    Qx: torch.Tensor,
    Qy: torch.Tensor,
    Kx: torch.Tensor,
    Ky: torch.Tensor,
    A: torch.Tensor,
    out: torch.Tensor,
    defocus_only: bool = False,
):
    """:func:`correct_aberrations_inplace` for a ``[Nk, Nqy, Nqx]`` chunk.

    Identical arithmetic -- the Warp kernel behind this is
    ``_direct_ptychography_forward_precomputed_A`` with its ``wp.tid()`` unpacking
    and its three subscripts permuted and nothing else changed, and the correction
    is a per-element multiply, so every output word is bit-identical to what the
    ``[Nqy, Nqx, Nk]`` op writes for the same input.

    Why the layout exists: ``ifft2`` over a chunk's two Q axes is **1.66x** faster
    when they are the INNER two (a natural batched transform) than when they are
    the outer two of a ``[Nqy, Nqx, Nk]`` tensor, and bit-for-bit identical
    (measured).  Only a caller that transforms the same chunk several times can
    afford the transposing copy that gets it there, so this is deliberately not
    the default path -- see ``_chunking._g_chunk_kouter_view``.

    Args:
        Gprime: ``[Nk, Nqy, Nqx]`` complex chunk, left UNTOUCHED.
        aberrations: aberration coefficients.
        rotation: scan/detector rotation in degrees (float or device tensor).
        semiconvergence_angle: semiconvergence angle.
        wavelength: wavelength.
        Qx, Qy, Kx, Ky: coordinate arrays, as for
            :func:`correct_aberrations_inplace`.
        A: ``[Nk, 2]`` float32 factor from :func:`build_aberration_bf_factor`.
            Pass an empty ``(0, 2)`` tensor to have the op build it.
        out: pre-allocated contiguous complex tensor shaped like ``Gprime``; the
            corrected values land here and it is what is returned.
        defocus_only: ``True`` iff ``aberrations[1:]`` is entirely ``+-0.0``,
            which selects a BIT-IDENTICAL kernel with the eleven dead polynomial
            terms folded away (1.1806x on the kernel).  The CALLER decides
            it -- deriving it here would put a host sync on every call.

    Returns:
        torch.Tensor - ``out``, as a complex view.
    """
    # ``out`` may be an integer TILE of ``Gprime`` on the two Q axes: the kernel
    # reads ``G`` modulo its own Q extents, so an upsampling caller passes the
    # un-tiled chunk and the corrected output at the upsampled grid, and the 4x
    # ``torch.tile`` materialisation that used to sit in front of this op is gone.
    # The bright-field axis must still match exactly -- it indexes ``Kx``/``Ky``/
    # ``A`` -- and the tile factors must divide, or the modular read would wrap
    # mid-plane and produce something that is not a tile of anything.
    if (
        out.dtype != Gprime.dtype
        or out.ndim != Gprime.ndim
        or out.shape[0] != Gprime.shape[0]
        or out.shape[-2] % Gprime.shape[-2] != 0
        or out.shape[-1] % Gprime.shape[-1] != 0
    ):
        raise ValueError(
            f"out must be an integer Q-tile of Gprime with the same dtype; got "
            f"{tuple(out.shape)}/{out.dtype} for "
            f"{tuple(Gprime.shape)}/{Gprime.dtype}"
        )
    out_real = torch.view_as_real(out)
    if not out_real.is_contiguous():
        raise ValueError("out must be contiguous")
    # The ``torch.ops`` boundary is only load-bearing while Dynamo is tracing --
    # that is the whole reason the op exists (its body does ``.item()`` and reads
    # ``A.numel()``, neither of which may be traced).  In EAGER it buys nothing and
    # costs the custom-op dispatch chain, measured on THIS op at 31.66 us/call
    # (``torch.ops`` entry minus the body called directly) against a 24 us device
    # kernel.  An earlier census priced the same constant and censused it at <=4.3 % of any row
    # then registered -- but ``r14_ffstem`` was registered after that census, and
    # ``co3o4/ptycho`` calls this op 25 times inside a 10.4 ms wall that is 63 %
    # GPU-idle, so here the same constant is 7.6 % of the row and it CASHES.
    #
    # BIT-EXACT by construction: the two branches are the same function object
    # applied to the same tensors, so no arithmetic is repeated and none can
    # differ.  Under ``torch.compiler.is_compiling()`` the ``torch.ops`` branch is
    # taken and behaviour is exactly what it was -- the op is still registered,
    # still declares ``out_real`` mutated, and is still what Dynamo sees.  The
    # same predicate already guards ``_sin_cos_rot`` and ``_contiguous_memo`` in
    # this file, for the same reason.
    if torch.compiler.is_compiling():
        call = torch.ops.scatterem.correct_aberrations_kouter
    else:
        call = _correct_aberrations_kouter_impl
    call(
        torch.view_as_real(Gprime).contiguous(),
        aberrations.contiguous(),
        _sin_cos_rot(rotation, Gprime.device),
        _contiguous_memo(Qx),
        _contiguous_memo(Qy),
        _contiguous_memo(Kx),
        _contiguous_memo(Ky),
        A.contiguous(),
        out_real,
        float(semiconvergence_angle),
        1e-3,
        float(wavelength),
        bool(defocus_only),
    )
    return out
