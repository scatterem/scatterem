"""The tiled ``grad_Sdt`` gather issued from a reusable ``wp.Launch``.

At ``r2_prism_project/small/project_backward`` the tiled gather is one ~0.5 ms
kernel behind ~0.55 ms of host issue.  ``wp.launch`` packs a fresh parameter block
on every call; this module keeps one ``wp.Launch`` per (kernel, device, thread)
and rebinds every slot before every launch.  The kernel, the grid and the
arguments are the same as on the ``wp.launch`` route, so the result is
bit-identical.  The ``prism_grad_sdt_launch`` backend row selects the route
(:func:`grad_sdt_recorded_supported` is its predicate).

The command is built with ``wp.Launch(kernel, device)`` directly, not with
``wp.launch(record_cmd=True)``.  A recorded command keeps its record-time
``wp.array`` objects in ``fwd_args``, and those would pin the first call's
tensors for the life of the memo (the whole gradient, 1 GiB at ``r2/medium``).
Built directly, the parameter block holds only ctypes descriptors.  The launcher
clears their tensor pins once the kernel is enqueued, so the memo keeps no
caller tensor alive.
"""

from __future__ import annotations

import threading

import torch
import warp as wp
from torch import Tensor

from scatterem.physics import _warp_interop as wi

__all__ = [
    "grad_sdt_recorded_supported",
    "launch_tiled_gather_recorded",
    "wrapped_positions",
]

#: ``(kernel, device, thread id) -> [wp.Launch, last dim]``.  One entry per
#: thread, so two threads never rebind the same parameter block.
_GRAD_SDT_CMD: dict = {}
#: ``(SY, SX, device) -> (1, 2)`` int32 divisor, so the wrapped positions are one
#: ``remainder`` instead of two ``remainder`` + ``stack`` + ``contiguous``.
_POS_DIVISOR: dict = {}
_CACHE_CAP = 8
#: The array ranks the tiled kernel declares: grad_out, basis_rs, pos_n, grad_Sdt.
_RANKS = (3, 4, 2, 3)


def grad_sdt_recorded_supported(positions: Tensor) -> bool:
    """True if the reusable command may issue the tiled ``grad_Sdt`` gather.

    The caller has already chosen the tiled gather (``Bp > 1``).  Here: CUDA
    ``(K, 2)`` int32 positions in contiguous memory (the cached divisor wraps
    them in one op), and no CUDA-graph capture in progress, so a capture records
    the plain ``wp.launch`` it recorded before.
    """
    return (
        positions.is_cuda
        and positions.dim() == 2
        and positions.dtype == torch.int32
        and positions.is_contiguous()
        and not torch.cuda.is_current_stream_capturing()
    )


def wrapped_positions(positions: Tensor, SY: int, SX: int) -> Tensor:
    """``positions mod (SY, SX)``, the same int32 values as the per-column spelling."""
    key = (int(SY), int(SX), positions.device)
    div = _POS_DIVISOR.get(key)
    if div is None:
        if len(_POS_DIVISOR) >= _CACHE_CAP:
            _POS_DIVISOR.clear()
        div = torch.tensor([[SY, SX]], dtype=positions.dtype, device=positions.device)
        _POS_DIVISOR[key] = div
    return torch.remainder(positions, div)


def _command(kernel, device) -> list:
    key = (kernel, str(device), threading.get_ident())
    entry = _GRAD_SDT_CMD.get(key)
    if entry is None:
        if len(_GRAD_SDT_CMD) >= _CACHE_CAP:
            _GRAD_SDT_CMD.clear()
        entry = [wp.Launch(kernel, device), None]
        _GRAD_SDT_CMD[key] = entry
    return entry


def launch_tiled_gather_recorded(
    kernel,
    dim: tuple[int, int, int],
    grad_out_wp,
    basis_rs_wp,
    pos_n: Tensor,
    scalars: tuple[int, ...],
    grad_Sdt: Tensor,
    device,
) -> bool:
    """Issue the tiled gather from the reusable command.

    Returns ``False`` (nothing launched) when an operand has a rank the kernel
    does not declare; the caller then takes the checking ``wp.launch`` route.
    """
    # ``requires_grad=False`` as on the ``wp.launch`` route.  At its default,
    # ``from_torch`` follows the tensor's flag and allocates a Warp grad buffer.
    cts = (
        grad_out_wp.__ctype__(),
        basis_rs_wp.__ctype__(),
        wi.from_torch(pos_n, dtype=wp.int32, requires_grad=False, return_ctype=True),
        wi.from_torch(grad_Sdt, dtype=wp.vec2f, requires_grad=False, return_ctype=True),
    )
    # ``from_torch`` checks dtype, inner shape and strides, but not the rank.
    if tuple(int(c.ndim) for c in cts) != _RANKS:
        return False
    entry = _command(kernel, device)
    cmd = entry[0]
    if entry[1] != dim:
        cmd.set_dim(dim)
        entry[1] = dim
    for i in range(3):
        cmd.set_param_at_index_from_ctype(i, cts[i])
    for i, v in enumerate(scalars):
        cmd.set_param_at_index_from_ctype(3 + i, v)
    cmd.set_param_at_index_from_ctype(3 + len(scalars), cts[3])
    cmd.launch()
    # ``from_torch(return_ctype=True)`` pins its tensor on the descriptor, and the
    # parameter block keeps the descriptor.  Without this, the last call's
    # gradient stays allocated after the caller drops it (+64 MiB).  The
    # kernel is already enqueued, which is when ``wp.launch`` lets its arrays go.
    for c in cts:
        c._ref = None
        # ``from_torch`` sets ``_gradref`` at run time; the stub does not declare it.
        c._gradref = None  # pyright: ignore[reportAttributeAccessIssue]
    return True
