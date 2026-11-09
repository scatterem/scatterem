"""The multislice slice loop in L2-sized blocks, replayed as one CUDA graph.

Two flavours: the Warp fused-transmit flavour (1.39-1.41x on
``r3_simulator/medium/simulate``) and the torch flavour (the library
default, 1.39-1.41x on ``r3_simulator/medium/simulate --no-warp``).

The dense loop of :class:`~scatterem.nn.scan_exit_wave.MultisliceScanOp` makes
several DRAM passes per slice over a ``(1, K, ph, pw)`` stack whose positions
are coupled by nothing.  Running ``block`` positions through every slice while
they sit in L2 removes most of that traffic (77.9 -> 53.0 ms device on
``r3_simulator/medium``).  Done eagerly, the extra ``(block, slice)`` iterations
cost ~160 us of host issue each; recording ONE block's chain as a CUDA graph
and replaying it per block removes that cost and keeps the capture cheap.

Bit-identity: the graph body is the dense loop itself
(:func:`~scatterem.nn.ptychography.multislice_transmit_slice` per slice) on the
same wave layout -- a stride-0 ``expand`` of one probe is reproduced as an
``expand`` of a static probe, so slice 0 takes the same kernel as the dense loop.
Every op but the FFTs is elementwise per position, and the FFTs are gated by
:func:`cufft_batch_bit_invariant`.  Selected by the ``multislice_l2_graph`` and
``gather_l2_graph`` backend rows; their predicates are the input gates below.
Inside either graph, row ``gather_cmul_sep`` computes each slice's gather from
two small tables of the verified box-linear index.

The helpers here (:func:`cufft_batch_bit_invariant`, :func:`l2_power_block`,
:func:`env_key`, :func:`capture_graph`, :func:`memoised_graph`) are shared with
the PRISM S-matrix build (:mod:`._prism_s_build_graph`).
"""

from __future__ import annotations

import contextlib
import os
import weakref
from collections.abc import Callable

import torch
from torch import Tensor

from scatterem.nn._backends import GATHER_CMUL, SEPARABLE_GATHER
from scatterem.nn._backends_graphs import (
    GATHER_DEFER_NORM,
    GATHER_L2_GRAPH,
    MULTISLICE_L2_GRAPH,
)
from scatterem.nn.functional._forward_ad import carries_tangent
from scatterem.nn.modules import Propagator
from scatterem.nn.ptychography import multislice_transmit_slice
from scatterem.nn.ptychography._gather_per_slice import _gather_cmul_fn
from scatterem.physics import _warp_interop as wi
from scatterem.physics._aten_cmul_order import aten_cmul_imag_swapped
from scatterem.utils.env import resolve_backend, switched_on

__all__ = [
    "L2_GRAPH_STATS",
    "capture_graph",
    "cufft_batch_bit_invariant",
    "env_key",
    "gather_l2_graph_supported",
    "l2_blocked_slice_loop",
    "l2_power_block",
    "memoised_graph",
    "multislice_l2_graph_supported",
]

#: Reach of the blocked route (both flavours): calls served, graphs captured, and
#: calls declined after the row accepted them (cuFFT batch-variant or a capture
#: that raised).
L2_GRAPH_STATS = {"calls": 0, "captures": 0, "declined": 0}

#: ``(device, dtype, ph, pw, block, n) -> bool``: does cuFFT return the same bits at
#: batch ``block`` (and the remainder ``n % block``) as at the full batch ``n``?
_CUFFT_BATCH_EXACT: dict = {}

#: operator -> ``(key, _SliceLoopGraph | None)``.  Held here rather than on the
#: module so ``copy.deepcopy`` / ``state_dict`` of a model never meet a CUDA graph.
_L2_GRAPHS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def cufft_batch_bit_invariant(waves: Tensor, block: int) -> bool:
    """Ask, once per shape, whether blocking the batch moves a bit of either FFT.

    cuFFT chooses its kernel partly from the batch count, so a blocked walk can
    land on a different algorithm (a 128^2 grid does on sm_86).  Same test as
    ``simulation/_wave/exit_walk._cufft_batch_bit_invariant`` (which ``nn`` may
    not import): both transforms, both batch counts the walk uses.

    Parameters
    ----------
    waves : torch.Tensor
        ``(M, n, ph, pw)`` complex waves, possibly a stride-0 ``expand``; it is
        materialised only on a memo miss.
    block : int
        The block batch count.

    Returns
    -------
    bool
        Whether batch ``block`` and ``n % block`` give the full batch's bits.
    """
    n = int(waves.shape[0] * waves.shape[1])
    key = (waves.device.index, waves.dtype, *waves.shape[-2:], int(block), n)
    verdict = _CUFFT_BATCH_EXACT.get(key)
    if verdict is None:
        flat = waves.reshape(n, *waves.shape[-2:])
        batches = [block] + ([n % block] if n % block else [])
        verdict = True
        for fn, kw in ((torch.fft.fft2, {}), (torch.fft.ifft2, {"norm": "forward"})):
            full = fn(flat, dim=(-2, -1), **kw)
            for b in batches:
                same = torch.equal(fn(flat[:b], dim=(-2, -1), **kw), full[:b])
                verdict = verdict and bool(same)
            del full
        _CUFFT_BATCH_EXACT[key] = verdict
    return verdict


def l2_power_block(device: torch.device, n: int, per_item: int) -> int:
    """The largest power of two ``b < n`` whose ``b * per_item`` bytes fit L2.

    Returns 0 when the dense stack (``n * per_item``) already fits 3/4 of L2 --
    there is no traffic to recover -- or when the device reports no L2.
    """
    l2 = getattr(torch.cuda.get_device_properties(device), "L2_cache_size", 0)
    if l2 <= 0 or n * per_item <= 0.75 * l2:
        return 0
    block = 1
    while 2 * block * per_item <= l2 and 2 * block < n:
        block *= 2
    return block


def env_key() -> tuple:
    """The ``SCATTEREM_*`` environment, as part of a graph's memo key.

    A captured graph freezes every kernel choice the recorded code made from the
    environment, so a change of any of it must recapture.
    """
    return tuple(sorted(kv for kv in os.environ.items() if kv[0][:10] == "SCATTEREM_"))


def _warp_scope(stream: torch.cuda.Stream, warp: bool):
    """Route Warp launches onto ``stream`` (else they miss the capture).

    ``sync_enter=False``: the entered stream must not wait on Warp's previous,
    non-capturing stream, which would invalidate the capture.
    """
    if not warp:
        return contextlib.nullcontext()

    from scatterem.utils.warp.launch import ensure_warp_initialized

    ensure_warp_initialized()
    return wi.warp_capture_stream(stream)


def capture_graph(
    body: Callable[[], Tensor], device: torch.device, *, warp: bool = False
) -> tuple[torch.cuda.CUDAGraph, Tensor]:
    """Warm ``body`` up off-capture on a side stream, then record it once.

    The warm-up loads every cuFFT plan, Triton kernel and Warp module, which a
    capture cannot.  Returns the graph and the static output of the recorded call.
    """
    side = torch.cuda.Stream(device=device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side), _warp_scope(side, warp):
        body()
    torch.cuda.current_stream(device).wait_stream(side)
    aten_cmul_imag_swapped(device)  # prime the FMA order off-capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side), _warp_scope(side, warp):
        out = body()
    return graph, out


def memoised_graph(memo, owner, key: tuple, build: Callable, stats: dict):
    """The graph memoised for ``owner`` under ``key``, capturing it on a miss.

    A capture that raises is remembered as ``None`` for that key and not retried;
    ``None`` is returned (and counted as declined) so the caller runs dense.
    """
    state = memo.get(owner)
    if state is None or state[0] != key:
        memo.pop(owner, None)  # release the old graph before capturing
        try:
            graph = build()
        except Exception:  # noqa: BLE001 - any capture failure -> dense loop
            graph = None
        memo[owner] = state = (key, graph)
        if graph is not None:
            stats["captures"] += 1
    if state[1] is None:
        stats["declined"] += 1
        return None
    stats["calls"] += 1
    return state[1]


def _slice_loop(Ts, waves: Tensor, propagator, index: Tensor, fused: bool) -> Tensor:
    """``MultisliceScanOp``'s slice loop, verbatim, before the exit aperture."""
    NZ = len(Ts)
    for z in range(NZ):
        waves = multislice_transmit_slice(
            Ts[z],
            waves,
            propagator if z < NZ - 1 else None,
            inverse_index=index if fused else None,
            gather_index=None if fused else index,
        )
    return waves


def _sep_slice_loop(Ts, waves: Tensor, propagator, index, fused, tables, defer=False):
    """:func:`_slice_loop` with each slice's transmit as ``gather_cmul_sep``.

    On the torch route and on the Warp route, the gather is computed
    from the row and column tables ``tables = (R, C)`` of the verified box-linear
    index, and the product is rounded as the dense loop's kernel rounds it (ATen's
    on the torch route, the Warp ``fused_transmit`` kernel's on the Warp route).
    A slice the kernel declines (the stride-0 first slice of an ``expand``) takes
    :func:`multislice_transmit_slice` with the full index, as the dense loop does.
    With ``defer`` (row ``gather_defer_norm``) the next slice takes the
    propagator's ``1/N``, in the kernel or as ``* inv_n`` if it declines.  Returns the waves and whether any slice read the full index.
    """
    from scatterem.nn.functional._gather_cmul_triton import gather_cmul_sep

    NZ = len(Ts)
    R, C = tables
    used, inv_n = False, 1.0
    for z in range(NZ):
        p = propagator if z < NZ - 1 else None
        s = None if inv_n == 1.0 else inv_n
        out = gather_cmul_sep(Ts[z], R, C, waves, warp_cmul=fused, scale=s)
        if out is None:
            used = True
            waves = waves if s is None else waves * s  # the deferred pass, unfused
            kw = {"inverse_index": index} if fused else {"gather_index": index}
            out = multislice_transmit_slice(Ts[z], waves, None, **kw)
        inv_n = 1.0
        if p is None or not defer:
            waves = out if p is None else p(out)
        else:
            waves, inv_n = p(out, defer_norm=True)
    return waves, used


def _separable_tables(index: Tensor, fused: bool):
    """``(R, C)`` for :func:`_sep_slice_loop`, or ``None`` (row ``gather_cmul_sep``).

    On the Warp route the Warp kernel's FMA order is asked here, before any
    capture, and an unknown order keeps the full index.
    """
    if not _gather_cmul_fn():
        return None
    from scatterem.nn.functional._gather_cmul_triton import (
        separable_tables,
        warp_transmit_imag_swapped,
    )

    if fused and warp_transmit_imag_swapped(index.device) is None:
        return None
    return separable_tables(index)


def _block_size(T: Tensor, waves: Tensor, index: Tensor, propagator) -> int:
    """Positions per block, or 0 to keep the dense loop (both flavours).

    CUDA; a complex64 ``(NZ, U)`` stack with ``NZ >= 2``; one probe mode; a
    contiguous ``int32`` ``(K, ph, pw)`` index; waves contiguous or a stride-0
    ``expand`` of one contiguous probe; a plain :class:`Propagator` (whose ramp
    the graph holds by address); no autograd or dual; outside ``torch.compile``
    and an enclosing capture.  The block holds the complex64 wave plane plus the int32
    index plane of each position in L2.
    """
    if not (T.is_cuda and waves.is_cuda and index.is_cuda):
        return 0
    if T.dim() != 2 or waves.dim() != 4 or index.dim() != 3 or T.shape[0] < 2:
        return 0
    if {T.dtype, waves.dtype} != {torch.complex64} or index.dtype != torch.int32:
        return 0
    if waves.shape[0] != 1 or tuple(waves.shape[1:]) != tuple(index.shape):
        return 0
    # A replay copies primal values only, so it would drop a forward-mode tangent.
    if type(propagator) is not Propagator or carries_tangent(T, waves):
        return 0
    if not (T.is_contiguous() and index.is_contiguous()):
        return 0
    expand = waves.stride(1) == 0 and waves[:, :1].is_contiguous()
    if not (waves.is_contiguous() or expand) or 2 * waves.numel() >= 2**31:
        return 0
    if torch.is_grad_enabled() and (T.requires_grad or waves.requires_grad):
        return 0
    if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        return 0
    K, ph, pw = index.shape
    block = l2_power_block(
        waves.device, K, ph * pw * (waves.element_size() + index.element_size())
    )
    return block if block >= 2 else 0


def multislice_l2_graph_supported(
    T: Tensor, waves: Tensor, index: Tensor, propagator
) -> bool:
    """The ``multislice_l2_graph`` row's gate (Warp fused-transmit flavour).

    Parameters
    ----------
    T : torch.Tensor
        ``(NZ, U)`` per-unique-pixel transmission stack.
    waves : torch.Tensor
        ``(1, K, ph, pw)`` incident waves.
    index : torch.Tensor
        ``(K, ph, pw)`` int32 inverse index.
    propagator : Propagator
        The inter-slice propagator.

    Returns
    -------
    bool
        Whether a block of at least two positions applies.
    """
    return _block_size(T, waves, index, propagator) > 0


def gather_l2_graph_supported(
    T: Tensor, waves: Tensor, index: Tensor, propagator
) -> bool:
    """The ``gather_l2_graph`` row's gate (torch-route flavour).

    As :func:`multislice_l2_graph_supported`, and only where the dense loop runs
    the fused Triton ``gather_cmul`` on every slice after the first (the route
    that was benchmarked) and the opt-in separable gather is off (its first call pays a
    ``torch.compile`` and its proof reads the host, neither of which a capture
    can do).

    Parameters
    ----------
    T, waves, index, propagator
        As :func:`multislice_l2_graph_supported`.

    Returns
    -------
    bool
        Whether the blocked route applies.
    """
    if switched_on(SEPARABLE_GATHER, "separable") or not _gather_cmul_fn():
        return False
    if resolve_backend(GATHER_CMUL) != "triton":
        return False
    return _block_size(T, waves, index, propagator) > 0


class _SliceLoopGraph:
    """The whole slice loop for ``block`` positions, recorded once as a CUDA graph.

    Per block: copy the block's waves and index rows into the static inputs,
    replay, copy the static output into the caller's fresh result.  ``T`` is held
    by address, so it is copied in once per call.  A ``K % block`` remainder runs
    the dense loop eagerly.

    With ``tables`` = ``(R, C)`` from :func:`_separable_tables` the body is
    :func:`_sep_slice_loop`: the block's table rows are copied in
    as well, and the index rows only if a recorded slice reads them.
    """

    def __init__(self, T, waves, index, propagator, block, fused, tables, defer):
        self.block, self.fused = block, fused
        self.sep, self.needs_index = tables is not None, True
        if tables is not None:
            self.static_R = tables[0][:block].clone()
            self.static_C = tables[1][:block].clone()
        self.expand = waves.stride(1) == 0
        self.ramp = propagator.prop  # pinned: the graph reads it by address
        self.static_T = T.clone()
        if self.expand:
            self.static_probe = waves[:, :1].clone()
            self.static_w = self.static_probe.expand(
                waves.shape[0], block, *waves.shape[2:]
            )
        else:
            self.static_w = waves[:, :block].clone()
        self.static_idx = index[:block].clone()
        Ts = torch.unbind(self.static_T, 0)

        def body():
            if not self.sep:
                return _slice_loop(
                    Ts, self.static_w, propagator, self.static_idx, fused
                )
            tabs = (self.static_R, self.static_C)
            out, self.needs_index = _sep_slice_loop(
                Ts, self.static_w, propagator, self.static_idx, fused, tabs, defer
            )
            return out

        self.graph, self.static_out = capture_graph(body, waves.device, warp=fused)

    def run(
        self, T: Tensor, waves: Tensor, index: Tensor, propagator, tables=None
    ) -> Tensor:
        b = self.block
        K = waves.shape[1]
        out = torch.empty(waves.shape, dtype=waves.dtype, device=waves.device)
        self.static_T.copy_(T)
        if self.expand:
            self.static_probe.copy_(waves[:, :1])
        n_full = K - K % b
        for k0 in range(0, n_full, b):
            if not self.expand:
                self.static_w.copy_(waves[:, k0 : k0 + b])
            if self.needs_index:
                self.static_idx.copy_(index[k0 : k0 + b])
            if tables is not None:
                self.static_R.copy_(tables[0][k0 : k0 + b])
                self.static_C.copy_(tables[1][k0 : k0 + b])
            self.graph.replay()
            out[:, k0 : k0 + b].copy_(self.static_out)
        if n_full < K:
            # The remainder has its own shapes; run its (short) loop eagerly.
            tail = _slice_loop(
                torch.unbind(T, 0),
                waves[:, n_full:],
                propagator,
                index[n_full:],
                self.fused,
            )
            out[:, n_full:].copy_(tail)
        return out


def l2_blocked_slice_loop(
    op, T: Tensor, waves: Tensor, propagator, inverse_index, gather_index
) -> Tensor | None:
    """The slice loop through :class:`_SliceLoopGraph`, or ``None`` to run dense.

    The graph is memoised per operator (the simulator memoises its model, so a
    repeat ``simulate`` replays it) and keyed on everything it holds by address
    or decided at capture: flavour, shapes, wave layout, the propagator and its
    phase ramp, and the ``SCATTEREM_*`` environment.

    Parameters
    ----------
    op : MultisliceScanOp
        The operator, the memo's owner.
    T, waves, propagator
        As :meth:`MultisliceScanOp.forward`.
    inverse_index, gather_index : torch.Tensor or None
        Exactly one is given: the Warp fused-transmit or the torch-route index.

    Returns
    -------
    torch.Tensor or None
        The exit waves before the exit aperture, or ``None``.
    """
    # Decided before the registry is asked: a traced or training forward (the
    # reconstruction path) must not run the registry's host Python at all.
    if torch.compiler.is_compiling() or not T.is_cuda:
        return None
    if torch.is_grad_enabled() and (T.requires_grad or waves.requires_grad):
        return None
    fused = inverse_index is not None
    index = inverse_index if fused else gather_index
    if index is None:
        return None
    row = MULTISLICE_L2_GRAPH if fused else GATHER_L2_GRAPH
    if resolve_backend(row, None, T, waves, index, propagator) != "graph":
        return None
    block = _block_size(T, waves, index, propagator)
    if not cufft_batch_bit_invariant(waves, block):
        L2_GRAPH_STATS["declined"] += 1
        return None
    tables = _separable_tables(index, fused)
    pick = resolve_backend(GATHER_DEFER_NORM, None, waves) if tables else "pass"
    defer = pick == "deferred"
    ramp = propagator.prop
    key = (
        fused,
        waves.device.index,
        tuple(T.shape),
        tuple(index.shape[1:]),
        block,
        waves.stride(1) == 0,
        tables is not None,
        defer,
        id(propagator),
        (ramp.data_ptr(), ramp._version),
        env_key(),
    )
    graph = memoised_graph(
        _L2_GRAPHS,
        op,
        key,
        lambda: _SliceLoopGraph(
            T, waves, index, propagator, block, fused, tables, defer
        ),
        L2_GRAPH_STATS,
    )
    return None if graph is None else graph.run(T, waves, index, propagator, tables)
