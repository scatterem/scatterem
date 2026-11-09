"""The PRISM S-matrix build in L2-sized blocks of beams, replayed as one CUDA graph.

Measured: 1.20x on ``r3_simulator/small/prism_simulate``; the same graph on the
beam-chunked branch, 1.20x on ``prism_simulate_chunked``; the parent waves
generated per block inside the graph, 1.077x chunked / 1.058x single-pass on top;
and a two-beam block admitted at 512^2, 1.075x on
``r3_simulator/medium/prism_simulate``.

The dense build walks a ``(Bp, MY, MX)`` stack (1.17 GB at
``r3_simulator/small``) through every slice's transmit + ``fft2`` + ramp +
``ifft2`` at the DRAM roof, yet nothing in the build couples beams.  Running
``block`` beams through all slices while they sit in L2 removes most of that
traffic -- the multislice recipe of :mod:`._l2_block_graph` with beams in place
of positions.  Every op but the FFTs is elementwise per beam and the FFTs are
gated by :func:`._l2_block_graph.cufft_batch_bit_invariant`, so the S-matrix is
bit-identical to the dense build.  Taken only for a complex64, grad-free build
without an exit antialias aperture; selected by the
``prism_build_l2_graph`` and ``prism_build_gen_graph`` backend rows.
"""

from __future__ import annotations

import weakref

import torch
from torch import Tensor

from scatterem.nn._backends_graphs import PRISM_BUILD_GEN_GRAPH, PRISM_BUILD_L2_GRAPH
from scatterem.nn.modules import Propagator, _bcast_cmul_
from scatterem.nn.scan_exit_wave._l2_block_graph import (
    capture_graph,
    cufft_batch_bit_invariant,
    env_key,
    l2_power_block,
    memoised_graph,
)
from scatterem.utils.env import resolve_backend

__all__ = [
    "S_BUILD_GEN_GRAPH_STATS",
    "S_BUILD_L2_GRAPH_STATS",
    "s_build_gen_graph_supported",
    "s_build_l2_blocked",
    "s_build_l2_generated",
    "s_build_l2_graph_supported",
]

#: Reach of the block graph (:class:`_SBuildL2BlockGraph`): builds (or chunks)
#: served, graphs captured, and builds declined after the row accepted them.
S_BUILD_L2_GRAPH_STATS = {"calls": 0, "captures": 0, "declined": 0}

#: Reach of the generating graph (:class:`_SBuildGenL2BlockGraph`), as above.
S_BUILD_GEN_GRAPH_STATS = {"calls": 0, "captures": 0, "declined": 0}

#: operator -> ``(key, graph | None)``; off-module so ``copy.deepcopy`` /
#: ``state_dict`` of a model never meet a CUDA graph.
_S_BUILD_GRAPHS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_S_GEN_GRAPHS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

#: The generating graph keeps the minimum block of the plain graph: at 512^2 only b2 fits L2 and
#: a refused generated build would have allocated a ~20 GB ``S`` for nothing.
_GEN_MIN_BLOCK = 8


def _s_build_chain(Ts, x: Tensor, propagator) -> Tensor:
    """The single-pass build's slice loop (``_build_s_matrix``)."""
    NZ = len(Ts)
    inv_n = 1.0
    for z in range(NZ):
        if inv_n == 1.0:
            x = Ts[z].unsqueeze(0) * x
        elif not _bcast_cmul_(x, Ts[z], True, scale=inv_n):
            x = Ts[z].unsqueeze(0) * (x * inv_n)
        if z < NZ - 1:
            x, inv_n = propagator(x, defer_norm=True)
    return x


def _op_takes(op) -> bool:
    """Operator-level refusals: fp16 build, an exit aperture."""
    return (
        op._build_dtype() == torch.complex64
        and op.exit_antialias_aperture is None
    )


def _operands_ok(T: Tensor, propagator) -> bool:
    """CUDA contiguous complex64 ``(NZ >= 2, MY, MX)`` ``T``, a plain propagator.

    A plain :class:`Propagator` only: a subclass may compute its kernel per call,
    which the graph would freeze.  No autograd, outside ``torch.compile`` and
    outside an enclosing capture.
    """
    if not T.is_cuda or T.dim() != 3 or T.shape[0] < 2:
        return False
    if T.dtype != torch.complex64 or not T.is_contiguous():
        return False
    if type(propagator) is not Propagator:
        return False
    if torch.is_grad_enabled() and T.requires_grad:
        return False
    return not (
        torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing()
    )


def _s_build_l2_block(T: Tensor, waves: Tensor, propagator) -> int:
    """Beams per block, or 0 to keep the dense loop.

    :func:`_operands_ok`, and contiguous grad-free complex64 waves.  The block is
    the largest power of two whose beam planes fit L2, at
    least 2 (at 192^2 the law picks b16, at 512^2 b2, measured 1.127x of
    the build).
    """
    if not (_operands_ok(T, propagator) and waves.is_cuda):
        return 0
    if waves.dtype != torch.complex64 or not waves.is_contiguous():
        return 0
    if torch.is_grad_enabled() and waves.requires_grad:
        return 0
    block = l2_power_block(waves.device, waves.shape[0], waves[0].numel() * 8)
    return block if block >= 2 else 0


def _gen_block(op, T: Tensor, cols: int | None = None) -> int:
    """The generating graph's block for ``op``'s ``(Bp, MY, MX)`` output, or 0.

    ``cols`` (the dense build's beams per chunk) caps the block: the L2 law
    alone picks 256 beams at 128^2 on a 48 MB L2 (L4), where a caller's
    ``max_build_bytes`` asked for 32, and the build transient then outgrew the
    chunk bound.  A capped
    block is equal to the chunk, not to the power of two below it.  Thus each
    block is one chunk of the dense build, and the eager remainder is its last
    chunk.  cuFFT then runs the same batch counts as the dense build.
    """
    MY, MX = int(op.s_meta.M[0]), int(op.s_meta.M[1])
    block = l2_power_block(T.device, int(op.s_meta.Bp), MY * MX * 8)
    if cols is not None:
        block = min(block, max(1, int(cols)))
    return block if block >= _GEN_MIN_BLOCK else 0


def s_build_l2_graph_supported(op, T: Tensor, waves: Tensor, propagator) -> bool:
    """The ``prism_build_l2_graph`` row's gate.

    Parameters
    ----------
    op : PrismScanOp
        The operator building the S-matrix.
    T : torch.Tensor
        ``(NZ, MY, MX)`` transmission stack.
    waves : torch.Tensor
        ``(n, MY, MX)`` fresh parent waves (the whole stack or one chunk).
    propagator : Propagator
        The inter-slice propagator.

    Returns
    -------
    bool
        Whether a block of at least two beams applies.
    """
    return _op_takes(op) and _s_build_l2_block(T, waves, propagator) > 0


def _key(T: Tensor, block: int, propagator, *extra) -> tuple:
    ramp = propagator.prop
    return (
        T.device.index,
        tuple(T.shape),
        block,
        *extra,
        id(propagator),
        (ramp.data_ptr(), ramp._version),
        env_key(),
    )


class _SBuildL2BlockGraph:
    """The single-pass S-matrix build for ``block`` beams, recorded once.

    Per block: copy the block's parent waves into the static input, replay, copy
    the static output back over the same rows (each block's input is consumed
    before its rows are overwritten, so the build needs no second stack).  ``T``
    is copied in once per call; a ``n % block`` remainder runs eagerly.
    """

    def __init__(self, T: Tensor, waves: Tensor, propagator, block: int):
        self.block = block
        self.ramp = propagator.prop  # pinned: the graph reads it by address
        self.static_T = T.clone()
        self.static_w = waves[:block].clone()
        Ts = torch.unbind(self.static_T, 0)
        self.graph, self.static_out = capture_graph(
            lambda: _s_build_chain(Ts, self.static_w.clone(), propagator),
            waves.device,
        )

    def run(self, T: Tensor, waves: Tensor, propagator) -> Tensor:
        """Build in place over ``waves`` (fresh parent waves) and return it."""
        b = self.block
        n = waves.shape[0]
        self.static_T.copy_(T)
        n_full = n - n % b
        for k0 in range(0, n_full, b):
            self.static_w.copy_(waves[k0 : k0 + b])
            self.graph.replay()
            waves[k0 : k0 + b].copy_(self.static_out)
        if n_full < n:
            tail = _s_build_chain(torch.unbind(T, 0), waves[n_full:], propagator)
            waves[n_full:].copy_(tail)
        return waves


def s_build_l2_blocked(op, T: Tensor, waves: Tensor, propagator) -> Tensor | None:
    """The build of ``waves`` through :class:`_SBuildL2BlockGraph`, or ``None``.

    Serves the single-pass branch (all parent waves) and each chunk of the
    beam-chunked branch (one graph for every chunk, since its key holds no
    beam count).  The graph is memoised per operator and keyed on shapes, the
    propagator and its phase ramp, and the ``SCATTEREM_*`` environment.

    Parameters
    ----------
    op : PrismScanOp
        The operator, the memo's owner.
    T : torch.Tensor
        ``(NZ, MY, MX)`` transmission stack.
    waves : torch.Tensor
        ``(n, MY, MX)`` fresh parent waves; overwritten with the result.
    propagator : Propagator
        The inter-slice propagator.

    Returns
    -------
    torch.Tensor or None
        ``waves`` holding the raw S-matrix rows, or ``None`` to build dense.
    """
    if torch.compiler.is_compiling() or not T.is_cuda:
        return None
    if torch.is_grad_enabled() and T.requires_grad:
        return None
    if resolve_backend(PRISM_BUILD_L2_GRAPH, None, op, T, waves, propagator) != (
        "graph"
    ):
        return None
    block = _s_build_l2_block(T, waves, propagator)
    if not cufft_batch_bit_invariant(waves.unsqueeze(0), block):
        S_BUILD_L2_GRAPH_STATS["declined"] += 1
        return None
    graph = memoised_graph(
        _S_BUILD_GRAPHS,
        op,
        _key(T, block, propagator),
        lambda: _SBuildL2BlockGraph(T, waves, propagator, block),
        S_BUILD_L2_GRAPH_STATS,
    )
    return None if graph is None else graph.run(T, waves, propagator)


class _SBuildGenL2BlockGraph:
    """:class:`_SBuildL2BlockGraph` with each block's parent waves generated in it.

    The block graph still needs the whole parent-wave stack first:
    ``_initial_parent_waves`` writes a zero stack, sets one pixel per beam, runs
    a full-stack ortho ``ifft2`` and an ``fscale`` pass, all at the DRAM roof.
    Here the block's beam pixels are copied into a static index pair and the
    body does the same ``zeros`` / ``index_put_`` / ``ifft2(norm="ortho")`` /
    ``*= fscale`` on ``block`` beams while they sit in L2, then the slice chain.
    Same ops in the same order per beam, so the S-matrix is bit-identical.
    """

    def __init__(self, T, iy, ix, shape, fscale, propagator, block: int):
        self.block, self.shape, self.fscale = block, shape, fscale
        self.ramp = propagator.prop  # pinned: the graph reads it by address
        self.static_T = T.clone()
        self.static_iy = iy[:block].clone()
        self.static_ix = ix[:block].clone()
        self.rows = torch.arange(block, device=T.device)
        # A device-resident 1: ``psi[...] = 1.0`` copies a CPU scalar, which a
        # capture refuses.  Same value, so ``psi`` is the same bits.
        self.one = torch.ones((), dtype=torch.complex64, device=T.device)
        Ts = torch.unbind(self.static_T, 0)
        self.graph, self.static_out = capture_graph(
            lambda: self._body(Ts, propagator), T.device
        )

    def _body(self, Ts, propagator) -> Tensor:
        """``_initial_parent_waves`` for the static block, then the slice chain."""
        psi = torch.zeros(
            (self.block, *self.shape), dtype=torch.complex64, device=self.rows.device
        )
        psi.index_put_((self.rows, self.static_iy, self.static_ix), self.one)
        w = torch.fft.ifft2(psi, norm="ortho")
        del psi
        if self.fscale is not None:
            w *= self.fscale
        return _s_build_chain(Ts, w, propagator)

    def run(self, T: Tensor, iy: Tensor, ix: Tensor, S: Tensor) -> int:
        """Write every full block of ``S``; return the first row left unbuilt."""
        b = self.block
        n_full = S.shape[0] - S.shape[0] % b
        self.static_T.copy_(T)
        for k0 in range(0, n_full, b):
            self.static_iy.copy_(iy[k0 : k0 + b])
            self.static_ix.copy_(ix[k0 : k0 + b])
            self.graph.replay()
            S[k0 : k0 + b].copy_(self.static_out)
        return n_full


def s_build_gen_graph_supported(
    op, T: Tensor, propagator, cols: int | None = None
) -> bool:
    """The ``prism_build_gen_graph`` row's gate, decided before ``S`` exists.

    As :func:`s_build_l2_graph_supported` asked of the ``(Bp, MY, MX)`` output,
    with a block of at least 8 beams after the cap at ``cols``: at 512^2 a
    refused build would otherwise have allocated a ~20 GB transient that the
    chunked branch exists to avoid.

    Parameters
    ----------
    op : PrismScanOp
        The operator building the S-matrix.
    T : torch.Tensor
        ``(NZ, MY, MX)`` transmission stack.
    propagator : Propagator
        The inter-slice propagator.
    cols : int or None, optional
        The dense build's beams per chunk, which caps the block; ``None`` for
        the L2 law alone.

    Returns
    -------
    bool
        Whether the generating graph applies.
    """
    return _op_takes(op) and _operands_ok(T, propagator) and _gen_block(op, T, cols) > 0


def s_build_l2_generated(op, T: Tensor, propagator, cols: int) -> Tensor | None:
    """The whole build through :class:`_SBuildGenL2BlockGraph`, or ``None``.

    Runs ahead of both branches of ``_build_s_matrix`` (they build the same
    bits).  The cuFFT batch check is asked once per graph key on the real parent
    waves of the dense build's first batch -- all ``Bp`` beams on the
    single-pass branch, one chunk of ``cols`` on the chunked one, so the check's
    transient stays within the chunk bound that branch was asked to respect.
    The ``Bp % block`` remainder is generated and built eagerly.  A block
    capped at ``cols`` is equal to ``cols``, so this remainder is the last
    chunk of the dense build.

    Parameters
    ----------
    op : PrismScanOp
        The operator, the memo's owner.
    T : torch.Tensor
        ``(NZ, MY, MX)`` transmission stack.
    propagator : Propagator
        The inter-slice propagator.
    cols : int
        The dense build's beams per chunk (``>= Bp``: single pass).

    Returns
    -------
    torch.Tensor or None
        The raw ``(Bp, MY, MX)`` S-matrix, or ``None`` to try the other routes.
    """
    if torch.compiler.is_compiling() or not T.is_cuda:
        return None
    if torch.is_grad_enabled() and T.requires_grad:
        return None
    if resolve_backend(PRISM_BUILD_GEN_GRAPH, None, op, T, propagator, cols) != (
        "graph"
    ):
        return None
    MY, MX = int(op.s_meta.M[0]), int(op.s_meta.M[1])
    Bp = int(op.s_meta.Bp)
    block = _gen_block(op, T, cols)
    if not block:  # the row checks the same block; this guards a forced row
        S_BUILD_GEN_GRAPH_STATS["declined"] += 1
        return None
    coords = op.s_meta.parent_beams_coords  # as ``_initial_parent_waves``
    iy = coords[:, 0].long() % MY
    ix = coords[:, 1].long() % MX
    fscale = float(op.fy * op.fx) ** 0.5 if (op.fy != 1 or op.fx != 1) else None

    def build():
        waves = op._initial_parent_waves(0, min(Bp, max(1, int(cols))))
        exact = cufft_batch_bit_invariant(waves.unsqueeze(0), block)
        del waves
        if not exact:
            return None
        return _SBuildGenL2BlockGraph(T, iy, ix, (MY, MX), fscale, propagator, block)

    graph = memoised_graph(
        _S_GEN_GRAPHS,
        op,
        _key(T, block, propagator, Bp, min(Bp, int(cols)), fscale),
        build,
        S_BUILD_GEN_GRAPH_STATS,
    )
    if graph is None:
        return None
    S = torch.empty((Bp, MY, MX), dtype=torch.complex64, device=op.s_meta.device)
    n_full = graph.run(T, iy, ix, S)
    if n_full < Bp:
        tail = op._initial_parent_waves(n_full, Bp)
        S[n_full:] = _s_build_chain(torch.unbind(T, 0), tail, propagator)
    return S
