"""CUDA-graph replay of the L2-blocked exit walk.

:func:`.exit_walk._slice_exit_wave` walks a wave stack block by block so every
intermediate stays L2-resident.  On ``r7_eels/large`` (256^2, complex64) a block
is 9 planes and its walk is ~30 launches, i.e. host-issue-bound; this module
captures one block's walk (plus its ``fft2`` when the caller folds the spectrum)
as a CUDA graph and replays it per full block.  Row ``eels_walk_graph``
(``scatterem.simulation._backends``), predicate :func:`walk_graph_supported`;
the remainder block keeps the eager walk.

Bit-exact: every replayed kernel is the eager block walk's kernel at the same
batch (``_cufft_batch_bit_invariant`` has verified that batch) on operands that
are copies of the live ones.  Each replay costs one extra block copy-in
(L2-resident).  Measured: 1.08x on ``r7_eels/large/conventional``
(10000.3 -> 9254.6 ms), own peak 4026.5 -> 4022.0 MiB, process peak +84.5 MiB
for the persistent graphs (a labelled trade).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from scatterem.physics._aten_cmul_order import aten_cmul_imag_swapped
from scatterem.physics.propagation import _scaled_propagator
from scatterem.simulation._backends import EELS_WALK_GRAPH
from scatterem.utils.env import resolve_backend

#: Reach of the replayed walk: ``_slice_exit_wave`` calls that took it, and
#: graphs captured.
WALK_GRAPH_STATS = {"calls": 0, "captures": 0}
#: Captured walks keyed by everything the captured launches depend on (see
#: :func:`walk_graph`), and the static operands they share.  Never evicted: a
#: dropped graph strands its private pool, so past ``_WALK_GRAPHS_MAX``
#: walks a new key walks eagerly.
_WALK_GRAPHS: dict = {}
_WALK_STATICS: dict = {}
_WALK_GRAPHS_MAX = 32


class _WalkStatics:
    """Static operands a captured walk reads.

    The transmissions, the SCALED propagator (what ``propagate`` multiplies by,
    via ``_scaled_propagator``'s memo, which is pre-seeded on ``kernel`` here) and
    one input block.  They are copies, loaded from the live operands on every
    call, because a graph reads fixed addresses: a caller builds a fresh
    propagator per call, and its scaled plane lives only as long as it does.
    """

    def __init__(self, block: Tensor, transmissions: Tensor, scaled: Tensor):
        self.block = torch.empty_like(block)
        self.T = torch.empty_like(transmissions, memory_format=torch.contiguous_format)
        self.kernel = torch.empty_like(scaled)  # never read: only its memo is
        self.scaled = torch.empty_like(scaled)
        n_pixels = int(block.shape[-2]) * int(block.shape[-1])
        self.kernel._scatterem_prop_scaled = (n_pixels, self.scaled)  # pyright: ignore[reportAttributeAccessIssue]

    def load(self, transmissions: Tensor, kernel: Tensor) -> None:
        """Copy this call's transmissions and scaled propagator in."""
        self.T.copy_(transmissions)
        n_pixels = int(self.block.shape[-2]) * int(self.block.shape[-1])
        self.scaled.copy_(_scaled_propagator(kernel, n_pixels))


class _WalkGraph:
    """One block's dense walk (+ its ``fft2`` when ``fold``) as a CUDA graph."""

    def __init__(self, statics, dense: Callable, start, unit_slices, fold):
        self.statics = statics
        self.dense = dense
        self.start = start
        self.unit = unit_slices
        self.fold = fold
        dev = statics.block.device
        side = torch.cuda.Stream(device=dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side):  # warm-up: cuFFT plans outside the capture
            self._body()
        torch.cuda.current_stream(dev).wait_stream(side)
        aten_cmul_imag_swapped(dev)  # prime the FMA order off-capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=side):
            self.out = self._body()
        WALK_GRAPH_STATS["captures"] += 1

    def _body(self) -> Tensor:
        st = self.statics
        r = self.dense(st.block, st.T, st.kernel, self.start, self.unit)
        return torch.fft.fft2(r, dim=(-2, -1)) if self.fold else r

    def run(self, flat: Tensor, transmissions: Tensor, kernel: Tensor, out: Tensor):
        """Walk every full block of ``flat`` into ``out``; return where the eager
        remainder starts.  ``out`` may be ``flat`` itself: each block is copied
        in before its result is copied back over it."""
        st = self.statics
        st.load(transmissions, kernel)
        B = int(st.block.shape[0])
        nf = int(flat.shape[0]) - int(flat.shape[0]) % B
        for i in range(0, nf, B):
            st.block.copy_(flat[i : i + B])
            self.graph.replay()
            out[i : i + B].copy_(self.out)
        WALK_GRAPH_STATS["calls"] += 1
        return nf


def walk_graph_supported(
    flat: Tensor, transmissions: Tensor, kernel: Tensor, start: int, block: int
) -> bool:
    """Whether a captured walk may replace the eager blocked walk of this call.

    Declined off CUDA, inside a capture or compile, when anything carries a graph, with
    fewer than two slices or two full blocks to walk, and unless the walk keeps
    ``flat``'s dtype (complex transmissions and propagator of ``flat``'s dtype),
    which is what lets the caller write the result back into ``flat``.
    """
    if not flat.is_cuda:  # CUDA graphs; ``_exit_block_planes`` can be overridden
        return False
    if torch.cuda.is_current_stream_capturing() or torch.compiler.is_compiling():
        return False
    if transmissions.requires_grad or kernel.requires_grad or flat.requires_grad:
        return False
    if not (flat.is_complex() and transmissions.dtype == flat.dtype == kernel.dtype):
        return False
    if transmissions.ndim != 3 or kernel.shape != flat.shape[-2:]:
        return False
    if transmissions.device != flat.device or kernel.device != flat.device:
        return False
    # One slice left is one multiply; one full block replays once.  Neither
    # repays a capture.
    return int(transmissions.shape[0]) - int(start) >= 2 and int(
        flat.shape[0]
    ) >= 2 * int(block)


def walk_graph(flat, transmissions, kernel, start, unit_slices, block, fold, dense):
    """The captured walk for this call, or ``None`` for the eager blocked walk.

    ``dense`` is :func:`.exit_walk._slice_exit_wave_dense` (passed in, so this
    module does not import its caller).  A new key past ``_WALK_GRAPHS_MAX``
    walks eagerly rather than evict.
    """
    args = (flat, transmissions, kernel, start, block)
    if resolve_backend(EELS_WALK_GRAPH, None, *args) != "graph":
        return None
    unit = None if unit_slices is None else tuple(bool(u) for u in unit_slices)
    plane = tuple(flat.shape[-2:])
    skey = (
        flat.device.index,
        flat.dtype,
        tuple(transmissions.shape),
        int(block),
        plane,
    )
    key = (*skey, int(start), unit, bool(fold))
    walk = _WALK_GRAPHS.get(key)
    if walk is None:
        if len(_WALK_GRAPHS) >= _WALK_GRAPHS_MAX:
            return None
        statics = _WALK_STATICS.get(skey)
        if statics is None:
            scaled = _scaled_propagator(kernel, plane[0] * plane[1])
            statics = _WALK_STATICS[skey] = _WalkStatics(
                flat[:block], transmissions, scaled
            )
        statics.load(transmissions, kernel)
        statics.block.copy_(flat[:block])
        walk = _WalkGraph(statics, dense, int(start), unit, bool(fold))
        _WALK_GRAPHS[key] = walk
    return walk
