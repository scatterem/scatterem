"""The L2-blocked, graph-replayed S-matrix build of ``PrismScanOp``.

Four cases: the single-pass branch, the beam-chunked branch, the parent waves
generated per block inside the graph, and a b2 block at 512^2.
The blocked routes must be BIT-identical to the dense loops: ``r3_simulator``'s
PRISM oracle is a sha256 of the cube.  The tests rebuild with a fresh ``T`` after
the capture, because a replay that read a stale static input would still return
the capture-time values.  The dense reference is forced by
monkeypatching ``resolve_backend``.
"""

import numpy as np
import pytest
import torch
from _pinned_device import pin_device_properties

from scatterem.data import SMeta
from scatterem.nn.modules import Propagator
from scatterem.nn.scan_exit_wave import PrismScanOp
from scatterem.nn.scan_exit_wave import _prism_s_build_graph as pr

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


@pytest.fixture(autouse=True)
def _a6000_sized_gates(monkeypatch):
    # Block sizes and gate decisions below are asserted for a 6 MiB / 84-SM card.
    pin_device_properties(monkeypatch)


N = 192  # r3_simulator/small's PRISM grid: non-power-of-two, so the deferred 1/N runs
NZ = 4


def _only(*rows):
    """A ``resolve_backend`` that answers "graph" for ``rows`` only."""
    names = {r.name for r in rows}

    def resolve(row, requested=None, *args):
        from scatterem.utils.env import resolve_backend

        if row.name not in names:
            return "dense"
        return resolve_backend(row, requested, *args)

    return resolve


@pytest.fixture
def route(monkeypatch):
    """The plain block graph only (the generating graph runs ahead of it)."""
    monkeypatch.delenv("SCATTEREM_PRISM_BUILD_COLS", raising=False)
    monkeypatch.setattr(pr, "resolve_backend", _only(pr.PRISM_BUILD_L2_GRAPH))
    pr._S_BUILD_GRAPHS.clear()
    yield
    pr._S_BUILD_GRAPHS.clear()


@pytest.fixture
def gen_route(monkeypatch):
    monkeypatch.delenv("SCATTEREM_PRISM_BUILD_COLS", raising=False)
    pr._S_BUILD_GRAPHS.clear()
    pr._S_GEN_GRAPHS.clear()
    yield
    pr._S_BUILD_GRAPHS.clear()
    pr._S_GEN_GRAPHS.clear()


def _smeta(n, take):
    return SMeta(
        all_beams=take,
        parent_beams=take,
        numerical_aperture_radius_pixels=3,
        dx=np.array([0.1, 0.1], dtype=np.float32),
        M=(n, n),
        device=torch.device("cuda"),
    )


def _op(by, bx, f=1):
    take = np.zeros((N, N), dtype=bool)
    take[90 : 90 + by, 90 : 90 + bx] = True
    return PrismScanOp(_smeta(N, take), f=f).to("cuda")


def _prop(dz=2.0, n=N):
    return Propagator(dz, 0.0197, (0.1, 0.1), (n, n), device="cuda")


def _T(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.view_as_complex(
        torch.randn(NZ, N, N, 2, device="cuda", generator=g) * 0.1
    ).exp()


def _same(a, b):
    return torch.equal(torch.view_as_real(a), torch.view_as_real(b))


def _dense(op, T, prop):
    with pytest.MonkeyPatch.context() as m:
        m.setattr(pr, "resolve_backend", lambda *a, **k: "dense")
        return op._build_s_matrix(T, prop)


def _blocked(op, T, prop):
    calls = pr.S_BUILD_L2_GRAPH_STATS["calls"]
    out = op._build_s_matrix(T, prop)
    return out, pr.S_BUILD_L2_GRAPH_STATS["calls"] - calls


def _generated(op, T, prop):
    calls = pr.S_BUILD_GEN_GRAPH_STATS["calls"]
    out = op._build_s_matrix(T, prop)
    return out, pr.S_BUILD_GEN_GRAPH_STATS["calls"] - calls


# 6x6 = 36 beams = two blocks of 16 + an eager remainder of 4; 4x8 = 32 = no remainder
@pytest.mark.parametrize("box", [(6, 6), (4, 8)])
@torch.no_grad()
def test_blocked_bit_identical_on_fresh_T(route, box):
    op, prop = _op(*box), _prop()
    caps = pr.S_BUILD_L2_GRAPH_STATS["captures"]
    for seed in (1, 2, 3):  # 1 captures; 2 and 3 replay on a T it never saw
        T = _T(seed)
        out, reached = _blocked(op, T, prop)
        assert reached == 1
        assert _same(out, _dense(op, T, prop))
    assert pr.S_BUILD_L2_GRAPH_STATS["captures"] - caps == 1
    assert pr._S_BUILD_GRAPHS[op][1].block == 16


# the BEAM-CHUNKED branch builds each chunk through the same graph.  9x9 = 81
# beams at 40 per chunk = 40 + 40 (blocked, 2 x 16 + an eager remainder of 8 each)
# + 1 (fits L2 -> dense loop).
@torch.no_grad()
def test_chunked_branch_blocked_bit_identical(route, monkeypatch):
    op, prop = _op(9, 9), _prop()
    caps = pr.S_BUILD_L2_GRAPH_STATS["captures"]
    for seed in (1, 2):
        T = _T(seed)
        single = _dense(op, T, prop)
        monkeypatch.setenv("SCATTEREM_PRISM_BUILD_COLS", "40")
        out, reached = _blocked(op, T, prop)
        assert reached == 2
        ref = _dense(op, T, prop)
        monkeypatch.delenv("SCATTEREM_PRISM_BUILD_COLS")
        assert _same(out, ref) and _same(out, single)
    assert pr.S_BUILD_L2_GRAPH_STATS["captures"] - caps == 1


@torch.no_grad()
def test_recapture_on_propagator_change(route):
    op, T = _op(6, 6), _T(1)
    caps = pr.S_BUILD_L2_GRAPH_STATS["captures"]
    _blocked(op, T, _prop(2.0))
    prop2 = _prop(3.0)
    out, reached = _blocked(op, T, prop2)
    assert reached == 1
    assert pr.S_BUILD_L2_GRAPH_STATS["captures"] - caps == 2
    assert _same(out, _dense(op, T, prop2))


@torch.no_grad()
def test_row_keeps_dense_loop(route, monkeypatch):
    monkeypatch.setattr(pr, "resolve_backend", lambda *a, **k: "dense")
    op = _op(6, 6)
    _, reached = _blocked(op, _T(1), _prop())
    assert reached == 0 and op not in pr._S_BUILD_GRAPHS


def test_refused_under_grad(route):
    T = _T(1).requires_grad_(True)
    out, reached = _blocked(_op(6, 6), T, _prop())
    assert reached == 0
    assert out.requires_grad


@torch.no_grad()
def test_refused_with_fp16_or_exit_aperture(gen_route):
    T, prop = _T(1), _prop()
    w = torch.zeros((36, N, N), dtype=torch.complex64, device="cuda")
    op = _op(6, 6)
    assert pr.s_build_l2_graph_supported(op, T, w, prop)
    assert pr.s_build_gen_graph_supported(op, T, prop)
    op.exit_antialias_aperture = torch.ones((N, N), device="cuda")
    assert not pr.s_build_l2_graph_supported(op, T, w, prop)
    assert not pr.s_build_gen_graph_supported(op, T, prop)
    op = _op(6, 6)
    op.precision_config = {**op.precision_config, "build": "fp16"}
    assert not pr.s_build_gen_graph_supported(op, T, prop)


def test_gate_block_sizes():
    prop, T = _prop(), _T(1)
    w = torch.zeros((36, N, N), dtype=torch.complex64, device="cuda")
    assert pr._s_build_l2_block(T, w, prop) == 16
    # A stack that already fits 3/4 L2: nothing to recover.
    assert pr._s_build_l2_block(T, w[:8], prop) == 0
    # 512^2 (r3_simulator/medium): only b2 fits L2; the gate admits it.
    big = torch.zeros((16, 512, 512), dtype=torch.complex64, device="cuda")
    Tb = torch.ones((NZ, 512, 512), dtype=torch.complex64, device="cuda")
    assert pr._s_build_l2_block(Tb, big, _prop(n=512)) == 2
    # 1024^2: a single 8 MiB beam plane does not fit L2 -> dense.
    huge = torch.zeros((4, 1024, 1024), dtype=torch.complex64, device="cuda")
    Th = torch.ones((NZ, 1024, 1024), dtype=torch.complex64, device="cuda")
    assert pr._s_build_l2_block(Th, huge, _prop(n=1024)) == 0
    assert pr._s_build_l2_block(T.cpu(), w.cpu(), prop) == 0
    assert pr._s_build_l2_block(T[:1], w, prop) == 0  # one slice
    assert pr._s_build_l2_block(T, w.to(torch.complex128), prop) == 0
    assert pr._s_build_l2_block(T, w[:, ::2], prop) == 0  # non-contiguous


def test_b2_block_graph_bit_identical_at_512():
    """The b2 block the gate admits at 512^2, incl. an odd remainder."""
    g = torch.Generator(device="cuda").manual_seed(3)
    w = torch.randn((5, 512, 512), generator=g, device="cuda", dtype=torch.complex64)
    T = torch.exp(1j * 0.1 * torch.randn((NZ, 512, 512), generator=g, device="cuda"))
    T = T.to(torch.complex64)
    prop = _prop(n=512)
    assert pr._s_build_l2_block(T, w, prop) == 2
    if not pr.cufft_batch_bit_invariant(w.unsqueeze(0), 2):
        pytest.skip("cuFFT 512^2 is batch-variant on this device")
    with torch.no_grad():
        ref = pr._s_build_chain(torch.unbind(T, 0), w.clone(), prop)
        graph = pr._SBuildL2BlockGraph(T, w, prop, 2)
        T2 = T * torch.exp(1j * torch.ones((), device="cuda")).to(T.dtype)
        ref2 = pr._s_build_chain(torch.unbind(T2, 0), w.clone(), prop)
        out = graph.run(T, w.clone(), prop)
        out2 = graph.run(T2, w.clone(), prop)
    assert torch.equal(out, ref)
    assert torch.equal(out2, ref2)


# ---------------------------------------------------------------------------
# the parent waves generated per block inside the graph, ahead of the plain
# block graph on both branches.


# f=2 is r3_simulator/small's case: ``_initial_parent_waves`` also scales by fscale.
@pytest.mark.parametrize("f", [1, 2])
@pytest.mark.parametrize("box", [(6, 6), (4, 8)])
@torch.no_grad()
def test_generated_bit_identical_on_fresh_T(gen_route, box, f):
    op, prop = _op(*box, f=f), _prop()
    caps = pr.S_BUILD_GEN_GRAPH_STATS["captures"]
    blk = pr.S_BUILD_L2_GRAPH_STATS["calls"]
    for seed in (1, 2, 3):  # 1 captures; 2 and 3 replay on a T it never saw
        T = _T(seed)
        out, reached = _generated(op, T, prop)
        assert reached == 1
        assert _same(out, _dense(op, T, prop))
    assert pr.S_BUILD_GEN_GRAPH_STATS["captures"] - caps == 1
    assert pr.S_BUILD_L2_GRAPH_STATS["calls"] == blk  # the plain graph never reached
    assert pr._S_GEN_GRAPHS[op][1].block == 16


@torch.no_grad()
def test_generated_serves_chunked_branch(gen_route, monkeypatch):
    op, prop = _op(9, 9), _prop()
    T = _T(1)
    single = _dense(op, T, prop)
    monkeypatch.setenv("SCATTEREM_PRISM_BUILD_COLS", "40")
    out, reached = _generated(op, T, prop)
    assert reached == 1
    assert _same(out, _dense(op, T, prop)) and _same(out, single)


@torch.no_grad()
def test_gen_row_off_falls_back_to_block_graph(gen_route, monkeypatch):
    monkeypatch.setattr(pr, "resolve_backend", _only(pr.PRISM_BUILD_L2_GRAPH))
    op, T, prop = _op(6, 6), _T(1), _prop()
    blk = pr.S_BUILD_L2_GRAPH_STATS["calls"]
    out, reached = _generated(op, T, prop)
    assert reached == 0
    assert pr.S_BUILD_L2_GRAPH_STATS["calls"] - blk == 1
    assert _same(out, _dense(op, T, prop))


@torch.no_grad()
def test_gen_declines_when_cufft_batch_variant(gen_route, monkeypatch):
    monkeypatch.setattr(pr, "cufft_batch_bit_invariant", lambda w, b: False)
    op, T, prop = _op(6, 6), _T(1), _prop()
    out, reached = _generated(op, T, prop)
    assert reached == 0
    assert pr._S_GEN_GRAPHS[op][1] is None
    assert _same(out, _dense(op, T, prop))


def test_gen_refused_under_grad(gen_route):
    T = _T(1).requires_grad_(True)
    out, reached = _generated(_op(6, 6), T, _prop())
    assert reached == 0
    assert out.requires_grad


@torch.no_grad()
def test_gen_declines_512_without_allocating_S(gen_route):
    # r3_simulator/medium's grid: no block >= 8 fits L2, so the route must refuse
    # BEFORE allocating the (Bp, MY, MX) output (a ~20 GB transient at Bp = 9361).
    n = 512
    take = np.zeros((n, n), dtype=bool)
    take[250:254, 250:254] = True  # 16 beams = 32 MiB of S
    op = PrismScanOp(_smeta(n, take), f=1).to("cuda")
    T = torch.ones((NZ, n, n), dtype=torch.complex64, device="cuda")
    prop = _prop(n=n)
    assert not pr.s_build_gen_graph_supported(op, T, prop)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    assert pr.s_build_l2_generated(op, T, prop, 1 << 30) is None
    assert torch.cuda.max_memory_allocated() - base < 16 * n * n * 8


def test_gen_block_is_capped_to_the_chunk(monkeypatch):
    # On a 48 MiB L2 (an L4), the L2 law alone picks b128 for 256 beams at 192^2.
    # A smaller beam chunk sets the block equal to the chunk, not to the power
    # of two below it.  Below the minimum block of 8, the generating graph is
    # off.  This test checks only arithmetic, so cuFFT has no effect on it.
    pin_device_properties(monkeypatch, l2=48 * 2**20)
    op, T = _op(16, 16), _T(1)
    assert pr._gen_block(op, T) == 128
    assert pr._gen_block(op, T, 1 << 30) == 128
    assert pr._gen_block(op, T, 128) == 128
    assert pr._gen_block(op, T, 40) == 40
    assert pr._gen_block(op, T, 8) == 8
    assert pr._gen_block(op, T, 4) == 0


@torch.no_grad()
def test_gen_row_sees_the_chunk(gen_route, monkeypatch):
    # The row and the route use the same capped block.  Thus the route takes
    # each build that the row accepts.  The row refuses a chunk below the
    # minimum block, and the route does not count that build.
    pin_device_properties(monkeypatch, l2=48 * 2**20)
    op, prop, T = _op(16, 16), _prop(), _T(1)
    assert pr.s_build_gen_graph_supported(op, T, prop)
    assert pr.s_build_gen_graph_supported(op, T, prop, 40)
    assert not pr.s_build_gen_graph_supported(op, T, prop, 4)
    stats = dict(pr.S_BUILD_GEN_GRAPH_STATS)
    assert pr.s_build_l2_generated(op, T, prop, 4) is None
    assert pr.S_BUILD_GEN_GRAPH_STATS == stats


@pytest.mark.parametrize(("cols", "block"), [("40", 40), ("4", None)])
@torch.no_grad()
def test_generated_block_never_exceeds_the_chunk(gen_route, monkeypatch, cols, block):
    # On a 48 MiB L2 the L2 law alone picks b128 at 192^2.  That transient is much
    # larger than a 40-beam chunk, which SCATTEREM_PRISM_BUILD_COLS sets here.
    # The capped graph runs the batch counts of the dense build (6 x 40, then
    # 16 eagerly).  Thus it is bit-identical on all cards.  A chunk below the
    # minimum block declines.
    pin_device_properties(monkeypatch, l2=48 * 2**20)
    op, prop, T = _op(16, 16), _prop(), _T(1)
    monkeypatch.setenv("SCATTEREM_PRISM_BUILD_COLS", cols)
    out, reached = _generated(op, T, prop)
    if block is None:
        assert reached == 0
    else:
        assert reached == 1
        assert pr._S_GEN_GRAPHS[op][1].block == block
    assert _same(out, _dense(op, T, prop))
