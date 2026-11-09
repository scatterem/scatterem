"""The ``scatterem.nn`` backend rows for the captured-CUDA-graph block routes.

Split from :mod:`scatterem.nn._backends` (module size): each row picks between a
route that runs a loop in L2-sized blocks under one replayed CUDA graph and the
dense spelling it is bit-identical to.  Predicates are ``"module:function"``
strings, so importing this module imports no kernel module.
"""

from __future__ import annotations

from scatterem.utils.env import Backend, register_backend

__all__ = [
    "GATHER_CMUL_SEP",
    "GATHER_DEFER_NORM",
    "GATHER_L2_GRAPH",
    "MULTISLICE_L2_GRAPH",
    "PRISM_BUILD_GEN_GRAPH",
    "PRISM_BUILD_L2_GRAPH",
    "PRISM_PROJ_L2_GRAPH",
]

_F = "scatterem.nn.functional"

#: The no-grad sub-pixel Warp PRISM forward run in L2-resident blocks under one
#: captured CUDA graph (``prism_project/_l2_block.py``), bit-identical to the
#: dense ramp-mul -> ifft2 -> project chain.  No kill switch.
PRISM_PROJ_L2_GRAPH = register_backend(
    Backend(
        name="prism_proj_l2_graph",
        impls=("graph", "dense"),
        accepts={
            "graph": (f"{_F}.warp.prism_project._l2_block:prism_proj_l2_supported",)
        },
    )
)

_SEW = "scatterem.nn.scan_exit_wave"

#: The multislice slice loop in L2-sized blocks of positions under one replayed
#: CUDA graph (``scan_exit_wave/_l2_block_graph.py``), Warp fused-transmit flavour.
#: No kill switch; gated on the operands.
MULTISLICE_L2_GRAPH = register_backend(
    Backend(
        name="multislice_l2_graph",
        impls=("graph", "dense"),
        accepts={"graph": (f"{_SEW}._l2_block_graph:multislice_l2_graph_supported",)},
    )
)

#: The same for the torch-route (library-default) slice loop with the fused
#: Triton ``gather_cmul``.  No kill switch; gated on the operands.
GATHER_L2_GRAPH = register_backend(
    Backend(
        name="gather_l2_graph",
        impls=("graph", "dense"),
        accepts={"graph": (f"{_SEW}._l2_block_graph:gather_l2_graph_supported",)},
    )
)

#: Inside both graphs above, each slice's gather + transmit computed from a row
#: table and a column table instead of the full index plane, with ``T`` read as one
#: 8-byte word (Triton ``gather_cmul_sep`` on the torch route and
#: on the Warp route).  ``separable_tables`` verifies the box-linear identity
#: on every element before the route is used, and the kernel rounds the product as
#: the dense loop's kernel does, so the waves are bit-identical.
GATHER_CMUL_SEP = register_backend(
    Backend(
        name="gather_cmul_sep",
        impls=("separable", "index"),
        accepts={"separable": (f"{_F}._gather_cmul_triton:gather_cmul_sep_supported",)},
    )
)

#: Inside the separable route above, on a grid that is not a power of two: the
#: propagator leaves the inverse FFT's ``1/N`` to the next slice's
#: ``gather_cmul_sep(scale=)``, so its separate scaling pass goes.
#: Same values as the pass; only the sign of an exact-zero component can differ.
#: ``"pass"`` keeps the pass.
GATHER_DEFER_NORM = register_backend(
    Backend(
        name="gather_defer_norm",
        impls=("deferred", "pass"),
        accepts={
            "deferred": (f"{_F}._gather_cmul_triton:gather_defer_norm_supported",)
        },
    )
)

#: The PRISM S-matrix build in L2-sized blocks of beams under one replayed CUDA
#: graph (``scan_exit_wave/_prism_s_build_graph.py``).
#: No kill switch; gated on the operands.
PRISM_BUILD_L2_GRAPH = register_backend(
    Backend(
        name="prism_build_l2_graph",
        impls=("graph", "dense"),
        accepts={"graph": (f"{_SEW}._prism_s_build_graph:s_build_l2_graph_supported",)},
    )
)

#: The same build with each block's parent waves generated inside the graph;
#: declining falls back to ``prism_build_l2_graph``.
PRISM_BUILD_GEN_GRAPH = register_backend(
    Backend(
        name="prism_build_gen_graph",
        impls=("graph", "dense"),
        accepts={
            "graph": (f"{_SEW}._prism_s_build_graph:s_build_gen_graph_supported",)
        },
    )
)
