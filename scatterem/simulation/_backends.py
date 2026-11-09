"""The ``scatterem.simulation`` rows of the backend registry.

Same contract as ``scatterem/nn/_backends.py``: one
:class:`~scatterem.utils.env.Backend` per kernel that ``simulation`` can run
more than one way, with predicates named as ``"module:function"`` strings.

"""

from __future__ import annotations

from scatterem.utils.env import Backend, register_backend, resolve_backend



def is_warp(row: Backend, *args) -> bool:
    """Whether ``resolve_backend(row, None, *args)`` chose the Warp impl."""
    return resolve_backend(row, None, *args) == "warp"








#: The L2-blocked EELS exit walk (``_wave/exit_walk.py``): replay one captured
#: block walk per full block (``"graph"``) or issue it eagerly.  Same
#: kernels at the same batch on copies of the same operands, so bit-identical.
EELS_WALK_GRAPH = register_backend(
    Backend(
        name="eels_walk_graph",
        impls=("graph", "eager"),
        accepts={
            "graph": ("scatterem.simulation._wave._walk_graph:walk_graph_supported",)
        },
    )
)





#: EELS ``acc += |psi|^2`` (``abs2_acc_fused``): Warp for contiguous complex64
#: CUDA ``psi`` into float64 ``acc``, else the ATen chain.
EELS_ABS2_ACC = register_backend(
    Backend(
        name="eels_abs2_acc",
        impls=("warp", "torch"),
        accepts={
            "warp": (
                "scatterem.simulation.eels.forward.abs2_acc_fused:"
                "abs2_accumulate_fused_supported",
            )
        },
    )
)

#: EELS ``acc += |psi|^2.sum(dim=1)`` (``abs2_sum_acc_fused``): Warp for the
#: four-accumulator complex64 stack, else ATen.
EELS_ABS2_SUM = register_backend(
    Backend(
        name="eels_abs2_sum",
        impls=("warp", "torch"),
        accepts={
            "warp": (
                "scatterem.simulation.eels.forward.abs2_sum_acc_fused:"
                "abs2_sum_accumulate_fused_supported",
            )
        },
    )
)

#: The STEM-EELS image's per-atom ``index_add_`` (``forward/_scan_accumulator.py``):
#: one ordered ``cumsum`` flush, bitwise the per-call scatter, on CUDA; else
#: the per-call ``index_add_``.  No kill switch.
EELS_SCAN_ACC = register_backend(
    Backend(
        name="eels_scan_acc",
        impls=("scan", "index_add"),
        accepts={
            "scan": (
                "scatterem.simulation.eels.forward._scan_accumulator:"
                "scan_accumulate_supported",
            )
        },
    )
)

#: Fourier-shift phase ramp: the certified float-float Warp kernel
#: (``physics._ramp_df``, bitwise the float64 kernel), the
#: float64 Warp kernel (``physics._ramp_fused``), else the eager polar chain.
#: Both kernels take complex64 CUDA without a shifts gradient.  Registered, not
#: asked from ``physics.propagation``: layer 0 may not import the registry
#: (layer 1), so the caller keeps the predicates; the row still owns the probes.
EELS_RAMP = register_backend(
    Backend(
        name="eels_ramp",
        impls=("warp_df", "warp", "torch"),
        accepts={
            "warp_df": ("scatterem.physics._ramp_df:ramp_phase_df_supported",),
            "warp": ("scatterem.physics._ramp_fused:ramp_phase_fused_supported",),
        },
    )
)

#: ``wq * phase`` applied inside the float-float ramp kernel before its store
#: (``physics._ramp_df.ramp_mul_fused``), bitwise torch's
#: complex64 product; else the separate torch multiply.  The predicate takes
#: the single-chunk power-of-two route with 2-D waves.  Asked directly from
#: ``physics.propagation``, as :data:`EELS_RAMP` is.
EELS_RAMP_WQ_FOLD = register_backend(
    Backend(
        name="eels_ramp_wq_fold",
        impls=("fold", "separate"),
        accepts={"fold": ("scatterem.physics._ramp_df:ramp_wq_fold_supported",)},
    )
)







