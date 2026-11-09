"""The ``scatterem.nn`` rows of the backend registry.

One :class:`~scatterem.utils.env.Backend` per kernel that ``nn`` can run more
than one way; callers ask :func:`~scatterem.utils.env.resolve_backend` (or its
Warp-or-torch spelling :func:`~scatterem.utils.env.resolve_use_warp`) with the
row.  Predicates are named as ``"module:function"`` strings, so importing this
module imports no kernel module and cannot close an import cycle.
"""

from __future__ import annotations

from scatterem.utils.env import Backend, Switch, register_backend


_F = "scatterem.nn.functional"

#: The PRISM projection (``nn/functional/warp/prism_project/dispatch.py``):
#: ``use_warp``, else ``SCATTEREM_USE_WARP``, else torch.
PRISM_PROJECT = register_backend(
    Backend(
        name="prism_project",
        impls=("warp", "torch"),
        default="torch",
        use_warp_env=True,
    )
)

#: The torch PRISM projection's frozen-basis backward
#: (``prism_project/_torch_adjoint.py``): the per-position window ``addcmul_``
#: scatter (deterministic) or the ``index_add_`` body.  Registered, not
#: asked: the choice is made inside ``autograd.Function.backward``, which also
#: refuses the window under capture / compile or for a crop larger than the
#: grid, so it reads the switch with ``switched_on``.
#: ``SCATTEREM_PRISM_WINDOW_BWD=0`` is a supported escape hatch.
PRISM_WINDOW_BWD = register_backend(
    Backend(
        name="prism_window_bwd",
        impls=("window", "index_add"),
        switches={"window": (Switch("SCATTEREM_PRISM_WINDOW_BWD"),)},
    )
)





#: The separable object gather (``nn/ptychography/_transmit.py``,
#: ``_separable_expand``): a compiled two-table gather or ``index_select``.
#: OPT-IN through ``SCATTEREM_SEPARABLE_GATHER=1`` (its first call pays a
#: ``torch.compile`` that only a long forward-only sweep earns back).  Registered,
#: not asked: the tensor gates and the index proof are inline, and the switch is
#: read with ``switched_on``, which is safe while ``torch.compile`` traces.
SEPARABLE_GATHER = register_backend(
    Backend(
        name="separable_gather",
        impls=("separable", "index_select"),
        switches={"separable": (Switch("SCATTEREM_SEPARABLE_GATHER", default=False),)},
    )
)









#: ``_conj_mul_reduce_triton.conj_mul_reduce`` (the transmit adjoint's
#: ``grad_T``); ``SCATTEREM_CONJ_MUL_REDUCE=0`` forces torch.
CONJ_MUL_REDUCE = register_backend(
    Backend(
        name="conj_mul_reduce",
        impls=("triton", "torch"),
        switches={"triton": (Switch("SCATTEREM_CONJ_MUL_REDUCE"),)},
        declines={
            "triton": (f"{_F}._conj_mul_reduce_triton:conj_mul_reduce_declines",)
        },
    )
)

#: ``conj_mul_reduce_bcast``, the dual (``grad_T`` and ``grad_waves``) kernel;
#: ``SCATTEREM_CONJ_MUL_DUAL=0`` forces torch on the eager and compiled routes.
CONJ_MUL_REDUCE_BCAST = register_backend(
    Backend(
        name="conj_mul_reduce_bcast",
        impls=("triton", "torch"),
        switches={"triton": (Switch("SCATTEREM_CONJ_MUL_DUAL"),)},
        declines={
            "triton": (f"{_F}._conj_mul_reduce_triton:conj_mul_reduce_bcast_declines",)
        },
    )
)

#: ``conj_mul_bcast_products``: the transmit adjoint's two products in
#: one pass above ``_MAX_MODES``, where the dual kernel declines; torch keeps the
#: mode sum, so both gradients are ``torch.equal`` to the eager chain at any mode
#: count.  The dual row's ``SCATTEREM_CONJ_MUL_DUAL=0`` turns it off too.
CONJ_MUL_BCAST_PRODUCTS = register_backend(
    Backend(
        name="conj_mul_bcast_products",
        impls=("triton", "torch"),
        switches={"triton": (Switch("SCATTEREM_CONJ_MUL_DUAL"),)},
        declines={
            "triton": (
                f"{_F}._conj_mul_reduce_triton:conj_mul_bcast_products_declines",
            )
        },
    )
)



#: ``_gather_cmul_triton.gather_cmul``; its tensor gates are inline, so the row
#: holds only ``SCATTEREM_GATHER_CMUL``.
GATHER_CMUL = register_backend(
    Backend(
        name="gather_cmul",
        impls=("triton", "torch"),
        switches={"triton": (Switch("SCATTEREM_GATHER_CMUL"),)},
    )
)

#: ``_bcast_cmul_triton.bcast_cmul_``; tensor gates inline, the row holds
#: ``SCATTEREM_BCAST_CMUL``.
BCAST_CMUL = register_backend(
    Backend(
        name="bcast_cmul",
        impls=("triton", "torch"),
        switches={"triton": (Switch("SCATTEREM_BCAST_CMUL"),)},
    )
)

#: ``_bcast_mul_conj_triton.bcast_mul_conj``: fuse ``conj`` into the joint
#: PRISM backward's ``grad * conj(S_crop)``.  No kill switch -- gated on dtype /
#: device / contiguity / grad mode.
BCAST_MUL_CONJ = register_backend(
    Backend(
        name="bcast_mul_conj",
        impls=("triton", "torch"),
        accepts={"triton": (f"{_F}._bcast_mul_conj_triton:bcast_mul_conj_supported",)},
    )
)

#: ``_bcast_mul_conj_triton.bcast_mul_conj_ksum``: the same product summed
#: over the ``K`` positions in registers, for the joint PRISM backward on a
#: K-broadcast (integer-pixel) basis.  No kill switch; gated on the operands.
BCAST_MUL_CONJ_KSUM = register_backend(
    Backend(
        name="bcast_mul_conj_ksum",
        impls=("triton", "torch"),
        accepts={
            "triton": (f"{_F}._bcast_mul_conj_triton:bcast_mul_conj_ksum_supported",)
        },
    )
)

#: ``_crop_contract_triton.crop_contract``: the graph-building torch
#: PRISM forward's ``(basis_k * crop[None]).sum(1)`` without the product, reading
#: the crop or gathering it from ``Sdt``.  No kill switch; gated on the operands.
CROP_CONTRACT = register_backend(
    Backend(
        name="crop_contract",
        impls=("triton", "torch"),
        accepts={"triton": (f"{_F}._crop_contract_triton:crop_contract_supported",)},
    )
)

#: ``_prism_window_gather_triton.window_gather_grad_sdt``: the torch
#: PRISM window backward's ``grad_Sdt`` as one per-output-pixel gather instead of
#: the fill + K window ``addcmul_`` passes + conj.  Asked only when the
#: ``prism_window_bwd`` row chose the window.  No kill switch.
PRISM_WINDOW_GATHER = register_backend(
    Backend(
        name="prism_window_gather",
        impls=("triton", "torch"),
        accepts={
            "triton": (f"{_F}._prism_window_gather_triton:window_gather_supported",)
        },
    )
)

#: ``_ramp_adjoint_ksum_triton.ramp_adjoint_ksum``: the torch PRISM
#: route's ``basis_fft`` adjoint of ``basis_fft[:, :, None] * ramp`` summed over
#: ``K`` in one pass (bit-exact).  No kill switch; gated on the operands.
RAMP_ADJOINT_KSUM = register_backend(
    Backend(
        name="ramp_adjoint_ksum",
        impls=("triton", "torch"),
        accepts={
            "triton": (f"{_F}._ramp_adjoint_ksum_triton:ramp_adjoint_ksum_supported",)
        },
    )
)

#: ``_crop_contract_exact_triton.crop_contract_exact``: the no-grad
#: torch PRISM block body (crop gather, product, ``p``-sum, running add) as one
#: pass that sums in TensorIterator's order, so it is ``torch.equal`` to the eager
#: body.  No kill switch; gated on the operands (``1 <= nb < 1021``, one mode).
CROP_CONTRACT_EXACT = register_backend(
    Backend(
        name="crop_contract_exact",
        impls=("triton", "torch"),
        accepts={
            "triton": (
                f"{_F}._crop_contract_exact_triton:crop_contract_exact_supported",
            )
        },
    )
)

#: The torch PRISM route's no-grad block ramp multiply: the Warp
#: ``_ramp_mul`` (K-outer, bit-exact) in place of the eager stride-0 broadcast,
#: where ``crop_contract_exact`` consumes the result.  No kill switch.
PRISM_TORCH_RAMP_MUL = register_backend(
    Backend(
        name="prism_torch_ramp_mul",
        impls=("warp", "torch"),
        accepts={
            "warp": (f"{_F}.warp.prism_project._torch_blocks:torch_ramp_mul_supported",)
        },
    )
)

#: The same multiply on the torch PRISM route's frozen-probe GRAD block,
#: read by stride by ``_CropContractFrozenBasis``.  No kill switch; gated on the
#: operands and on the plane filling the device.
PRISM_TORCH_RAMP_MUL_GRAD = register_backend(
    Backend(
        name="prism_torch_ramp_mul_grad",
        impls=("warp", "torch"),
        accepts={
            "warp": (
                f"{_F}.warp.prism_project._torch_blocks:torch_ramp_mul_grad_supported",
            )
        },
    )
)

#: The launch of ``bcast_mul_conj_ksum``'s gather mode: BLOCK 1024 /
#: 2 warps / 8 beam groups where that still gives every SM three programs, else
#: the 512 / 4 / 16 launch.  Only the program mapping changes (bit-identical).
KSUM_GATHER_WIDE = register_backend(
    Backend(
        name="ksum_gather_wide",
        impls=("wide", "narrow"),
        accepts={"wide": (f"{_F}._bcast_mul_conj_triton:ksum_gather_wide_supported",)},
    )
)




