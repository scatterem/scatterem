"""The ``scatterem.data`` rows of the backend registry.

Same contract as ``scatterem/nn/_backends.py``: one
:class:`~scatterem.utils.env.Backend` per kernel that ``data`` can run
more than one way.
"""

from __future__ import annotations

from scatterem.utils.env import Backend, register_backend


_CAL = "scatterem.data.calibration"
_DS = "scatterem.data.datasets"














#: ``_com_marginals_triton.com_marginals``: both detector marginals of the
#: (masked) cube in one pass.  No kill switch -- gated on dtype / device /
#: contiguity / grad mode.
COM_MARGINALS = register_backend(
    Backend(
        name="com_marginals",
        impls=("triton", "torch"),
        accepts={"triton": (f"{_CAL}._com_marginals_triton:com_marginals_supported",)},
    )
)

#: ``_rotation_fused.rotation_curves_triton``: the CoM rotation search's
#: objective at every angle and both handednesses in one launch.  No kill switch.
ROTATION_SEARCH = register_backend(
    Backend(
        name="rotation_search",
        impls=("triton", "torch"),
        accepts={"triton": (f"{_CAL}._rotation_fused:triton_rotation_supported",)},
    )
)


#: ``datasets._clamp_cast_warp.clamp_and_row_sums_f64``: the
#: ``Dataset4DStem`` constructor's clamp (and its ``sqrt_``) and the
#: float64 copy of its per-row dose sums in one bit-exact Warp pass, or
#: ``clamp_min_`` (``sqrt_``) and a float64 copy per row.
#: ``SCATTEREM_USE_WARP=0`` selects the torch ops.
CLAMP_CAST = register_backend(
    Backend(
        name="clamp_cast",
        impls=("warp", "torch"),
        use_warp_env=True,
        available={"warp": ("scatterem.physics._warp_interop:warp_sees_cuda",)},
        accepts={"warp": (f"{_DS}._clamp_cast_warp:clamp_cast_supported",)},
    )
)

#: ``datasets._sum_order_warp.row_sums_torch_order``: inside the
#: ``clamp_cast`` pass, the per-row float64 dose sums added in the order of
#: torch 2.11's own ``.sum()`` (bit-exact, no float64 copy), or the float64 row
#: slots and torch's ``.sum()``.  The predicate takes the row length.
CLAMP_SUM_ORDER = register_backend(
    Backend(
        name="clamp_sum_order",
        impls=("warp", "torch"),
        use_warp_env=True,
        accepts={"warp": (f"{_DS}._sum_order_warp:sum_order_supported",)},
    )
)

#: ``datasets._shift_clamp_warp.fftshift_last2_clamp_and_row_sums_f64``:
#: the constructor's detector ``fftshift`` written by the torch-order
#: clamp + dose pass, or the shifted copy and then that pass.  It runs only
#: where ``clamp_cast`` and ``clamp_sum_order`` also choose ``"warp"``.
FFTSHIFT_CLAMP = register_backend(
    Backend(
        name="fftshift_clamp",
        impls=("warp", "torch"),
        use_warp_env=True,
        available={"warp": ("scatterem.physics._warp_interop:warp_sees_cuda",)},
        accepts={"warp": (f"{_DS}._shift_clamp_warp:shift_clamp_supported",)},
    )
)



#: ``datasets._crop_clamp_warp.crop_clamp_and_row_sums_f64``:
#: ``Dataset4DStem.crop(clone=True)``'s copy of the strided slice written by the
#: torch-order clamp + dose pass, or ``clone()`` and then the constructor's pass.
#: The predicate takes ``(parent, view)``.  It runs only where ``clamp_cast``
#: (on the parent) and ``clamp_sum_order`` also choose ``"warp"``.
CROP_CLAMP = register_backend(
    Backend(
        name="crop_clamp",
        impls=("warp", "torch"),
        use_warp_env=True,
        available={"warp": ("scatterem.physics._warp_interop:warp_sees_cuda",)},
        accepts={"warp": (f"{_DS}._crop_clamp_warp:crop_clamp_supported",)},
    )
)

#: ``datasets._crop_clamp_warp.crop_clamp_normalize_and_row_sums_f64``:
#: with ``normalize=True`` the ``crop_clamp`` pass also divides by the
#: constructor's ``mean(0).max()``, found first from the parent in torch's order
#: (``datasets._mean0_warp``), or the constructor's ``mean`` and ``/=``.  The
#: predicate takes ``(parent, view)``.  It runs only where ``crop_clamp`` also
#: chooses ``"warp"``.
CROP_NORMALIZE = register_backend(
    Backend(
        name="crop_normalize",
        impls=("warp", "torch"),
        use_warp_env=True,
        available={"warp": ("scatterem.physics._warp_interop:warp_sees_cuda",)},
        accepts={"warp": (f"{_DS}._crop_clamp_warp:crop_normalize_supported",)},
    )
)


#: ``datasets._shift_norm_warp.fftshift_last2_clamp_normalize_and_row_sums_f64``:
#: with ``normalize=True`` the ``fftshift_clamp`` pass also
#: divides by the constructor's ``mean(0).max()``, found first from the source
#: in torch's order, or the constructor's ``mean`` and ``/=``.  It runs only
#: where ``fftshift_clamp``, ``clamp_cast`` and ``clamp_sum_order`` also choose
#: ``"warp"``.
FFTSHIFT_NORMALIZE = register_backend(
    Backend(
        name="fftshift_normalize",
        impls=("warp", "torch"),
        use_warp_env=True,
        available={"warp": ("scatterem.physics._warp_interop:warp_sees_cuda",)},
        accepts={"warp": (f"{_DS}._shift_norm_warp:shift_normalize_supported",)},
    )
)

