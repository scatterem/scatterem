"""The ``scatterem.utils`` rows of the backend registry.

Same contract as ``scatterem/nn/_backends.py``: one
:class:`~scatterem.utils.env.Backend` per kernel that ``utils`` can run more
than one way.
"""

from __future__ import annotations

from scatterem.utils.env import Backend, Switch, register_backend


#: ``reconstruction.fusion._band_triton.fuse_half`` (the band fusion's half-spectrum
#: C2R) or the shipped ops; tensor gates inline, ``SCATTEREM_FUSE_HALF_C2R=0`` forces
#: torch.
FUSE_HALF = register_backend(
    Backend(
        name="fuse_half",
        impls=("triton", "torch"),
        switches={"triton": (Switch("SCATTEREM_FUSE_HALF_C2R"),)},
    )
)

#: ``_overlap_counts_warp.overlap_counts``: the analytical
#: SSNR's double/triple overlap counts as one bit-exact Warp kernel, or the ~70
#: float64 ATen launches of ``transfer.double_and_triple_pixel_counts_torch``.
#: No kill switch -- gated on device / dtype / grad mode / the scalar constants.
OVERLAP_COUNTS = register_backend(
    Backend(
        name="overlap_counts",
        impls=("warp", "torch"),
        accepts={
            "warp": ("scatterem.utils._overlap_counts_warp:overlap_counts_supported",)
        },
    )
)


#: ``_subpixel_tail_warp.refine_tail``: the argmax and both
#: parabola fits of ``subpixel._peak_offsets`` as one bit-exact Warp kernel, or
#: the ~65 small torch launches.  Gated on the operands;
#: ``SCATTEREM_USE_WARP=0``
#: selects the torch ops.
SUBPIXEL_TAIL = register_backend(
    Backend(
        name="subpixel_tail",
        impls=("warp", "torch"),
        use_warp_env=True,
        available={"warp": ("scatterem.physics._warp_interop:warp_sees_cuda",)},
        accepts={
            "warp": ("scatterem.utils._subpixel_tail_warp:refine_tail_supported",)
        },
    )
)
