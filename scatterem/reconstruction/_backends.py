"""The ``scatterem.reconstruction`` rows of the backend registry.

Same contract as ``scatterem/nn/_backends.py``: one
:class:`~scatterem.utils.env.Backend` per kernel that ``reconstruction`` can
run more than one way, with predicates named as ``"module:function"``
strings.
"""

from __future__ import annotations

from scatterem.utils.env import Backend, Switch, register_backend


_R = "scatterem.reconstruction"

#: The drizzle splat: the Warp gather kernel (``_drizzle_fused``) or the eager
#: scatter; ``SCATTEREM_DRIZZLE_FUSED=0`` forces the eager splat.
DRIZZLE = register_backend(
    Backend(
        name="drizzle",
        impls=("warp", "torch"),
        switches={"warp": (Switch("SCATTEREM_DRIZZLE_FUSED"),)},
        accepts={"warp": (f"{_R}.darkfield._drizzle_fused:drizzle_fused_supported",)},
    )
)









#: The analytical SSNR's tail after the PCTF kernel -- bright-field count, PCTF
#: normalisation, overlap counts, noise and SNR epilogue -- as two bit-exact
#: Warp launches (``direct_ptychography._ssnr_analytical_warp``),
#: or the ~20 ATen ops.  No kill switch.
SSNR_EPILOGUE = register_backend(
    Backend(
        name="ssnr_epilogue",
        impls=("warp", "torch"),
        accepts={
            "warp": (
                f"{_R}.ptychography.direct._ssnr_analytical_warp:"
                "ssnr_epilogue_supported",
            )
        },
    )
)


















