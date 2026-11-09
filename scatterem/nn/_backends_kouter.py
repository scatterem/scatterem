"""The ``scatterem.nn`` backend rows for the k-outer ``reduce="sum"`` store.

Split from :mod:`scatterem.nn._backends` (module size).  Each row picks how
``_direct_ptychography_kouter``'s ``reduce="sum"`` body stores the corrected
chunk: the half-spectrum store and its launch, or the
two-for-one packed store, each against the full-grid correction.
Predicates are ``"module:function"`` strings, so importing this module imports
no kernel module.
"""

from __future__ import annotations

from scatterem.utils.env import Backend, register_backend

__all__ = [
    "KOUTER_HALF",
    "KOUTER_HALF_ROWS",
    "KOUTER_PACKED",
]

_F = "scatterem.nn.functional"

#: The half-spectrum k-outer store or the full-grid correction; its kill
#: switch ``SCATTEREM_KOUTER_HALF_C2R`` was deleted.
KOUTER_HALF = register_backend(
    Backend(
        name="kouter_half",
        impls=("half", "full"),
        accepts={"half": (f"{_F}.ptychography.kouter_half:kouter_half_supported",)},
    )
)

#: How the half-spectrum k-outer store is launched: the row loop (each
#: thread walks ``KOUTER_HALF_ROWS`` columns of one row, bit-identical) or
#: one element per thread.  No predicate: :data:`KOUTER_HALF` has chosen the
#: half store already.
KOUTER_HALF_ROWS = register_backend(
    Backend(name="kouter_half_rows", impls=("rows", "element"))
)

#: The two-for-one packed k-outer store (two planes per C2C) or the
#: full-grid correction.  The caller asks this row only where :data:`KOUTER_HALF`
#: chose ``"full"`` (in practice an odd ``W``).  NOT bit-exact against the full
#: grid.
KOUTER_PACKED = register_backend(
    Backend(
        name="kouter_packed",
        impls=("packed", "full"),
        accepts={
            "packed": (f"{_F}.ptychography.kouter_packed:kouter_packed_supported",)
        },
    )
)
