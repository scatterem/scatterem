"""PyTorch functional operations for direct ptychography.

Split out of the former flat ``ptychography.py`` (2,581 lines) into one module
per kernel family: the shared memos (:mod:`._caches`), the hoisted aperture
factor (:mod:`.aperture`), and the four correction paths -- in-place
(:mod:`.inplace`), k-outer (:mod:`.kouter`), differentiable (:mod:`.autograd`)
and fused correct-and-reduce (:mod:`.ksum`) -- plus the transfer function
(:mod:`.ctf`).  The perf-lab squared-aperture switches and threshold live in
:mod:`._sq_aperture`, the k-SUM ``ik`` split in :mod:`._ksum_tiled`, the
half-spectrum k-outer store in :mod:`.kouter_half`, and its two-for-one packed
twin in :mod:`.kouter_packed`.

This package reproduces that module's ENTIRE attribute surface -- every public
name and every private one, because callers and tests reach for both by
attribute (``_mod._CONTIG_CACHE``, ``_mod._sin_cos_floats``) as well as by
import.  Functions are re-exported statically below; the module-level *data* is
served by ``__getattr__`` instead, so that ``_fp._KOUTER_CMD`` is the live dict
the op mutates rather than a binding frozen at import time -- which is what a
cache reset depends on.

What does NOT survive the split: SETTING an attribute here does not reach the
module that defines it.  Code (and tests) that monkeypatch an internal must
target the owning module -- ``._caches`` for the memos, ``.kouter``/``.ksum``
for the recorded-launch records and the bypass counters.
"""

from . import _caches as _caches
from . import _ksum_tiled as _ksum_tiled
from . import _sq_aperture as _sq_aperture
from . import kouter as kouter
from . import kouter_half as kouter_half
from . import kouter_packed as kouter_packed
from . import ksum as ksum
from ._caches import _chi_is_zero_flag as _chi_is_zero_flag
from ._caches import _contiguous_memo as _contiguous_memo
from ._caches import _sin_cos_floats as _sin_cos_floats
from ._caches import _sin_cos_rot as _sin_cos_rot
from ._caches import chi_is_zero_host as chi_is_zero_host
from ._ksum_tiled import _ksum_tiles as _ksum_tiles
from ._sq_aperture import _pctf_sq_threshold as _pctf_sq_threshold
from .aperture import build_aberration_bf_factor as build_aberration_bf_factor
from .aperture import (
    build_aberration_bf_factor_planes as build_aberration_bf_factor_planes,
)
from .autograd import _correct_aberrations_backward as _correct_aberrations_backward
from .autograd import _correct_aberrations_bwd as _correct_aberrations_bwd
from .autograd import _correct_aberrations_fwd as _correct_aberrations_fwd
from .autograd import (
    _correct_aberrations_setup_context as _correct_aberrations_setup_context,
)
from .autograd import correct_aberrations as correct_aberrations
from .ctf import (
    _phase_contrast_transfer_function_fwd as _phase_contrast_transfer_function_fwd,
)
from .ctf import phase_contrast_transfer_function as phase_contrast_transfer_function
from .inplace import _correct_aberrations_inplace_op as _correct_aberrations_inplace_op
from .inplace import correct_aberrations_inplace as correct_aberrations_inplace
from .kouter import _correct_aberrations_kouter_impl as _correct_aberrations_kouter_impl
from .kouter import _correct_aberrations_kouter_op as _correct_aberrations_kouter_op
from .kouter import _wp_ndim as _wp_ndim
from .kouter import correct_aberrations_kouter as correct_aberrations_kouter
from .kouter_half import (
    correct_aberrations_kouter_half as correct_aberrations_kouter_half,
)
from .kouter_half import kouter_half_supported as kouter_half_supported
from .kouter_packed import (
    correct_aberrations_kouter_packed as correct_aberrations_kouter_packed,
)
from .kouter_packed import kouter_packed_supported as kouter_packed_supported
from .ksum import (
    _correct_aberrations_ksum_backward as _correct_aberrations_ksum_backward,
)
from .ksum import _correct_aberrations_ksum_bwd as _correct_aberrations_ksum_bwd
from .ksum import _correct_aberrations_ksum_fwd as _correct_aberrations_ksum_fwd
from .ksum import (
    _correct_aberrations_ksum_planes_impl as _correct_aberrations_ksum_planes_impl,
)
from .ksum import (
    _correct_aberrations_ksum_planes_op as _correct_aberrations_ksum_planes_op,
)
from .ksum import (
    _correct_aberrations_ksum_setup_context as _correct_aberrations_ksum_setup_context,
)
from .ksum import _ksum_bind_key as _ksum_bind_key
from .ksum import _ksum_fwd_body as _ksum_fwd_body
from .ksum import _ksum_fwd_bypass_ok as _ksum_fwd_bypass_ok
from .ksum import correct_aberrations_ksum as correct_aberrations_ksum
from .ksum import correct_aberrations_ksum_planes as correct_aberrations_ksum_planes

# Module-level data is served live from the module that owns it: re-exporting by
# value would freeze the binding at import time, and a cache release reaches
# for ``_KOUTER_CMD``/``_KSUM_CMD`` here in order to ``.clear()`` the very dicts
# the op bodies read.
_LIVE_ATTRS = {
    "_CHI_ZERO_FLAG": _caches,
    "_CHI_ZERO_FLAG_MAX": _caches,
    "_CONTIG_CACHE": _caches,
    "_CONTIG_CACHE_MAX": _caches,
    "_SIN_COS_FLOAT_CACHE": _caches,
    "_SIN_COS_FLOAT_CACHE_MAX": _caches,
    "_SIN_COS_ROT_CACHE": _caches,
    "_SIN_COS_ROT_CACHE_MAX": _caches,
    "_functorch": _caches,
    "_KOUTER_ARRAY_SLOTS": kouter,
    "_KOUTER_CMD": kouter,
    "_KOUTER_CMD_MAX": kouter,
    "_KOUTER_VEC2_SLOTS": kouter,
    "_KSUM_ARRAY_SLOTS": ksum,
    "_KSUM_BYPASS_STATS": ksum,
    "_KSUM_CMD": ksum,
    "_KSUM_CMD_MAX": ksum,
    "_KSUM_VEC2_SLOTS": ksum,
    "_KSUM_TILE_MIN_TRIPLES": _ksum_tiled,
    "_KSUM_TILE_STATS": _ksum_tiled,
    "_KSUM_TILES_MAX": _ksum_tiled,
    "_PCTF_SQ_THRESHOLD_CACHE": _sq_aperture,
}


def __getattr__(name: str):
    owner = _LIVE_ATTRS.get(name)
    if owner is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(owner, name)


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(_LIVE_ATTRS))
