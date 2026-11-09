"""PRISM exit-wave projection: the fused Warp route and its torch oracle.

Split out of the former flat ``prism_project.py`` (2,283 lines).  The seam is
the one the file already had: the Warp KERNELS on one side and the torch that
drives them on the other.

  _funcs.py         the generic complex wp.func helpers both kernel sets use
  torch_oracle.py   the pure-torch reference the Warp route is graded against
  _fwd_kernels.py   forward Warp kernels + their fp16/fp32 overloads
  _bwd_kernels.py   backward Warp kernels (basis adjoint, S-matrix scatter-add)
  _sdt_gather_kernels.py  the S-matrix adjoint's gather kernels (with accumulate)
  _sdt_launch.py    the grad_Sdt launch, and the shared-accumulator custom_op
  ops.py            the two custom_ops and the autograd wiring between them
  _chunked.py       the chunked Warp route under grad as ONE autograd node
  _torch_adjoint.py the torch oracle's hand ``Sdt`` adjoint
  _ramp_kernels.py  Fourier-shift ramp kernels: the adjoint and the product
  ramp.py           the ramp/IFFT half in torch, around those kernels
  dispatch.py       prism_project, and the Warp route it dispatches to

WHAT THE SPLIT DOES AND DOES NOT CHANGE FOR WARP.  Each module is now its own
``wp.context.Module``, so the kernels are compiled and cached per module instead
of as one unit: the first run after this change recompiles and writes new cache
entries under a new set of module hashes.  The kernel SOURCE is unchanged, and
so are the launch parameters, the tile widths and the ``wp.overload``
declarations -- each stayed with the kernel it belongs to.  Cross-module
``wp.func`` calls are what ``warp/ptychography.py`` already does with
``scatterem.utils.warp``'s ``cmul``/``cexp``, so ``_funcs`` is that arrangement
one directory over.  The output was checked bit-exact against the monolith --
forward and all three gradients, digested, over 40 live combinations of device,
precision, periodicity, chunking and sub-pixel ramp, plus the 24 that raise.

This package reproduces the module's entire attribute surface, private names
included.  Module-level data is served by ``__getattr__`` from its owner;
setting an attribute here does not reach that owner.
"""

from __future__ import annotations

from . import _bwd_kernels as _bwd_kernels
from . import _fwd_kernels as _fwd_kernels
from . import _ramp_kernels as _ramp_kernels
from . import _sdt_gather_kernels as _sdt_gather_kernels
from . import torch_oracle as torch_oracle
from ._bwd_kernels import (
    _prism_project_bwd_basis_bcast_kernel as _prism_project_bwd_basis_bcast_kernel,
)
from ._bwd_kernels import (
    _prism_project_bwd_basis_bcast_tiled_kernel as _prism_project_bwd_basis_bcast_tiled_kernel,
)
from ._bwd_kernels import (
    _prism_project_bwd_basis_kernel as _prism_project_bwd_basis_kernel,
)
from ._bwd_kernels import _prism_project_bwd_sdt_kernel as _prism_project_bwd_sdt_kernel
from ._chunked import _prism_project_warp_chunked as _prism_project_warp_chunked
from ._chunked import _PrismProjectChunked as _PrismProjectChunked
from ._funcs import _cconj as _cconj
from ._funcs import _cmul as _cmul
from ._funcs import _fwd_prod as _fwd_prod
from ._fwd_kernels import (
    _make_prism_project_fwd_tiled_kernel as _make_prism_project_fwd_tiled_kernel,
)
from ._fwd_kernels import _prism_project_fwd_kernel as _prism_project_fwd_kernel
from ._ramp_kernels import (
    _make_prism_ramp_adjoint_kernel as _make_prism_ramp_adjoint_kernel,
)
from ._ramp_kernels import _make_prism_ramp_mul_kernel as _make_prism_ramp_mul_kernel
from ._ramp_kernels import _prism_ramp_adjoint as _prism_ramp_adjoint
from ._ramp_kernels import _prism_ramp_mul as _prism_ramp_mul
from ._ramp_kernels import _prism_ramp_mul_kernel as _prism_ramp_mul_kernel
from ._ramp_kernels import _ramp_mul_tile as _ramp_mul_tile
from ._sdt_gather_kernels import (
    _make_prism_project_bwd_sdt_gather_tiled_kernel as _make_prism_project_bwd_sdt_gather_tiled_kernel,
)
from ._sdt_gather_kernels import (
    _prism_project_bwd_sdt_gather_kernel as _prism_project_bwd_sdt_gather_kernel,
)
from ._sdt_launch import _launch_grad_sdt as _launch_grad_sdt
from ._sdt_launch import _prism_project_bwd_sdt_accum as _prism_project_bwd_sdt_accum
from ._torch_adjoint import _CropContractFrozenBasis as _CropContractFrozenBasis
from ._torch_adjoint import _periodic_segments as _periodic_segments
from .dispatch import _prism_project_warp as _prism_project_warp
from .dispatch import prism_project as prism_project
from .ops import _prism_project_bwd as _prism_project_bwd
from .ops import _prism_project_fwd as _prism_project_fwd
from .ops import _prism_project_fwd_backward as _prism_project_fwd_backward
from .ops import _prism_project_fwd_setup_context as _prism_project_fwd_setup_context
from .ramp import _build_basis_rs as _build_basis_rs
from .ramp import _ramp_mul as _ramp_mul
from .ramp import _RampIFFT2 as _RampIFFT2
from .ramp import _shift_and_ifft2 as _shift_and_ifft2
from .torch_oracle import _prism_project_torch as _prism_project_torch
from .torch_oracle import _ramp_is_identity as _ramp_is_identity
from .torch_oracle import _torch_project_block as _torch_project_block

# Module-level data, served live from the module that owns it. The two
# pre-instantiated tiled kernels are here too: they are module-level OBJECTS
# built by a factory at import, and a by-value re-export would pin whichever one
# existed first rather than track the owner.
_LIVE_ATTRS = {
    "_TORCH_PROJECT_BLOCK_BYTES": torch_oracle,
    "_PRISM_FWD_TILE_ACC": _fwd_kernels,
    "_PRISM_FWD_TILE_IDX": _fwd_kernels,
    "_PRISM_FWD_TILE_K": _fwd_kernels,
    "_PRISM_FWD_TILE_K_NARROW": _fwd_kernels,
    "_prism_project_fwd_tiled_kernel": _fwd_kernels,
    "_prism_project_fwd_tiled_narrow_kernel": _fwd_kernels,
    "_PRISM_BWD_BASIS_BLOCK_DIM": _bwd_kernels,
    "_PRISM_BWD_BASIS_K_UNROLL": _bwd_kernels,
    "_PRISM_BWD_BASIS_TILE_ACC": _bwd_kernels,
    "_PRISM_BWD_BASIS_TILE_P": _bwd_kernels,
    "_PRISM_BWD_SDT_TILE_P": _sdt_gather_kernels,
    "_PRISM_BWD_SDT_TILE_P_KB0": _sdt_gather_kernels,
    "_SDT_GATHER_MAX_WASTE": _sdt_gather_kernels,
    "_prism_project_bwd_sdt_gather_tiled_kb0_kernel": _sdt_gather_kernels,
    "_prism_project_bwd_sdt_gather_tiled_kernel": _sdt_gather_kernels,
    "_RAMP_ADJOINT_KERNELS": _ramp_kernels,
    "_RAMP_ADJOINT_TILES": _ramp_kernels,
    "_RAMP_MUL_KERNELS": _ramp_kernels,
    "_RAMP_MUL_TILE": _ramp_kernels,
    "_RAMP_MUL_TILES": _ramp_kernels,
}


def __getattr__(name: str):
    owner = _LIVE_ATTRS.get(name)
    if owner is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(owner, name)


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(_LIVE_ATTRS))
