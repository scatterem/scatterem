"""Bilinear-Hessian second-order solvers and oracles.

Implements the framework of Carlsson, Nikitin, Troedsson & Wendt:

* "The bilinear Hessian for large scale optimization", arXiv:2502.03070
* "Efficient near-field ptychography reconstruction using the Hessian
  operator", Opt. Express 33(14):30543 (2025), arXiv:2502.10755

Provides three solvers (``bh_gd``, ``bh_cg``, ``bh_qn``) that consume a
Hessian *oracle*: an object exposing ``loss()``, ``grad()`` and the Hessian
operator ``hess_op(v)`` over a list of (complex or real) parameter tensors,
with the real inner product ``rdot``.  The oracles here are generic:

* :class:`AutogradOracle` — Hessian-vector products via double-backward
  autograd.  Works on any differentiable closure.
* :class:`ChunkedAutogradOracle` — the same, accumulated over batches.
* :class:`LinearLeastSquaresOracle` — exact for linear least squares (used by
  linear tomography's BH-CG path).

"Second order" rather than "Gauss-Newton": the autograd oracles use the full
Hessian, residual term included; Gauss-Newton is an option of the analytic
ptychography oracle only.  The ptychography problem, its analytic oracle, the
overlap preconditioner and the one-call driver live in
:mod:`scatterem.reconstruction.ptychography.iterative.second_order`, because they import ``nn``
and this package is layer 1 (no scatterem imports).

Design notes: parameters live in their natural complex tensor spaces; the
inner product is ``Re<x, y>`` summed over the parameter list, which matches
PyTorch's convention for gradients of real losses w.r.t. complex leaves.
Solver iterates mutate the parameter tensors in place (under ``no_grad``).

One module per layer: the inner-product and
step primitives (:mod:`.primitives`), the per-group slicing and alphas
(:mod:`.groups`), the forward-mode derivative by double VJP
(:mod:`.autodiff`), the Hessian oracles (:mod:`.oracles`, :mod:`.linear_oracle`),
the iteration record (:mod:`.log`), the solvers (:mod:`.solvers`, with BH-CG in
:mod:`.cg`), and the batch step, pass
record and final-state selection of ``bh_gd_incremental``
(:mod:`._incremental`, :mod:`.selection`).  The private helpers are
re-exported beside ``__all__`` -- four of them are imported by name from the
guard and driver tests, and the six adaptive-pass constants are re-exported
for the same attribute-surface reason. The constants are read by
``.solvers`` at call time from ITS globals, so substituting one here would
not reach the solver; patch ``.solvers``.

One thing the move did NOT decide: the 23 ``zip()`` calls now carry an
explicit ``strict=False``, which is the behavior they already had.  It is not
an audit.  ``_step_groups`` in particular zips ``params`` against ``alphas``,
which are per-GROUP -- equal in length only while groups default to one per
parameter -- so flipping those to ``strict=True`` would turn a silent
truncation into an exception in custom-group runs.  That call is worth making,
with the solver semantics in hand, and not inside a file move.
"""

from .cg import bh_cg as bh_cg
from .groups import GroupSlice as GroupSlice
from .groups import _basis_from_groups as _basis_from_groups
from .groups import _default_groups as _default_groups
from .groups import _per_group_alphas as _per_group_alphas
from .groups import _step_groups as _step_groups
from .groups import _step_groups_sliced as _step_groups_sliced
from .linear_oracle import LinearLeastSquaresOracle as LinearLeastSquaresOracle
from .log import BHLog as BHLog
from .oracles import AutogradOracle as AutogradOracle
from .oracles import ChunkedAutogradOracle as ChunkedAutogradOracle
from .primitives import _is_missing_double_backward as _is_missing_double_backward
from .primitives import _newton_alpha as _newton_alpha
from .primitives import _rho2 as _rho2
from .primitives import _step as _step
from .primitives import rdot as rdot
from .solvers import _ADAPTIVE_ACCEPT_STRIDE as _ADAPTIVE_ACCEPT_STRIDE
from .solvers import _ADAPTIVE_ACCEPT_WARMUP as _ADAPTIVE_ACCEPT_WARMUP
from .solvers import _EARLY_STOP_PATIENCE as _EARLY_STOP_PATIENCE
from .solvers import _MIN_STEPS_PER_PASS as _MIN_STEPS_PER_PASS
from .solvers import _PASS_REVERT_TOL as _PASS_REVERT_TOL
from .solvers import _STALL_FRAC as _STALL_FRAC
from .solvers import bh_gd as bh_gd
from .solvers import bh_gd_incremental as bh_gd_incremental
from .solvers import bh_qn as bh_qn

__all__ = [
    "rdot",
    "AutogradOracle",
    "ChunkedAutogradOracle",
    "LinearLeastSquaresOracle",
    "BHLog",
    "bh_gd",
    "bh_gd_incremental",
    "bh_cg",
    "bh_qn",
]
