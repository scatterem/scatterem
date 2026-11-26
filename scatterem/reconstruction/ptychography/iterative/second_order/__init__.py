"""Second-order reconstruction of 2D mixed-state ptychography.

* :class:`PlanarPtychography` -- the 2D model: probe modes, one object slice,
  refined scan offsets and the measured amplitudes, with an optional detector
  mask and the per-pass constraints of :class:`PlanarConstraints`;
* :func:`solve_ptychography` -- the one-call driver over the solvers of
  :mod:`scatterem.optim.second_order`; ``solver="secant_gd"`` is minibatch
  gradient descent with per-group Newton step sizes from a secant curvature
  estimate;
* :class:`PassStopRule`, :class:`PassStopCallback` -- the stop and return
  rules of the solver passes;
* :class:`SolveTarget` -- the protocol between the driver and a model.
"""

from .driver import solve_ptychography as solve_ptychography
from .planar import PlanarPtychography as PlanarPtychography
from .planar_constraints import PlanarConstraints as PlanarConstraints
from .stop_rule import PassStopCallback as PassStopCallback
from .stop_rule import PassStopRule as PassStopRule
from .target import SolveTarget as SolveTarget

__all__ = [
    "PassStopCallback",
    "PassStopRule",
    "PlanarConstraints",
    "PlanarPtychography",
    "SolveTarget",
    "solve_ptychography",
]
