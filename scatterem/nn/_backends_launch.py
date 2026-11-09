"""The ``scatterem.nn`` backend rows for host-issue routes.

Split from :mod:`scatterem.nn._backends` (module size), like
:mod:`scatterem.nn._backends_graphs`.  Each row picks between a route that issues
the same kernels with less host work -- a recorded Warp launch, an eager call of a
custom op's body, or one kernel in place of three launches -- and the reference
route it stands in for.  Predicates are ``"module:function"`` strings, so importing this module imports no kernel
module.
"""

from __future__ import annotations

from scatterem.utils.env import Backend, register_backend


_PP = "scatterem.nn.functional.warp.prism_project"




# -- PRISM projection rows ---------------------------

#: The tiled PRISM ``grad_Sdt`` gather issued from one reusable ``wp.Launch`` per
#: (kernel, device, thread), every slot rebound per call
#: (``prism_project/_sdt_recorded.py``).  Bit-identical to ``wp.launch``.
PRISM_GRAD_SDT_LAUNCH = register_backend(
    Backend(
        name="prism_grad_sdt_launch",
        impls=("recorded", "launch"),
        accepts={"recorded": (f"{_PP}._sdt_recorded:grad_sdt_recorded_supported",)},
    )
)

#: The no-grad PRISM forward custom ops ``prism_ramp_mul`` and
#: ``prism_project_fwd`` run their op bodies directly in eager mode
#: (``prism_project/_fwd_eager.py``).  Bit-identical to the op dispatch.
PRISM_FWD_EAGER = register_backend(
    Backend(
        name="prism_fwd_eager",
        impls=("eager", "op"),
        accepts={"eager": (f"{_PP}._fwd_eager:prism_fwd_eager_supported",)},
    )
)


