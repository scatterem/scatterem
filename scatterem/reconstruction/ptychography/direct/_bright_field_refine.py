"""Float64 LBFGS refinement of the bright-field-shift aberration fit.

Used by :func:`.bright_field.fit_aberrations_and_rotation_to_bright_field_shifts`,
which seeds it. The refinement used to run in float32 with 1e-10 stopping
tolerances: the loss is ~1e-4 A^2, so a first step that changed it by less than
float32 rounding ended LBFGS at its seed, and whether it did depended on ~1e-7
differences in the input shifts. In float64, with parameters scaled by their
shift sensitivity, it converges and repeats bit for bit.
"""

from dataclasses import dataclass

import numpy as np
import torch

from scatterem.utils.transfer import aberrations_to_image_shifts

#: Index one past the last coefficient of each order (``correct_order``): 0 is
#: C10 only, 1 adds C12a/b, 2 adds C21a/b and C23a/b, 3 adds C30, C32a/b, C34a/b.
ORDER_END = {0: 1, 1: 3, 2: 7, 3: 12}
#: float64 stopping tolerances. The loss is a mean Huber value of order the
#: squared shift residual in Angstrom^2 and the parameters are O(1) in
#: preconditioned units, so 1e-12 is ~1e4x above float64 rounding of either.
_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ShiftProblem:
    """The fixed data of one shift fit: float64 targets of the included pixels."""

    target: torch.Tensor
    include: torch.Tensor
    k_sampling: np.ndarray
    wavelength: float
    mask: torch.Tensor

    def predict(self, aberrations, rotation):
        """Predicted shifts of the included pixels, shape ``(n, 2)``."""
        shifts = aberrations_to_image_shifts(
            aberrations, rotation, self.k_sampling, self.wavelength, self.mask.shape
        )
        return shifts[self.mask][self.include]

    def loss(self, aberrations, rotation):
        """The fit objective: mean Huber loss of the included shift components."""
        return torch.nn.functional.huber_loss(
            self.predict(aberrations, rotation), self.target
        )


def _precondition_scales(problem: ShiftProblem, rotation) -> torch.Tensor:
    """Per-coefficient scale ``1 / RMS(d shift / d C_i)`` over the included pixels.

    The shift is linear in the coefficients, so column ``i`` is the prediction
    for a unit coefficient ``i``. In these units the Hessian is close to
    isotropic; in Angstrom the columns span about four decades.
    """
    eye = torch.eye(12, dtype=torch.float64, device=problem.mask.device)
    with torch.no_grad():
        rms = torch.stack(
            [problem.predict(e, rotation).square().mean().sqrt() for e in eye]
        )
    return torch.where(rms > 0, 1.0 / rms.clamp_min(1e-300), torch.ones_like(rms))


def refine_shift_fit(
    problem: ShiftProblem,
    seed: torch.Tensor,
    rotation: torch.Tensor,
    free: torch.Tensor,
    fit_rotation: bool,
    max_iter: int,
    lr: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimize the shift loss over the ``free`` coefficients (and the rotation).

    Parameters
    ----------
    problem : ShiftProblem
        Targets, include mask and forward-model geometry.
    seed : torch.Tensor
        Initial 12 coefficients, in Angstrom; the frozen ones keep their value.
    rotation : torch.Tensor
        Initial scan rotation in degrees, shape ``(1,)``.
    free : torch.Tensor
        Boolean ``(12,)`` mask of the coefficients that may move.
    fit_rotation : bool
        Whether the rotation is optimized too.
    max_iter : int
        LBFGS iteration cap.
    lr : float
        LBFGS learning rate.

    Returns
    -------
    aberrations : torch.Tensor
        Refined coefficients, float64.
    rotation : torch.Tensor
        Refined rotation in degrees, float64, shape ``(1,)``.
    """
    seed = seed.detach().to(torch.float64)
    rotation = rotation.detach().to(torch.float64).clone()
    scales = _precondition_scales(problem, rotation)[free]
    x = torch.zeros(
        int(free.sum()), dtype=torch.float64, device=seed.device, requires_grad=True
    )
    params = [x]
    if fit_rotation:
        rotation.requires_grad_(True)
        params.append(rotation)

    def unpack():
        ab = seed.clone()
        ab[free] = seed[free] + scales * x
        return ab

    optimizer = torch.optim.LBFGS(
        params,
        lr=lr,
        max_iter=max_iter,
        max_eval=None,
        tolerance_grad=_TOLERANCE,
        tolerance_change=_TOLERANCE,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        loss = problem.loss(unpack(), rotation)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        return unpack(), rotation.detach()
