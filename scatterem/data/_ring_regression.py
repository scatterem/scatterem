"""Robust geometric least-squares fit of a circle to points.

The fit minimizes the geometric residuals ``e_k = |p_k - c| - R`` by
iteratively reweighted Gauss-Newton steps (Chernov 2010). Geometric residuals
avoid the bias of algebraic fits (Kasa 1976; Pratt 1987; Taubin 1991), which
grows on short arcs (Al-Sharadqah and Chernov 2009). The weights are the
Tukey biweight with constant 4.685 (Beaton and Tukey 1974; Huber and
Ronchetti 2009). The scale is the normalized median absolute deviation of the
residuals (Rousseeuw and Leroy 1987).

Literature: Kasa (1976), IEEE Trans. IM-25, 8-14; Pratt (1987), Computer
Graphics 21(4), 145-152; Taubin (1991), IEEE Trans. PAMI 13, 1115-1138;
Chernov (2010), *Circular and Linear Regression*; Al-Sharadqah and Chernov
(2009), Electron. J. Statist. 3, 886-911; Huber and Ronchetti (2009),
*Robust Statistics*; Rousseeuw and Leroy (1987), *Robust Regression and
Outlier Detection*; Beaton and Tukey (1974), Technometrics 16, 147-185.
This code was written for scatterem from this literature.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import torch

logger = logging.getLogger(__name__)

_TUKEY_C = 4.685
_MAD_TO_SIGMA = 1.4826


class CircleFit(NamedTuple):
    """Result of the robust circle fit.

    Attributes
    ----------
    center : tuple of float
        ``(y, x)`` of the circle [pixel].
    radius : float
        The radius [pixel].
    weights : torch.Tensor
        The final weight of each point, 0 for a rejected point.
    scale : float
        The final residual scale [pixel].
    """

    center: tuple[float, float]
    radius: float
    weights: torch.Tensor
    scale: float


def tukey_weight(z: torch.Tensor) -> torch.Tensor:
    """Return the Tukey biweight ``(1 - (z / 4.685)^2)^2``, 0 for ``|z| >= 4.685``."""
    inside = z.abs() < _TUKEY_C
    return torch.where(inside, (1.0 - (z / _TUKEY_C) ** 2) ** 2, 0.0)


def robust_circle(
    points: torch.Tensor,
    valid: torch.Tensor,
    start: tuple[float, float, float],
    *,
    scale_floor: float = 0.15,
    max_steps: int = 30,
    tol: float = 1e-6,
) -> CircleFit:
    """Fit a circle to 2-D points with Tukey-biweight Gauss-Newton steps.

    Parameters
    ----------
    points : torch.Tensor
        Shape ``(K, 2)``, float64, the points ``(y, x)`` [pixel].
    valid : torch.Tensor
        Shape ``(K,)``, bool. Only valid points enter the fit.
    start : tuple of float
        Start values ``(c_y, c_x, R)`` [pixel].
    scale_floor : float, optional
        The smallest residual scale [pixel]. It stops a noise-free fit from
        the rejection of points for small lattice errors.
    max_steps : int, optional
        The largest number of Gauss-Newton steps.
    tol : float, optional
        Stop when the largest step is below this value [pixel].

    Returns
    -------
    CircleFit
        The fitted circle and the final weights.

    Raises
    ------
    ValueError
        If fewer than 3 points are valid, or the normal matrix is singular.
    """
    if int(valid.sum().item()) < 3:
        raise ValueError("circle fit: fewer than 3 valid edge points")
    params = torch.tensor(start, dtype=torch.float64, device=points.device)
    good = valid.to(torch.float64)
    weights = good
    scale = scale_floor
    for _ in range(max_steps):
        diff = points - params[:2]
        dist = torch.linalg.vector_norm(diff, dim=1).clamp_min(1e-12)
        res = dist - params[2]
        res_valid = res[valid]
        mad = (res_valid - res_valid.median()).abs().median()
        scale = max(_MAD_TO_SIGMA * float(mad.item()), scale_floor)
        weights = tukey_weight(res / scale) * good
        jac = torch.cat((-diff / dist[:, None], -torch.ones_like(dist)[:, None]), 1)
        normal = jac.T @ (weights[:, None] * jac)
        rhs = -(jac.T @ (weights * res))
        try:
            step = torch.linalg.solve(normal, rhs)
        except RuntimeError as err:  # torch.linalg.LinAlgError
            raise ValueError(
                "circle fit: the normal matrix is singular; too few edge "
                "points support the circle"
            ) from err
        if not bool(torch.isfinite(step).all()):
            raise ValueError("circle fit: the Gauss-Newton step is not finite")
        params = params + step
        if float(step.abs().max().item()) < tol:
            break
    center = (float(params[0].item()), float(params[1].item()))
    logger.debug(
        "circle fit: centre %s, radius %.4f, scale %.4f",
        center,
        float(params[2].item()),
        scale,
    )
    return CircleFit(center, float(params[2].item()), weights, scale)
