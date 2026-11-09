"""Fit and subtract the descan: the slow drift of the CoM across the scan."""

from __future__ import annotations

import torch

from ._common import _to_tensor
from ._surfaces import (
    _linear_design_matrix,
    _make_xy_flat,
    bezier_two,
    fit_2D,
    parabola,
    plane,
)

#: Surface models :func:`fit_origin` will fit, keyed by the name callers pass.
_ORIGIN_SURFACES = {"plane": plane, "parabola": parabola, "bezier_two": bezier_two}


def _normal_equations_solve(A: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """Least-squares ``popts`` of ``A @ popts = Z`` by the normal equations.

    Both products are formed as REDUCTIONS over the observations.  cuSOLVER's
    lstsq (and ``A.T @ A``) put the (P, n_obs) x (n_obs, .) product on a cutlass
    FP64 GEMM that walks n_obs serially for P <= 9: ~0.5 ms at a 32x32 scan, vs
    ~0.13 ms here.  The basis is evaluated on a [0, 1] grid, so cond(A) <= ~30
    and squaring it costs nothing a float32 field can see (popts within ~3e-14
    of lstsq's).  Needs ``n_obs >= P``; no error check, so no host sync.
    """
    At, Zt = A.T, Z.T
    gram = (At[:, None, :] * At[None, :, :]).sum(-1)  # (P, P)
    rhs = (At[:, None, :] * Zt[None, :, :]).sum(-1)  # (P, R)
    return torch.linalg.solve_ex(gram, rhs)[0]


def _fit_2D_shared(
    function,
    planes: tuple[torch.Tensor, ...],
    data_mask: torch.Tensor | None,
    device: str | torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] | None:
    """:func:`fit_2D` for several planes that share one least-squares system.

    :func:`fit_origin` fits the x- and y-components of the origin with the
    same surface model, over the same coordinate grid, under the same point
    mask; only the right-hand side differs. Spelled as two :func:`fit_2D`
    calls that is two coordinate grids, two design matrices, two boolean
    gathers of ``A_all`` (each a ``nonzero``, hence a host sync) and two
    least-squares factorisations of the *same* matrix. Shared, the
    factorisation is paid once and the components ride it as extra
    right-hand sides.

    Returns ``None`` -- meaning "use the per-component path" -- when the
    basis is not linear in its coefficients, so the LBFGS fallback in
    ``_fit_nonlinear_model`` is never reached from here. Only the
    ``robust=False`` / ``need_pcov=False`` case is handled, which is the one
    every library consumer takes; the robust iteration diverges the two masks
    after its first step, and the covariance is per-component work that
    ``_fit_linear_model`` skips when asked to.
    """
    planes64 = []
    for measured in planes:
        t = _to_tensor(measured, dtype=torch.float32, device=device).to(torch.float64)
        if t.ndim != 2:
            raise ValueError(f"data must be 2D, got shape {tuple(t.shape)}")
        if planes64 and t.shape != planes64[0].shape:
            raise ValueError("all planes must have the same shape")
        planes64.append(t)
    shape: tuple[int, int] = (int(planes64[0].shape[0]), int(planes64[0].shape[1]))

    xy = _make_xy_flat(shape, device=device, dtype=torch.float64)  # (2, N)
    A_all = _linear_design_matrix(function, xy)
    if A_all is None:
        return None

    if data_mask is None:
        mask = torch.ones(shape, dtype=torch.bool, device=device)
    else:
        mask = _to_tensor(data_mask, dtype=torch.bool, device=device)
        if mask.shape != shape:
            raise ValueError(
                f"data_mask must have shape {shape}, got {tuple(mask.shape)}"
            )
    mask_flat = mask.reshape(-1)

    A = A_all[mask_flat]
    if A.shape[0] == 0:
        raise ValueError("No valid points to fit.")
    Z = torch.stack([p.reshape(-1) for p in planes64], dim=1)[mask_flat]  # (n_obs, R)

    if A.is_cuda and A.shape[0] >= A.shape[1]:
        popts = _normal_equations_solve(A, Z)  # (P, R)
    else:
        popts = torch.linalg.lstsq(A, Z).solution  # (P, R)
    fits = (A_all @ popts).T.reshape(len(planes64), *shape)

    n_params = int(A.shape[1])
    out = []
    for j in range(len(planes64)):
        pcov = torch.full(
            (n_params, n_params), float("nan"), dtype=torch.float64, device=device
        )
        out.append((popts[:, j], pcov, fits[j].to(dtype=torch.float32), mask.clone()))
    return out


def fit_origin(
    data: tuple[torch.Tensor, torch.Tensor],
    mask: torch.Tensor | None = None,
    fit_function: str = "plane",
    return_fit_params: bool = False,
    robust: bool = False,
    robust_steps: int = 3,
    robust_thresh: float = 2,
    device: str | torch.device | None = None,
):
    """
    Fit the drift of the diffraction-space origin across the scan.

    The two components of the measured origin are fitted independently, each
    with the same smooth surface model, and the residuals returned alongside.

    Parameters
    ----------
    data : (Tensor, Tensor)
        Measured origin, as (x-component, y-component), each (Rx, Ry).
    mask : Tensor, optional
        Boolean (Rx, Ry); scan positions where False are excluded from the fit
        but still get a fitted value and a residual.
    fit_function : str
        ``'constant'`` (per-component mean) or one of the keys of
        :data:`_ORIGIN_SURFACES`.
    return_fit_params : bool
        Also return the coefficients and covariances of the two fits. They are
        ``None`` for ``fit_function='constant'``, which has no fitted model.
    robust, robust_steps, robust_thresh
        Outlier rejection, forwarded to :func:`fit_2D`.
    device : str, torch.device or None
        Default: the device of ``data[0]`` if it is a tensor, else CPU.

    Returns
    -------
    (qx0_fit, qy0_fit, qx0_residuals, qy0_residuals)
        Each (Rx, Ry). If ``return_fit_params``, a 2-tuple of that and
        ``(popt_x, popt_y, pcov_x, pcov_y)``.

    Notes
    -----
    Clean-room implementation from a behavioural specification, with the
    interface kept from py4DSTEM, except that ``fit_function`` and
    ``return_fit_params`` are new spellings of ``fitfunction`` and
    ``returnfitp`` (see :mod:`._surfaces`). The method is that of
    Savitzky et al. (2021) and Ophus (2019): the descan is a slow, smooth drift, so
    a low-order surface fitted to each component removes it and leaves the
    specimen signal in the residual.
    """
    if not isinstance(data, (tuple, list)) or len(data) != 2:
        raise ValueError("data must be a 2-tuple of (x, y) origin components")
    comp_x = _to_tensor(data[0], dtype=torch.float32, device=device)
    device = comp_x.device
    comp_y = _to_tensor(data[1], dtype=torch.float32, device=device)
    if comp_x.ndim != 2 or comp_x.shape != comp_y.shape:
        raise ValueError(
            "both origin components must be 2-D and the same shape, got "
            f"{tuple(comp_x.shape)} and {tuple(comp_y.shape)}"
        )

    if fit_function == "constant":
        # No model: the fitted origin is each component's average.
        fit_x = comp_x.mean().expand_as(comp_x).clone()
        fit_y = comp_y.mean().expand_as(comp_y).clone()
        params = (None, None, None, None)
    else:
        if fit_function not in _ORIGIN_SURFACES:
            known = ["constant", *_ORIGIN_SURFACES]
            raise ValueError(
                f"unknown fit_function {fit_function!r}; expected one of {known}"
            )
        surface = _ORIGIN_SURFACES[fit_function]
        shared = None
        if not (robust or return_fit_params):
            shared = _fit_2D_shared(surface, (comp_x, comp_y), mask, device)
        if shared is None:
            shared = [
                fit_2D(
                    surface,
                    component,
                    data_mask=mask,
                    robust=robust,
                    robust_steps=robust_steps,
                    robust_thresh=robust_thresh,
                    device=device,
                    need_pcov=return_fit_params,
                )
                for component in (comp_x, comp_y)
            ]
        (popt_x, pcov_x, fit_x, _), (popt_y, pcov_y, fit_y, _) = shared
        params = (popt_x, popt_y, pcov_x, pcov_y)

    surfaces = (fit_x, fit_y, comp_x - fit_x, comp_y - fit_y)
    if return_fit_params:
        return surfaces, params
    return surfaces
