"""The three surface models and the 2-D fitter that fits them.

All three models are linear in their coefficients, so :func:`fit_2D` builds a
design matrix and solves a least-squares system rather than iterating -- see
:func:`_linear_design_matrix`. The L-BFGS path is for any other model a
caller passes in.

Provenance: the earlier version of this package descended from py4DSTEM
(GPL-3.0). In 2026-10 it was rewritten clean-room by two teams. A
specification team read the earlier code and described its behaviour in words
and formulas. An implementation team wrote the bodies from that specification,
the kept signatures and docstrings, and the tests; it did not read py4DSTEM or
the earlier bodies. The method is that of Savitzky et al. (2021) and Ophus
(2019). The interface is kept from py4DSTEM so that callers keep it: the names
and coefficient order of ``plane``, ``parabola``, ``bezier_two`` and
``fit_2D``, the name, defaults and return order of ``fit_origin``, and the
CoM helpers in the sibling modules. Two ``fit_origin`` parameters,
``fit_function`` and ``return_fit_params``, are new spellings of py4DSTEM's
``fitfunction`` and ``returnfitp``; quantem (MIT) spells the first one
``fit_function`` too. See THIRD-PARTY-NOTICES.md.
"""

from __future__ import annotations

from inspect import signature

import torch

from ._common import _to_tensor

# ---- fitting basis functions ----------------------------------------------
#
# Each one evaluates a surface model at the coordinates ``xy = (x, y)`` given
# its coefficients, and works elementwise on anything supporting arithmetic
# (float, ndarray, Tensor). All three are linear in their coefficients, so the
# fit itself goes through the design matrix in ``_linear_design_matrix``; these
# are the closed forms used to evaluate a model once its coefficients are known.
# The coefficient ORDER of each signature is the column order of that design
# matrix -- do not permute one without the other.


def plane(xy, mx, my, b):
    """Affine surface ``z = mx*x + my*y + b`` (gradient ``(mx, my)``, offset ``b``)."""
    x, y = xy[0], xy[1]
    return mx * x + my * y + b


def parabola(xy, c0, cx1, cx2, cy1, cy2, cxy):
    """General 2-D quadratic surface, evaluated in Horner form.

    ``z = c0 + cx1*x + cx2*x^2 + cy1*y + cy2*y^2 + cxy*x*y`` -- the complete
    second-order polynomial in two variables, grouped as
    ``c0 + x*(cx1 + cx2*x + cxy*y) + y*(cy1 + cy2*y)`` so each variable is
    touched once.
    """
    x, y = xy[0], xy[1]
    return c0 + x * (cx1 + cx2 * x + cxy * y) + y * (cy1 + cy2 * y)


def _bernstein_quadratic(t):
    """The three degree-2 Bernstein polynomials ``B_k(t) = C(2,k) t^k (1-t)^(2-k)``.

    Returned in order ``k = 0, 1, 2``; they sum to one for any ``t``.
    """
    u = 1 - t
    return u * u, 2 * u * t, t * t


def bezier_two(xy, c00, c01, c02, c10, c11, c12, c20, c21, c22):
    """Tensor-product quadratic Bezier (biquadratic Bernstein) surface.

    ``z = sum_{i,j} c_ij * B_i(x) * B_j(y)`` over the degree-2 Bernstein basis
    of :func:`_bernstein_quadratic`, i.e. a 3x3 grid of control values ``c_ij``
    with ``i`` indexing ``x`` and ``j`` indexing ``y``. Because the basis is a
    partition of unity the surface interpolates the corner coefficients
    (``c00`` at ``(0,0)``, ``c22`` at ``(1,1)``) and stays within the convex
    hull of all nine.
    """
    bx = _bernstein_quadratic(xy[0])
    by = _bernstein_quadratic(xy[1])
    control = ((c00, c01, c02), (c10, c11, c12), (c20, c21, c22))
    return sum(bx[i] * sum(control[i][j] * by[j] for j in range(3)) for i in range(3))


def _make_xy_flat(
    shape: tuple[int, int],
    device: str | torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Unit-square sample coordinates of a ``shape`` grid, one point per column.

    Row 0 is the axis-0 coordinate and row 1 the axis-1 coordinate, each running
    from 0 to 1 inclusive. Point ``n`` is grid index ``(n // shape[1], n %
    shape[1])``, i.e. the grid read in C order.
    """
    rows = torch.linspace(0.0, 1.0, shape[0], device=device, dtype=dtype)
    cols = torch.linspace(0.0, 1.0, shape[1], device=device, dtype=dtype)
    # Broadcast each 1-D axis across the other, then read both out row-major.
    first = rows[:, None].expand(shape[0], shape[1])
    second = cols[None, :].expand(shape[0], shape[1])
    return torch.stack((first.reshape(-1), second.reshape(-1)))


def _bezier_columns(x: torch.Tensor, y: torch.Tensor) -> list[torch.Tensor]:
    """The nine ``B_i(x) * B_j(y)`` products, ``i`` slow, as ``bezier_two`` orders them."""
    along_x = _bernstein_quadratic(x)
    along_y = _bernstein_quadratic(y)
    return [bi * bj for bi in along_x for bj in along_y]


def _linear_design_matrix(function, xy: torch.Tensor) -> torch.Tensor | None:
    """Jacobian of a built-in surface model with respect to its coefficients.

    The three models in this module are linear in their coefficients, so the
    Jacobian does not depend on them: column ``p`` is the basis term that
    coefficient ``p`` multiplies, in the order of the model's signature. The
    result is ``(N, P)`` for the ``N`` points of ``xy``. Any other callable is
    not known to be linear and gives ``None``.
    """
    x, y = xy[0], xy[1]
    if function is plane:
        columns = [x, y, torch.ones_like(x)]
    elif function is parabola:
        columns = [torch.ones_like(x), x, x * x, y, y * y, x * y]
    elif function is bezier_two:
        columns = _bezier_columns(x, y)
    else:
        return None
    return torch.stack(columns, dim=1)


def _fit_linear_model(
    A_all: torch.Tensor,
    z_flat: torch.Tensor,
    mask_flat: torch.Tensor,
    target_shape: tuple[int, int],
    need_pcov: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ordinary least squares on the selected rows of a design matrix.

    Solves ``min ||A p - z||^2`` over the rows where ``mask_flat`` is True and
    returns ``(p, pcov, fit)``. ``fit`` is the model at every row, masked or not,
    reshaped to ``target_shape``. ``pcov`` is ``s^2 (A^T A)^+`` with the residual
    variance ``s^2 = RSS / max(n - P, 1)`` taken over the selected rows.

    ``need_pcov=False`` returns the NaN placeholder ``_fit_nonlinear_model``
    already uses instead of the parameter covariance.  The covariance costs a
    residual pass, a Gram product and a ``pinv`` -- on CUDA that ``pinv`` is a
    cuSOLVER SVD which, even at 3x3, is the single most expensive op in this
    function (0.359 of 0.960 ms measured at a 128x128 field).  Callers that do
    not return it should not pay for it.
    """
    design = A_all[mask_flat]
    n_obs, n_params = int(design.shape[0]), int(design.shape[1])
    if n_obs == 0:
        raise ValueError("No valid points to fit.")
    target = z_flat[mask_flat]
    coeffs = torch.linalg.lstsq(design, target[:, None]).solution[:, 0]
    fit = (A_all @ coeffs).reshape(target_shape)

    if not need_pcov:
        nan_cov = torch.full(
            (n_params, n_params), float("nan"), dtype=coeffs.dtype, device=A_all.device
        )
        return coeffs, nan_cov, fit

    misfit = design @ coeffs - target
    variance = (misfit * misfit).sum() / max(n_obs - n_params, 1)
    cov = variance * torch.linalg.pinv(design.T @ design)
    return coeffs, cov, fit


def _fit_nonlinear_model(
    function,
    xy_all: torch.Tensor,
    z_flat: torch.Tensor,
    mask_flat: torch.Tensor,
    target_shape: tuple[int, int],
    p0: torch.Tensor,
    max_iter: int = 200,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean-squared-error fit of an arbitrary model by L-BFGS.

    ``function(xy, *coefficients)`` is minimised over the selected points in one
    ``LBFGS.step`` (learning rate 1, ``max_iter`` inner iterations, strong-Wolfe
    line search), starting from ``p0``. No covariance is estimated: the second
    output is a NaN ``(P, P)`` matrix. The third is the model at every point,
    reshaped to ``target_shape``. A model that returns something other than a
    tensor is converted, which cuts it off from autograd.
    """
    if not bool(mask_flat.any()):
        raise ValueError("No valid points to fit.")
    xy_used = xy_all[:, mask_flat]
    z_used = z_flat[mask_flat]

    def evaluate(points: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        value = function(points, *coeffs.unbind())
        if isinstance(value, torch.Tensor):
            return value
        return torch.as_tensor(value, dtype=z_flat.dtype, device=z_flat.device)

    coeffs = p0.detach().clone().requires_grad_(True)
    solver = torch.optim.LBFGS(
        [coeffs], lr=1, max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def objective() -> torch.Tensor:
        solver.zero_grad()
        loss = ((evaluate(xy_used, coeffs) - z_used) ** 2).mean()
        loss.backward()
        return loss

    solver.step(objective)

    found = coeffs.detach()
    with torch.no_grad():
        fit = evaluate(xy_all, found).reshape(target_shape)
    n_params = int(found.numel())
    nan_cov = torch.full(
        (n_params, n_params), float("nan"), dtype=found.dtype, device=found.device
    )
    return found, nan_cov, fit


def fit_2D(
    function,
    data: torch.Tensor,
    data_mask: torch.Tensor | None = None,
    popt=None,
    robust: bool = False,
    robust_steps: int = 3,
    robust_thresh: float = 2,
    device: str | torch.device | None = None,
    need_pcov: bool = True,
):
    """
    Fit a surface model ``function(xy, *coefficients)`` to a 2-D field.

    The field is sampled on the unit square (:func:`_make_xy_flat`). The three
    models of this module are solved in closed form by linear least squares;
    any other callable is fitted by L-BFGS from ``popt``. Clean-room
    implementation from a behavioural specification, with the interface kept
    from py4DSTEM (see the module docstring); method of Savitzky et al. (2021)
    and Ophus (2019).

    Parameters
    ----------
    function : callable
        Model whose first argument is the ``(2, N)`` coordinates and whose other
        arguments are the ``P`` coefficients.
    data : Tensor (H, W)
        Field to fit; it is read as float32 and fitted in float64.
    data_mask : Tensor (H, W), optional
        Boolean; only points where it is True enter the fit. Default: all.
    popt : sequence or Tensor, optional
        Initial coefficients, ``P`` of them. Default: zeros. Used only by the
        L-BFGS path.
    robust : bool
        Re-fit ``robust_steps`` more times, each time removing from the mask the
        points whose squared residual is larger than ``robust_thresh**2`` times
        the mean squared residual over the whole field.
    robust_steps : int
    robust_thresh : float
    device : str, torch.device or None
        Default: the device of ``data`` if it is a tensor, else CPU.
    need_pcov : bool
        ``False`` skips the parameter covariance of a linear basis and
        returns the NaN placeholder in its slot; everything else is unchanged.
        Default ``True``.

    Returns
    -------
    popt : Tensor (P,), float64
    pcov : Tensor (P, P), float64
    fit : Tensor (H, W), float32
    mask : Tensor (H, W), bool
        The points used by the last fit.
    """
    field = _to_tensor(data, dtype=torch.float32, device=device)
    device = field.device
    field = field.to(torch.float64)
    if field.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {tuple(field.shape)}")
    shape: tuple[int, int] = (int(field.shape[0]), int(field.shape[1]))

    if data_mask is None:
        keep = torch.ones(shape, dtype=torch.bool, device=device)
    else:
        # Clone: _to_tensor gives back the caller's tensor when it is already
        # bool on this device, and the returned mask must not alias it.
        keep = _to_tensor(data_mask, dtype=torch.bool, device=device).clone()
        if tuple(keep.shape) != shape:
            raise ValueError(
                f"data_mask must have shape {shape}, got {tuple(keep.shape)}"
            )

    n_params = len(signature(function).parameters) - 1
    if popt is None:
        coeffs = torch.zeros(n_params, dtype=torch.float64, device=device)
    else:
        coeffs = _to_tensor(popt, dtype=torch.float64, device=device).reshape(-1)
        if coeffs.numel() != n_params:
            raise ValueError(
                f"Initial popt has {coeffs.numel()} params, expected {n_params}"
            )

    xy = _make_xy_flat(shape, device=device, dtype=torch.float64)
    design = _linear_design_matrix(function, xy)
    z = field.reshape(-1)

    def fit_once(start, mask):
        if design is not None:
            return _fit_linear_model(
                design, z, mask.reshape(-1), shape, need_pcov=need_pcov
            )
        return _fit_nonlinear_model(function, xy, z, mask.reshape(-1), shape, start)

    coeffs, cov, surface = fit_once(coeffs, keep)
    for _ in range(robust_steps if robust else 0):
        # Outlier rejection: the cut is relative to the mean over the
        # whole field, and a point once dropped stays dropped.
        sq_res = (surface - field) ** 2
        keep = keep & (sq_res <= sq_res.mean() * robust_thresh**2)
        coeffs, cov, surface = fit_once(coeffs, keep)

    return coeffs, cov, surface.to(torch.float32), keep
