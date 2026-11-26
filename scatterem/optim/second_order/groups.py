"""Per-group step slicing and the per-group Newton alphas."""

from __future__ import annotations

import math
from collections.abc import MutableSequence, Sequence

import torch
from torch import Tensor

from .primitives import (
    _is_missing_double_backward,
    _scalars_to_host,
    _structural_zeros_like,
    rdot,
)


def _per_group_alphas(
    oracle,
    g: Sequence[Tensor],
    s: Sequence[Tensor],
    fallback: float,
    method: str = "exact",
    eps_hints: Sequence[float | None] | None = None,
    basis: Sequence[Sequence[Tensor]] | None = None,
) -> tuple:
    """Per-group Newton step sizes via the G x G bilinear-Hessian solve.

    With group directions ``e_i`` (s restricted to parameter group i), the
    local quadratic model is minimized over ``sum_i alpha_i e_i`` by solving
    ``A alpha = -b`` with ``A_ij = H(e_i, e_j)`` and ``b_i = <g, e_i>``.
    This is the structural advantage LSQML exploits.  Only scalars are
    needed: if the oracle provides ``hess_quad_matrix`` all A_ij come from
    ONE fused sweep over the data; otherwise one ``hess_op`` per group.

    Returns ``(alphas, hss)`` with ``hss = H(s, s)`` (used by callers for
    curvature checks).  Groups with a vanishing direction get ``alpha = 0``;
    if the reduced system is not positive definite the uniform scalar
    Newton fallback is used for all groups.
    """
    if basis is None:
        basis = [
            [sj if j == i else _structural_zeros_like(sj) for j, sj in enumerate(s)]
            for i in range(len(s))
        ]
    G = len(basis)
    A_full: Tensor | None = None
    A_terms: list[Tensor] = []
    if method == "fd":
        # secant path: G plain gradient sweeps, no double-backward
        A_full = oracle.hess_quad_matrix_fd(basis, g, eps_hints=eps_hints).to(
            torch.float64
        )
    elif hasattr(oracle, "hess_quad_matrix"):
        # fused path: all A_ij in one sweep over the data
        try:
            A_full = oracle.hess_quad_matrix(basis).to(torch.float64)
        except RuntimeError as e:
            if not _is_missing_double_backward(e) or not hasattr(
                oracle, "hess_quad_matrix_fd"
            ):
                raise
            # The forward's custom-op backward is not itself differentiable in
            # the current process state (e.g. the fused Warp transmit op, whose
            # ``transmit_bwd`` carries only a first-order formula), so the exact
            # double-backward Hessian is unavailable. Fall back to the
            # finite-difference secant Hessian -- first-order grads only, still
            # real curvature -- rather than crash. ``g`` (current gradient) and
            # ``basis`` are already in scope.
            A_full = oracle.hess_quad_matrix_fd(basis, g, eps_hints=eps_hints).to(
                torch.float64
            )
    else:
        for i in range(G):
            He_i = oracle.hess_op(basis[i])
            A_terms.extend(rdot(He_i, basis[j]) for j in range(G))
    # Same reductions as one float(rdot(...)) each, but read back in ONE copy:
    # each float() is a device sync, and on a host-exposed row the G*G + 2G
    # round trips are the cost.
    b_terms = [rdot(g, basis[i]) for i in range(G)]
    act_terms = [rdot(basis[i], basis[i]) for i in range(G)]
    vals = _scalars_to_host(A_terms + b_terms + act_terms)
    if A_full is None:
        A_full = torch.tensor(vals[: G * G], dtype=torch.float64).reshape(G, G)
        vals = vals[G * G :]
    A_full = 0.5 * (A_full + A_full.T)  # H symmetric; remove numerical asymmetry
    b_full = torch.tensor(vals[:G], dtype=torch.float64)
    hss = float(A_full.sum())  # H(s, s), since s = sum_i e_i
    active = [i for i in range(G) if vals[G + i] > 0.0]
    if active:
        A = A_full[active][:, active]
        b = b_full[active]
        if torch.isfinite(A).all() and torch.isfinite(b).all():
            L, info = torch.linalg.cholesky_ex(A)
            if int(info) == 0:
                sol = torch.cholesky_solve(-b.unsqueeze(1), L).squeeze(1)
                if torch.isfinite(sol).all():
                    alphas = [0.0] * G
                    for ii, i in enumerate(active):
                        alphas[i] = float(sol[ii])
                    return alphas, hss
    # Not positive definite / degenerate: uniform scalar Newton rule
    # (same math as _newton_alpha, on the already-reduced scalars).
    tiny = 1e-30
    gs = float(b_full.sum())
    ss = float(rdot(s, s))
    if hss > tiny * ss and math.isfinite(hss) and math.isfinite(gs):
        alpha = -gs / hss
    else:
        alpha = fallback / math.sqrt(max(ss, 1e-300))
    return [alpha] * G, hss


def _step_groups(
    params: Sequence[Tensor], alphas: Sequence[float], s: Sequence[Tensor]
) -> None:
    with torch.no_grad():
        for p, a, si in zip(params, alphas, s, strict=False):
            p += a * si


# A group is (param_index, mode) with mode=None for the whole tensor or an
# int selecting a dim-0 slice (e.g. one probe mode of a (modes,K,M,M) tensor).

GroupSlice = tuple


def _default_groups(params: Sequence[Tensor]) -> list[GroupSlice]:
    return [(i, None) for i in range(len(params))]


def _basis_from_groups(
    s: Sequence[Tensor], groups: Sequence[GroupSlice]
) -> list[list[Tensor]]:
    """Group-restricted direction vectors e_g (full param-list layout)."""
    basis = []
    for pi, m in groups:
        e = [_structural_zeros_like(sj) for sj in s]
        if m is None:
            e[pi] = s[pi]
        else:
            # The selected block is only PARTIALLY zero, so it must NOT carry the
            # tag: a fresh untagged ``torch.zeros_like`` is written here on
            # purpose, and the mode slice is filled into it.
            e[pi] = torch.zeros_like(s[pi])
            e[pi][m] = s[pi][m]
        basis.append(e)
    return basis


def _step_groups_sliced(
    params: MutableSequence[Tensor],
    groups: Sequence[GroupSlice],
    alphas: Sequence[float],
    s: Sequence[Tensor],
) -> None:
    with torch.no_grad():
        for (pi, m), a in zip(groups, alphas, strict=False):
            if a == 0.0:
                continue
            if m is None:
                params[pi] += a * s[pi]
            else:
                params[pi][m] += a * s[pi][m]


def _relax_and_clip(alphas, relax, step_clip, groups, s):
    """Scale the per-group step sizes by ``relax``, then cap each one.
    A group's cap is on the largest parameter change it would make (``|alpha|``
    times the largest component of its search direction), so the clip is in
    parameter units, and a numerically zero direction is left alone."""
    if relax != 1.0:
        alphas = [relax * a for a in alphas]
    if step_clip is None:
        return alphas
    for gi, (clip, (pi, mo)) in enumerate(zip(step_clip, groups, strict=False)):
        if clip is None:
            continue
        si = s[pi] if mo is None else s[pi][mo]
        m = float(si.abs().max())
        if m > 0.0 and abs(alphas[gi]) * m > clip:
            alphas[gi] = math.copysign(clip / m, alphas[gi])
    return alphas
