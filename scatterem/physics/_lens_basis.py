"""Polynomial basis of the lens wavefront in Cartesian angles.

This private module was written for scatterem from Krivanek, Dellby and
Lupini (1999), Ultramicroscopy 78, 1-11; Haider, Uhlemann and Zach (2000),
Ultramicroscopy 81, 163-175; and Kirkland (2020), Advanced Computing in
Electron Microscopy, 3rd ed., ch. 3.

With the angles ``u = lambda k_x``, ``v = lambda k_y`` [rad],
``r2 = u^2 + v^2`` and ``z = u + i v``, every term of the wavefront is a
polynomial:

    theta^(n+1) cos(m phi) = r2^((n+1-m)/2) Re(z^m)
    theta^(n+1) sin(m phi) = r2^((n+1-m)/2) Im(z^m)

The module uses no square root and no ``atan2``. Thus the value at ``k = 0``
is exactly zero and all derivatives are finite there.
"""

from __future__ import annotations

import torch

# (order n, multiplicity m, part) of each slot of a lens vector. The part
# "c" multiplies cos(m phi) and the part "s" multiplies sin(m phi).
TERMS: tuple[tuple[int, int, str], ...] = (
    (1, 0, "c"),
    (1, 2, "c"),
    (1, 2, "s"),
    (2, 1, "c"),
    (2, 1, "s"),
    (2, 3, "c"),
    (2, 3, "s"),
    (3, 0, "c"),
    (3, 2, "c"),
    (3, 2, "s"),
    (3, 4, "c"),
    (3, 4, "s"),
)


def _z_powers(
    u: torch.Tensor, v: torch.Tensor
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return the real and imaginary parts of ``z^m`` for ``m = 0..4``.

    Parameters
    ----------
    u, v : torch.Tensor
        Angles ``lambda k_x`` and ``lambda k_y`` [rad], same shape.

    Returns
    -------
    tuple of two lists
        ``re[m]`` and ``im[m]`` [rad^m] for ``m`` from 0 to 4.
    """
    one = torch.ones_like(u)
    zero = torch.zeros_like(u)
    zr2 = u * u - v * v
    zi2 = 2.0 * (u * v)
    zr3 = zr2 * u - zi2 * v
    zi3 = zr2 * v + zi2 * u
    zr4 = zr2 * zr2 - zi2 * zi2
    zi4 = 2.0 * (zr2 * zi2)
    return [one, u, zr2, zr3, zr4], [zero, v, zi2, zi3, zi4]


def _r2_powers(u: torch.Tensor, v: torch.Tensor) -> list[torch.Tensor]:
    """Return ``r2^k`` for ``k = 0..2`` with ``r2 = u^2 + v^2`` [rad^(2k)]."""
    r2 = u * u + v * v
    return [torch.ones_like(u), r2, r2 * r2]


def basis_stack(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Return the 12 rows ``P_j(u, v) / (n_j + 1)``, shape ``(12, *u.shape)``.

    Parameters
    ----------
    u, v : torch.Tensor
        Angles ``lambda k_x`` and ``lambda k_y`` [rad], same shape.

    Returns
    -------
    torch.Tensor
        The basis terms [rad^(n+1)], one row per slot.
    """
    re, im = _z_powers(u, v)
    rp = _r2_powers(u, v)
    rows = []
    for n, m, part in TERMS:
        q = re[m] if part == "c" else im[m]
        rows.append(rp[(n + 1 - m) // 2] * q / float(n + 1))
    return torch.stack(rows, dim=0)


def basis_slope_stacks(
    u: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the derivatives of the basis rows by ``u`` and by ``v``.

    The derivative of ``r2^k Q`` by ``u`` is ``2 k u r2^(k-1) Q + r2^k dQ/du``.
    For ``Q = z^m``: ``dz^m/du = m z^(m-1)`` and ``dz^m/dv = i m z^(m-1)``.

    Parameters
    ----------
    u, v : torch.Tensor
        Angles ``lambda k_x`` and ``lambda k_y`` [rad], same shape.

    Returns
    -------
    tuple of torch.Tensor
        ``(dP/du, dP/dv)``, each of shape ``(12, *u.shape)`` [rad^n], each row
        divided by ``n_j + 1``.
    """
    re, im = _z_powers(u, v)
    rp = _r2_powers(u, v)
    zero = torch.zeros_like(u)
    rows_u, rows_v = [], []
    for n, m, part in TERMS:
        k = (n + 1 - m) // 2
        if part == "c":
            q = re[m]
            dq_u = m * re[m - 1] if m > 0 else zero
            dq_v = -m * im[m - 1] if m > 0 else zero
        else:
            q = im[m]
            dq_u = m * im[m - 1]
            dq_v = m * re[m - 1]
        du = rp[k] * dq_u
        dv = rp[k] * dq_v
        if k > 0:
            du = du + (2.0 * k) * u * rp[k - 1] * q
            dv = dv + (2.0 * k) * v * rp[k - 1] * q
        rows_u.append(du / float(n + 1))
        rows_v.append(dv / float(n + 1))
    return torch.stack(rows_u, dim=0), torch.stack(rows_v, dim=0)
