"""Argument checks of the structure container (private).

Helpers of :mod:`scatterem.physics.atom_set`. Every error is a
``ValueError`` whose message starts with the argument name.
"""

from __future__ import annotations

import numpy as np
import torch

from scatterem.crystal.phase import DEFAULT_U2
from scatterem.rotation.axis_angle import to_matrix

#: Largest asymmetry of a tensor, relative to its largest entry. Tensors
#: from files carry about 1e-16 relative rounding; 1e-10 refuses only real
#: asymmetry.
_SYMMETRY = 1e-10

#: Largest atomic number of the element table.
_MAX_Z = 118


def real(x, name: str) -> np.ndarray:
    raw = np.asarray(x)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name}: must be real numbers; got dtype {raw.dtype}")
    return raw


def per_atom(x, name: str, n: int, default: float) -> np.ndarray:
    if x is None:
        return np.full(n, default, dtype=np.float64)
    raw = real(x, name)
    if raw.ndim == 0:
        out = np.full(n, float(raw), dtype=np.float64)
    elif raw.shape == (n,):
        out = np.array(raw, dtype=np.float64)
    else:
        raise ValueError(
            f"{name}: must be a scalar or have shape ({n},); got {raw.shape}"
        )
    if not np.isfinite(out).all():
        raise ValueError(f"{name}: values must be finite")
    return out


def check_fractional(fractional) -> np.ndarray:
    f = np.array(real(fractional, "fractional"), dtype=np.float64)
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"fractional: must have shape (N, 3); got {f.shape}")
    if not np.isfinite(f).all():
        raise ValueError("fractional: values must be finite")
    return f


def rot_from_axis_angle(turn_axis, turn_angle) -> np.ndarray:
    """(3, 3) float64 matrix of the active right-handed turn."""
    R = to_matrix(
        torch.as_tensor(turn_axis, dtype=torch.float64),
        torch.as_tensor(turn_angle, dtype=torch.float64),
    )
    return R.numpy().astype(np.float64)


def check_atom_data(n, element_z, site_fraction, msd, adp, full_tolerance):
    """Check and convert the per-atom data of N atoms.

    Returns
    -------
    tuple
        ``(z, site_fraction, msd, adp)``: (N,) int64, (N,) float64, (N,) float64
        Å², and (N, 3, 3) float64 Å² or None. New arrays. Site fractions in
        ``(1, 1 + full_tolerance]`` become 1.
    """
    z = real(element_z, "element_z")
    if z.shape != (n,):
        raise ValueError(f"element_z: must have shape ({n},); got {z.shape}")
    zf = z.astype(np.float64)
    if not (np.isfinite(zf) & (zf == np.rint(zf))).all():
        raise ValueError("element_z: values must be integers")
    if n and (zf.min() < 1 or zf.max() > _MAX_Z):
        raise ValueError(f"element_z: values must be in 1..{_MAX_Z}")
    z = zf.astype(np.int64)
    fill = per_atom(site_fraction, "site_fraction", n, 1.0)
    if n and (fill.min() < 0.0 or fill.max() > 1.0 + full_tolerance):
        raise ValueError(
            f"site_fraction: values must be in [0, 1]; got min {fill.min()}, max {fill.max()}"
        )
    fill[fill > 1.0] = 1.0
    u2 = per_atom(msd, "msd", n, DEFAULT_U2)
    if n and u2.min() < 0.0:
        raise ValueError(f"msd: values must be >= 0; got min {u2.min()}")
    if adp is not None:
        adp = np.array(real(adp, "adp"), dtype=np.float64)
        if adp.shape != (n, 3, 3):
            raise ValueError(f"adp: must have shape ({n}, 3, 3); got {adp.shape}")
        if not np.isfinite(adp).all():
            raise ValueError("adp: values must be finite")
        if n:
            asym = np.abs(adp - adp.transpose(0, 2, 1)).max()
            if asym > _SYMMETRY * max(np.abs(adp).max(), 1e-30):
                raise ValueError(f"adp: tensors must be symmetric; asymmetry {asym:g}")
    return z, fill, u2, adp
