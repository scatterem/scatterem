"""The PRISM beam-basis geometry: aperture support and the derived ``N`` grid.

Private.  The PRISM beam-basis builder (``prism_smeta``) needs exactly these
two.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "PROBE_RIM_PX",
    "SUPPORT_FLOOR",
    "aperture_support",
    "derived_prism_grid",
    "probe_disc",
]


#: Rim width [pixel of axis 0] of the probe-forming aperture of the PRISM
#: probes and beam sets. It is 1 pixel, the edge of every probe that the
#: simulations form, so that a beam set covers the probe it expands. The
#: optics default ``RIM_WIDTH_PX`` (2 pixels) would give a different probe and
#: a beam set 1 pixel wider.
PROBE_RIM_PX = 1.0

#: The beam set holds the beams where the aperture is larger than this value.
#: With the 1-pixel rim this is ``d < R + 4.5`` pixels, the support of the
#: float32 aperture of v0.3; measured on 189 random windows, apertures and
#: pixel ratios (even and odd axes), the two sets are equal.
SUPPORT_FLOOR = 2.0**-26


def probe_disc(
    pixels, dk, wavelength: float, semiconvergence_angle: float, device, dtype=None
):
    """The probe-forming aperture of the PRISM probes on a ``pixels`` window.

    A disc of radius ``alpha / lambda`` in angle (so the pixel ratio
    ``dk_x / dk_y`` keeps it round), with the logistic rim of
    :data:`PROBE_RIM_PX`, corner origin, and exactly zero where the rim is not
    above :data:`SUPPORT_FLOOR`. :func:`aperture_support` reads the beam set off
    this function, so the two cannot disagree. The zero outside the set
    matters: a probe coefficient outside the beam set has no beam, and the
    logistic tail out there (below 1.5e-8) moved the partitioned BiP-PRISM map
    at ``n_radial=10`` from 1.3e-7 to 2.2e-4 of the exact map
    (``tests/test_prism_eels_simulator.py``). The v0.3 float32 aperture was
    zero there too.

    Parameters
    ----------
    pixels : (int, int)
        The probe / detector window ``(P_y, P_x)``.
    dk : (float, float)
        Reciprocal pixel ``(dk_y, dk_x)`` [1/A].
    wavelength : float
        Electron wavelength [A].
    semiconvergence_angle : float
        Aperture semi-angle [rad].
    device : torch.device or str
        Device of the result.
    dtype : torch.dtype, optional
        Real dtype; None is the torch default.

    Returns
    -------
    torch.Tensor
        ``(P_y, P_x)`` aperture, values in ``[0, 1]``.
    """
    from scatterem.physics.optics import smooth_disc  # noqa: PLC0415

    dk_y, dk_x = float(dk[0]), float(dk[1])
    disc = smooth_disc(
        (int(pixels[0]), int(pixels[1])),
        float(semiconvergence_angle) / (float(wavelength) * dk_y),
        rim_px=PROBE_RIM_PX,
        px_ratio=dk_x / dk_y,
        dtype=torch.float64,
        device=device,
    )
    disc = torch.where(disc > SUPPORT_FLOOR, disc, torch.zeros_like(disc))
    return disc.to(torch.get_default_dtype() if dtype is None else dtype)


def aperture_support(
    pixels, dx, wavelength: float, semiconvergence_angle: float, device
):
    """Where the probe-forming aperture is non-zero on the ``pixels`` window.

    THE beam set of a plane-wave expansion: the probe has a Fourier coefficient
    where its aperture is non-zero, so that -- not a disk of some chosen
    radius -- is what PRISM and partitioned PRISM must have a beam for.
    It is read off :func:`probe_disc`, the aperture the PRISM probes are built
    from, so the two cannot disagree. The rim of that aperture is a logistic
    of fixed width in PIXELS; the set ends where it falls below
    :data:`SUPPORT_FLOOR`, 4.5 px outside the radius. That is why the fixed
    FRACTIONAL margin this replaces (``1.05 * alpha / lambda``) truncated it on
    small apertures: 1.3 px at a 26 px radius, where the aperture is still
    5.5e-3 -- PRISM at f = 1, the same linear map as multislice, was 8.7e-4 off
    it (1.6e-2 on a 16 px window).

    Parameters
    ----------
    pixels : (int, int)
        The probe / detector window ``(P_y, P_x)``.
    dx : array-like (2,)
        Real-space pitch per axis [A].
    wavelength : float
        Electron wavelength [A].
    semiconvergence_angle : float
        Aperture semi-angle [rad].
    device : torch.device or str
        Device the aperture is rasterised on (the model's).

    Returns
    -------
    torch.Tensor
        ``(P_y, P_x)`` bool, corner origin (DC at ``[0, 0]``).
    """
    P = (int(pixels[0]), int(pixels[1]))
    dy_, dx_ = (float(v) for v in np.broadcast_to(np.asarray(dx, dtype=float), (2,)))
    dk = (1.0 / (P[0] * dy_), 1.0 / (P[1] * dx_))
    disc = probe_disc(
        P, dk, wavelength, semiconvergence_angle, device, dtype=torch.float64
    )
    return disc > 0


def derived_prism_grid(
    pixels: tuple[int, int],
    f: int | tuple[int, int],
) -> tuple[int, int]:
    """The PRISM simulation grid ``N = (f_y, f_x) * pixels``: derived, never chosen.

    The detector window is ``pixels`` and the beam lattice has spacing ``f``,
    so this is the only grid on which a beam lands on its own detector pixel.
    An explicit grid used to be accepted beside them with only ``2f | M``
    checked; every other value scattered the f-lattice beams over the wrong
    pixels (``M=(160, 160)`` at ``pixels=128``: rel. L2 2.39 against
    multislice), so there is no such argument.

    Parameters
    ----------
    pixels : (int, int)
        Detector window ``(P_y, P_x)``.
    f : int or (int, int)
        Interpolation factor, scalar or per axis.

    Returns
    -------
    tuple of int
        ``(N_y, N_x)``.
    """
    fy, fx = (int(f), int(f)) if isinstance(f, int) else (int(f[0]), int(f[1]))
    return (fy * int(pixels[0]), fx * int(pixels[1]))
