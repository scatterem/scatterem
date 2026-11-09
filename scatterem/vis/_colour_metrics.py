"""CIELAB colour differences and colour-vision-deficiency simulation.

The colour policy (:mod:`scatterem.vis.colors`) and its tests measure colour
maps with these functions. The metric is the Euclidean CIELAB difference
(Delta E 1976), which is Eq. 1 of Crameri, Shephard & Heron (2020), "The misuse
of colour in science communication", Nat. Commun. 11, 5444.

Oklab, which :mod:`scatterem.vis.complex_color` uses to build its phase wheel,
was measured first and rejected for this job. Its cube root has no linear
segment near black, so it reports grayC and batlowK as non-uniform at the dark
end. CIELAB L*, the space those maps are designed in, is linear there.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "MACHADO_2009",
    "local_delta_e",
    "min_delta_e_to",
    "simulate_cvd",
    "srgb_to_lab",
]

# Linear sRGB -> CIE XYZ for the D65 white point (IEC 61966-2-1).
_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883])
_DELTA = 6.0 / 29.0

#: Machado, Oliveira & Fernandes (2009), severity 1.0, applied to linear sRGB.
#: Checked on 2026-10-14 against colorspacious.cvd.MACHADO_ET_AL_MATRICES[...][100].
MACHADO_2009: dict[str, NDArray] = {
    "protanopia": np.array(
        [
            [0.152286, 1.052583, -0.204868],
            [0.114503, 0.786281, 0.099216],
            [-0.003882, -0.048116, 1.051998],
        ]
    ),
    "deuteranopia": np.array(
        [
            [0.367322, 0.860646, -0.227968],
            [0.280085, 0.672501, 0.047413],
            [-0.011820, 0.042940, 0.968881],
        ]
    ),
    "tritanopia": np.array(
        [
            [1.255528, -0.076749, -0.178779],
            [-0.078411, 0.930809, 0.147602],
            [0.004733, 0.691367, 0.303900],
        ]
    ),
}


def _rgb(colours: ArrayLike) -> NDArray:
    """Colours as an ``(N, 3)`` float array, without an alpha column."""
    return np.atleast_2d(np.asarray(colours, dtype=float))[:, :3]


def _srgb_to_linear(colours: ArrayLike) -> NDArray:
    """Undo the sRGB transfer function."""
    c = _rgb(colours)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(linear: NDArray) -> NDArray:
    """Apply the sRGB transfer function to linear values in [0, 1]."""
    return np.where(
        linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1.0 / 2.4) - 0.055
    )


def srgb_to_lab(colours: ArrayLike) -> NDArray:
    """CIELAB coordinates of sRGB colours.

    Parameters
    ----------
    colours : array_like
        ``(N, 3)`` or ``(N, 4)`` sRGB values in ``[0, 1]``. An alpha column is
        ignored.

    Returns
    -------
    numpy.ndarray
        ``(N, 3)`` array of ``L*``, ``a*``, ``b*``.
    """
    t = (_srgb_to_linear(colours) @ _RGB_TO_XYZ.T) / _WHITE_D65
    f = np.where(t > _DELTA**3, np.cbrt(t), t / (3.0 * _DELTA**2) + 4.0 / 29.0)
    return np.stack(
        [
            116.0 * f[:, 1] - 16.0,
            500.0 * (f[:, 0] - f[:, 1]),
            200.0 * (f[:, 1] - f[:, 2]),
        ],
        axis=1,
    )


def local_delta_e(colours: ArrayLike, *, cyclic: bool = False) -> NDArray:
    """Delta E 1976 between each colour and the next.

    Parameters
    ----------
    colours : array_like
        ``(N, 3)`` or ``(N, 4)`` sRGB values in ``[0, 1]``, in map order.
    cyclic : bool, default=False
        Also measure the step from the last colour back to the first.

    Returns
    -------
    numpy.ndarray
        ``N - 1`` steps, or ``N`` when ``cyclic``.
    """
    lab = srgb_to_lab(colours)
    if cyclic:
        lab = np.vstack([lab, lab[:1]])
    return np.linalg.norm(np.diff(lab, axis=0), axis=1)


def min_delta_e_to(colours: ArrayLike, reference: ArrayLike) -> float:
    """Smallest Delta E 1976 between any of ``colours`` and ``reference``.

    Parameters
    ----------
    colours : array_like
        ``(N, 3)`` or ``(N, 4)`` sRGB values in ``[0, 1]``.
    reference : array_like
        One sRGB colour.

    Returns
    -------
    float
        The distance of the closest colour.
    """
    diff = srgb_to_lab(colours) - srgb_to_lab(reference)
    return float(np.linalg.norm(diff, axis=1).min())


def simulate_cvd(colours: ArrayLike, kind: str) -> NDArray:
    """Simulate how a person with a full dichromacy sees sRGB colours.

    Parameters
    ----------
    colours : array_like
        ``(N, 3)`` or ``(N, 4)`` sRGB values in ``[0, 1]``.
    kind : {"protanopia", "deuteranopia", "tritanopia"}
        Which cone type is missing (Machado et al. 2009, severity 1.0).

    Returns
    -------
    numpy.ndarray
        ``(N, 3)`` simulated sRGB values in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the three dichromacies.
    """
    try:
        matrix = MACHADO_2009[kind]
    except KeyError:
        raise ValueError(
            f"unknown CVD kind {kind!r}; expected one of {sorted(MACHADO_2009)}"
        ) from None
    linear = np.clip(_srgb_to_linear(colours) @ matrix.T, 0.0, 1.0)
    return _linear_to_srgb(linear)
