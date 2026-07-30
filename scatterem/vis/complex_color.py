"""Colouring complex-valued images: amplitude as lightness, phase as hue.

Replaces the quantem-derived ``array_to_rgba`` / ``list_of_arrays_to_rgba`` /
``add_arg_cbar_to_ax`` trio, and does the job in a different colour space for a
reason.

A domain-coloured image is only honest if equal steps in phase look like equal
steps of hue and do not also change apparent brightness. That is what a
perceptually-uniform space buys. The replaced code used CIECAM02-ish ``JCh`` via
``colorspacious``, an extra dependency for one function. This uses **Oklab**
(Ottosson, 2020), whose forward transform is a pair of 3x3 matrices and a cube
root -- so it needs numpy and nothing else, and it is both newer and better
behaved than JCh for exactly this use: its lightness is more uniform and its hue
lines stay straighter at high chroma, which is where a phase wheel lives.

Dropping ``colorspacious`` also removed the last dependency the release did not
otherwise need.

The convention here: **lightness carries amplitude, hue carries phase**, and
chroma is constant except where it must be reduced to stay inside sRGB. That
ordering matters -- putting amplitude in chroma instead makes low-amplitude
regions grey *and* washes out their phase, which hides exactly the weak-signal
detail a phase image is usually being read for.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Oklab -> LMS' (the inverse of the matrix Ottosson gives for sRGB -> Oklab).
_OKLAB_TO_LMS = np.array(
    [
        [1.0, +0.3963377774, +0.2158037573],
        [1.0, -0.1055613458, -0.0638541728],
        [1.0, -0.0894841775, -1.2914855480],
    ]
)

# LMS -> linear sRGB.
_LMS_TO_LINEAR_RGB = np.array(
    [
        [+4.0767416621, -3.3077115913, +0.2309699292],
        [-1.2684380046, +2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147, +1.7076147010],
    ]
)


def _linear_to_srgb(linear: NDArray) -> NDArray:
    """The sRGB transfer function, applied elementwise."""
    linear = np.clip(linear, 0.0, 1.0)
    low = linear <= 0.0031308
    out = np.empty_like(linear)
    out[low] = 12.92 * linear[low]
    out[~low] = 1.055 * np.power(linear[~low], 1.0 / 2.4) - 0.055
    return out


def oklch_to_srgb(lightness: NDArray, chroma: NDArray, hue: NDArray) -> NDArray:
    """Convert Oklch to sRGB in ``[0, 1]``, shape ``(..., 3)``.

    Args:
        lightness: ``L`` in ``[0, 1]``.
        chroma: ``C``, typically ``<= 0.4``; larger values leave the sRGB gamut.
        hue: ``h`` in radians.
    """
    a = chroma * np.cos(hue)
    b = chroma * np.sin(hue)
    lab = np.stack(np.broadcast_arrays(lightness, a, b), axis=-1)
    lms = lab @ _OKLAB_TO_LMS.T
    linear = (lms**3) @ _LMS_TO_LINEAR_RGB.T
    return _linear_to_srgb(linear)


def _max_in_gamut_chroma(
    lightness: NDArray, hue: NDArray, ceiling: float, steps: int = 16
) -> NDArray:
    """Largest chroma per pixel that still lands inside sRGB.

    Clipping out-of-gamut colours instead would distort hue -- clipping one
    channel moves the colour sideways, not just inwards -- so chroma is reduced
    until the colour fits. A bisection converges to well under a perceptual step
    in 16 iterations, and unlike a closed-form gamut boundary it needs no
    per-hue tables.
    """
    low = np.zeros_like(lightness)
    high = np.full_like(lightness, ceiling)
    for _ in range(steps):
        mid = 0.5 * (low + high)
        rgb = oklch_to_srgb(lightness, mid, hue)
        # A colour is in gamut when the *linear* conversion needed no clipping;
        # test the sRGB result's extremes, which is equivalent and cheaper.
        inside = (rgb > 0.0).all(axis=-1) & (rgb < 1.0).all(axis=-1)
        low = np.where(inside, mid, low)
        high = np.where(inside, high, mid)
    return low


def complex_to_rgba(
    amplitude: NDArray,
    phase: NDArray | None = None,
    *,
    cmap: str = "gray",
    chroma: float = 0.13,
) -> NDArray:
    """Colour an image, using ``phase`` as hue when it is given.

    Args:
        amplitude: values in ``[0, 1]`` (normalise before calling).
        phase: angles in radians. When None, ``cmap`` is applied and the result is
            an ordinary colormapped image.
        cmap: matplotlib colormap name, used only when ``phase`` is None.
        chroma: Oklab chroma for the phase wheel. The default is a compromise --
            high enough for hue to read, low enough to stay in gamut across most
            of the lightness range.

    Returns:
        ``(..., 4)`` RGBA in ``[0, 1]``, alpha fully opaque.
    """
    amplitude = np.asarray(amplitude, dtype=np.float64)
    if phase is None:
        import matplotlib as mpl

        return np.asarray(mpl.colormaps.get_cmap(cmap)(np.clip(amplitude, 0.0, 1.0)))

    phase = np.asarray(phase, dtype=np.float64)
    if phase.shape != amplitude.shape:
        raise ValueError(
            f"amplitude {amplitude.shape} and phase {phase.shape} must match"
        )

    lightness = np.clip(amplitude, 0.0, 1.0)
    usable = _max_in_gamut_chroma(lightness, phase, chroma)
    rgb = oklch_to_srgb(lightness, usable, phase)
    return np.concatenate([rgb, np.ones(rgb.shape[:-1] + (1,))], axis=-1)


def phase_wheel(n: int = 256, *, lightness: float = 0.72, chroma: float = 0.13):
    """``(colors, angles)`` for a phase colorbar: one full turn of hue.

    ``lightness`` is fixed so the bar reads as hue alone. The default sits near
    the middle of the range where the sRGB gamut is widest, so the wheel keeps its
    chroma all the way round instead of desaturating at some hues.
    """
    angles = np.linspace(-np.pi, np.pi, n)
    L = np.full(n, lightness)
    usable = _max_in_gamut_chroma(L, angles, chroma)
    return oklch_to_srgb(L, usable, angles), angles


def add_phase_colorbar(fig, cax, *, chroma: float = 0.13):
    """Draw a phase colorbar into ``cax``, labelled in multiples of pi."""
    from matplotlib import colorbar, colors

    rgb, _ = phase_wheel(chroma=chroma)
    cmap = colors.ListedColormap(rgb)
    norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
    bar = colorbar.Colorbar(cax, cmap=cmap, norm=norm)
    bar.set_ticks([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi])
    bar.set_ticklabels([r"$-\pi$", r"$-\pi/2$", "$0$", r"$\pi/2$", r"$\pi$"])
    bar.set_label("phase")
    return bar


def overlay_to_rgba(arrays, *, norm=None, chroma: float = 0.13) -> NDArray:
    """Blend several same-shaped images into one, giving each its own hue.

    Replaces ``list_of_arrays_to_rgba``. Each array gets an evenly spaced hue
    around the Oklab wheel and contributes in proportion to its own normalised
    amplitude, so a pixel where the arrays agree comes out desaturated and a pixel
    dominated by one of them takes that array's hue. This is what
    ``combine_images=True`` means -- a mosaic of panels side by side is
    :func:`tile_to_rgba`, which is a different question.

    Args:
        arrays: the images to blend. All must share a shape.
        norm: callable mapping amplitude to ``[0, 1]``, applied per array. If
            omitted, each is scaled by its own finite min and max.
        chroma: Oklab chroma of the hues, before gamut clamping.

    Returns:
        ``(..., 4)`` RGBA in ``[0, 1]``.
    """
    amplitudes = [_normalised_amplitude(a, norm) for a in arrays]
    if not amplitudes:
        raise ValueError("overlay_to_rgba needs at least one array")
    shapes = {a.shape for a in amplitudes}
    if len(shapes) != 1:
        raise ValueError(f"all arrays must share a shape; got {sorted(shapes)}")

    hues = np.linspace(0.0, 2.0 * np.pi, len(amplitudes), endpoint=False)
    total = np.sum(amplitudes, axis=0)
    safe = np.where(total > 0, total, 1.0)

    # Weighted circular mean of the hues gives the blend its colour, and the
    # length of that mean vector gives its purity: equal contributions cancel to
    # a short vector and so to a desaturated pixel, which is the point.
    x = sum(a * np.cos(h) for a, h in zip(amplitudes, hues)) / safe
    y = sum(a * np.sin(h) for a, h in zip(amplitudes, hues)) / safe
    hue = np.arctan2(y, x)
    purity = np.clip(np.hypot(x, y), 0.0, 1.0)

    lightness = np.clip(np.max(amplitudes, axis=0), 0.0, 1.0)
    usable = _max_in_gamut_chroma(lightness, hue, chroma * purity)
    rgb = oklch_to_srgb(lightness, usable, hue)
    return np.concatenate([rgb, np.ones(rgb.shape[:-1] + (1,))], axis=-1)


def _normalised_amplitude(panel, norm) -> NDArray:
    """Amplitude of ``panel`` mapped into ``[0, 1]``.

    :func:`complex_to_rgba` requires ``[0, 1]`` and *clips*, so an un-normalised
    0..1000 image renders as 99.9% pure white. Every caller goes through here.
    """
    amplitude = (
        np.abs(panel) if np.iscomplexobj(panel) else np.asarray(panel, dtype=float)
    )
    if norm is not None:
        return np.asarray(norm(amplitude), dtype=float)
    finite = amplitude[np.isfinite(amplitude)]
    if finite.size == 0:
        return np.zeros_like(amplitude)
    low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros_like(amplitude)
    return (amplitude - low) / (high - low)


def tile_to_rgba(
    rows, *, norm=None, cmap: str = "gray", chroma: float = 0.13
) -> NDArray:
    """Composite a grid of same-shaped images into one RGBA mosaic.

    Replaces ``list_of_arrays_to_rgba``. Each entry is coloured independently and
    the results are tiled, so a bright panel cannot darken its neighbours -- the
    alternative, one normalisation across the whole grid, makes every panel but
    the brightest unreadable.

    Args:
        rows: sequence of rows, each a sequence of same-shaped 2D arrays.
        norm: callable mapping amplitude to ``[0, 1]``, applied to each panel. If
            omitted, each panel is scaled by its own finite min and max.

            This argument is not optional in spirit: :func:`complex_to_rgba`
            requires ``[0, 1]`` and *clips*, so handing it raw data silently
            renders a 0..1000 image as 99.9% pure white. An earlier version of
            this function did exactly that.
        cmap: colormap for real-valued panels.
        chroma: Oklab chroma for the phase hue of complex panels.
    """

    coloured = [
        [
            complex_to_rgba(
                _normalised_amplitude(a, norm),
                np.angle(a) if np.iscomplexobj(a) else None,
                cmap=cmap,
                chroma=chroma,
            )
            for a in row
        ]
        for row in rows
    ]
    shapes = {panel.shape for row in coloured for panel in row}
    if len(shapes) != 1:
        raise ValueError(f"all panels must share a shape; got {sorted(shapes)}")
    return np.concatenate([np.concatenate(row, axis=1) for row in coloured], axis=0)
