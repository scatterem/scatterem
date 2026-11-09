"""Default figures for direct-ptychography results (scatterem.vis.show)."""

from __future__ import annotations

from typing import Any

import numpy as np

from scatterem.data.aberrations import CARTESIAN_NAMES
from scatterem.physics.optics import lens_wavefront
from scatterem.vis._display_input import _to_numpy
from scatterem.vis.show import FigureContext, Panel
from scatterem.vis.show import _panels as P


def _ssnr_radial(ssnr: np.ndarray, sampling: Any) -> tuple[Any, Any, str]:
    """Radial SSNR mean against ``|q|``: in 1/unit with a calibration, else 1/px.

    The SSNR map has the pixel size of ``phase``, so ``sampling.pixel_size``
    gives its frequency grid directly.
    """
    pixel = getattr(sampling, "pixel_size", None)
    if pixel is not None and len(pixel) >= 2 and min(pixel[-2:]) > 0:
        units = getattr(sampling, "units", None) or ("Å",)
        unit = {"A": "Å"}.get(units[-1], units[-1])
        q, prof = P.radial_frequency_profile(ssnr, (pixel[-2], pixel[-1]))
        return q, prof, f"spatial frequency (1/{unit})"
    q, prof = P.radial_frequency_profile(ssnr)
    return q, prof, "spatial frequency (1/px)"


def direct_panels(result: Any, ctx: FigureContext) -> list[Panel]:
    """Phase, SSNR map and radial mean."""
    out = [
        Panel(
            "phase",
            lambda ax, c: P.image(
                ax, result.phase, ctx=c, role="phase", sampling=result.sampling
            ),
            "phase",
        )
    ]
    if result.ssnr is not None:
        # The SSNR map is corner-centred (``direct_ptychography_ssnr`` and the
        # empirical path both bin an unshifted FFT grid). Shift DC to the centre,
        # where the map display and ``radial_profile`` expect it.
        ssnr = np.fft.fftshift(_to_numpy(result.ssnr).astype(float), axes=(-2, -1))
        log_ssnr = np.log10(np.clip(ssnr, 1e-6, None))
        radius, prof, xlabel = _ssnr_radial(ssnr, result.sampling)
        out.append(
            Panel(
                "ssnr",
                lambda ax, c: P.image(ax, log_ssnr, ctx=c, role="diffraction"),
                "SSNR (log10)",
            )
        )
        out.append(
            Panel(
                "ssnr_radial",
                lambda ax, c: P.curve(
                    ax,
                    radius,
                    {"SSNR": prof},
                    ctx=c,
                    xlabel=xlabel,
                    ylabel="SSNR",
                    logy=True,
                ),
                "SSNR radial mean",
            )
        )
    return out








def aberration_panels(result: Any, ctx: FigureContext) -> list[Panel]:
    """Coefficient table; chi(k) when ``wavelength`` (Å) and ``k_max`` (1/Å) are given."""
    coeffs = _to_numpy(result.aberrations).astype(float)
    rows = [(n, f"{v:.3g} Å") for n, v in zip(CARTESIAN_NAMES, coeffs, strict=True)]
    if result.rotation is not None:
        rows.append(("rotation", f"{result.rotation:.3g}°"))
    out = [
        Panel(
            "coefficients",
            lambda ax, c: P.table(ax, rows, ctx=c),
            "aberration coefficients",
        )
    ]
    wavelength, k_max = ctx.opts.get("wavelength"), ctx.opts.get("k_max")
    if wavelength is not None and k_max is not None:
        import torch

        k = torch.linspace(-k_max, k_max, 129, dtype=torch.float64)
        qy, qx = torch.meshgrid(k, k, indexing="ij")
        chi = _to_numpy(
            lens_wavefront(torch.as_tensor(coeffs), qy, qx, lambda_a=float(wavelength))
        )
        out.append(
            Panel(
                "chi",
                lambda ax, c: P.image(ax, chi, ctx=c, role="signed", scalebar=False),
                "χ(k)",
            )
        )
    return out
