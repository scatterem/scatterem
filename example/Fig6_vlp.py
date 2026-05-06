"""
VLP 4D-STEM example (“Fig. 6”): large scan step, rotation, aberration priors, reconstructions.

Typical pipeline: bright-field crop, calibrate ``dk``, fit aberrations with seeded
Zernike-style coefficients, then direct ptychography, tcDF, and fused full field.

The first cell defines ``bin_4dstem_last2`` for binning the diffraction axes of a 4D
stack; it is **not called** in the cells below (kept for reuse or experiments).

Run cell-by-cell using ``# %%`` markers.

**Before you run**

- **GPU**: CUDA by default.
- **Data**: ``DATA_PATH`` — 4D ``.npy`` ``(ny, nx, qy, qx)``.
- **Metadata**: ``dr`` here is large (nm-scale step in Å); match ``rotation`` and
  aberration seeds to your dataset.
"""

import matplotlib

matplotlib.use("inline")  # Notebook-friendly backend

import matplotlib.pyplot as plt
import numpy as np
import torch

from scatterem2.vis.visualization import show_2d_array
from scatterem2.vis.visualization_utils import add_scalebar_to_ax

import scatterem2.vis as vis

from scatterem2.utils.data.aberrations import Aberrations
from scatterem2.utils.data.data_classes import Metadata4dstem
from scatterem2.utils.data.datasets import Dataset4dstem
from scatterem2.utils.stem import energy2wavelength

# %% 1) Helper: bin last two axes of 4D-STEM (optional; not used below)

try:
    import torch
    _has_torch = True
except Exception:
    _has_torch = False

def bin_4dstem_last2(x, by=(2, 2), mode="mean", crop="trim", pad_value=0):
    """
    Bin the last two axes (diffraction pattern) of a 4D-STEM dataset.

    Parameters
    ----------
    x : np.ndarray or torch.Tensor
        Shape (..., qy, qx). (Leading scan dims are untouched.)
    by : (int, int)
        Binning factors along (qy, qx).
    mode : {"mean", "sum", "max"}
        Reduction inside each bin.
    crop : {"trim", "pad", "error"}
        - "trim": drop extra rows/cols so size is divisible by by.
        - "pad": pad to next multiple using pad_value, then bin.
        - "error": raise if not divisible.
    pad_value : float
        Used when crop == "pad".

    Returns
    -------
    y : same type as x
        Shape (..., qy//by[0], qx//by[1]) (or ceil when crop="pad").
    """
    is_torch = _has_torch and isinstance(x, torch.Tensor)
    backend = torch if is_torch else np

    qy, qx = int(x.shape[-2]), int(x.shape[-1])
    byy, byx = int(by[0]), int(by[1])

    def _ceil_div(a, b): return (a + b - 1) // b

    if crop == "error":
        if (qy % byy) or (qx % byx):
            raise ValueError(f"Last two dims ({qy},{qx}) not divisible by by={by}.")
        qy_eff, qx_eff = qy, qx
    elif crop == "trim":
        qy_eff = (qy // byy) * byy
        qx_eff = (qx // byx) * byx
        if (qy_eff != qy) or (qx_eff != qx):
            x = x[..., :qy_eff, :qx_eff]
    elif crop == "pad":
        qy_eff = _ceil_div(qy, byy) * byy
        qx_eff = _ceil_div(qx, byx) * byx
        pad_qy = qy_eff - qy
        pad_qx = qx_eff - qx
        if pad_qy or pad_qx:
            if is_torch:
                # pad format is (last_dim_left, last_dim_right, second_last_left, second_last_right, ...)
                pad = (0, pad_qx, 0, pad_qy) + (0, 0) * (x.ndim - 2)
                x = torch.nn.functional.pad(x, pad, mode="constant", value=pad_value)
            else:
                pad_width = [(0, 0)] * x.ndim
                pad_width[-2] = (0, pad_qy)
                pad_width[-1] = (0, pad_qx)
                x = np.pad(x, pad_width, mode="constant", constant_values=pad_value)
    else:
        raise ValueError("crop must be one of {'trim','pad','error'}")

    # reshape into bins: (..., qy_eff/byy, byy, qx_eff/byx, byx)
    new_shape = (*x.shape[:-2], qy_eff // byy, byy, qx_eff // byx, byx)
    x = x.reshape(new_shape)

    # reduce within each bin
    if mode == "mean":
        if is_torch:
            y = x.mean(dim=(-1, -3))
        else:
            y = x.mean(axis=(-1, -3))
    elif mode == "sum":
        if is_torch:
            y = x.sum(dim=(-1, -3))
        else:
            y = x.sum(axis=(-1, -3))
    elif mode == "max":
        if is_torch:
            y = x.amax(dim=(-1, -3))
        else:
            y = x.max(axis=(-1, -3))
    else:
        raise ValueError("mode must be one of {'mean','sum','max'}")

    return y

# %% 2) Device, beam parameters, load 4D stack

device = torch.device("cuda")
E = 200e3  # eV
semiangle_cutoff = 30.6e-3  # rad
dr = 11.0  # Å (large step — check against acquisition)

DATA_PATH = "/home/shengbo/data/vlp/fig6_vlp.npy"
data = np.load(DATA_PATH)

# %% 3) Wavelength + metadata + dataset

wavelength = energy2wavelength(E)
meta = Metadata4dstem(
    energy=E,
    semiconvergence_angle=float(semiangle_cutoff),
    rotation=-83.8,
    defocus_guess=float(0),
    sample_thickness_guess=0,
    vacuum_probe=None,
    sampling=(dr, dr, 1, 1),
    units=["A", "A", "A^-1", "A^-1"],
    shape=np.array(data.shape, dtype=np.int32),
    aberrations=Aberrations(array=torch.zeros((12,), device=device)),
)
ny, nx, M, M = data.shape

dataset = Dataset4dstem.from_array(
    array=data,
    origin=np.array((ny / 2, nx / 2, M / 2, M / 2), dtype=np.float32),
    name="dataset",
    signal_units="arb. units",
    meta=meta,
    transform_to_amplitudes=False,
    device=device,
    normalize=True,
)
print(dataset)

# %% 4) Bright-field crop

data_bf = dataset.crop_brightfield_(
    thresh_lower=0.1,
    thresh_upper=0.99,
)
vis.show_2d(data_bf.array[:25, :25].sum((0, 1)), cbar=True, title="data")

print(f"fluence = {dataset.fluence} e-/A^2")

# %% 5) k-space calibration

rBF, c = data_bf.bright_field_radius_and_center_(
    thresh_lower=0.1, thresh_upper=0.99, plot_rbf=True, method="area"
)
dalpha = semiangle_cutoff / rBF
dk = dalpha / data_bf.meta.wavelength
data_bf.meta.sampling = (dr, dr, dk, dk)
data_bf.sampling = (dr, dr, dk, dk)
print(f"dk = {dk}")
print(f"dalpha = {dalpha}")
print(f"wavelength = {data_bf.meta.wavelength}")
print(data_bf)
print(data_bf.meta)
dataset.meta = data_bf.meta
dataset.sampling = data_bf.sampling
k_max = rBF * dk
nyquist = 1 / (2 * k_max)
print(f"nyquist = {nyquist}")
print(f"k_max = {k_max}")

# %% 6) Aberration fit (with non-zero initial aberration vector)

bright_field_mask_threshold = 0.3
correction_method = "total-variation"  # "bright-field-shifts"
fit_rotation = False
target_percentage_nonzero_pixels = 0.75
n_batches = 25
registration_upsample_factor = 10
lowpass_fwhm_bright_field = None
bin = 1
arrow_scale = 25e-2
verbosity = 1
correct_order = 2
gradient_mask = torch.ones(12, dtype=torch.bool)
num_iterations = 50
lr = 1

roi_shape = (450, 450)
roi_center = "center"
upsample = 1.0
n_center_indices = 25
dataset.meta.aberrations.array[0] = -150.0
dataset.meta.aberrations.array[1] = 16.0
dataset.meta.aberrations.array[2] = -140.0
dataset.determine_aberrations_(
    bright_field_mask_threshold=bright_field_mask_threshold,
    correction_method=correction_method,
    fit_rotation=fit_rotation,
    target_percentage_nonzero_pixels=target_percentage_nonzero_pixels,
    n_batches=n_batches,
    registration_upsample_factor=registration_upsample_factor,
    lowpass_fwhm_bright_field=lowpass_fwhm_bright_field,
    bin=bin,
    arrow_scale=arrow_scale,
    verbosity=verbosity,
    correct_order=correct_order,
    gradient_mask=gradient_mask,
    num_iterations=num_iterations,
    lr=lr,
    roi_shape=roi_shape,
    roi_center=roi_center,
    upsample=upsample,
    n_center_indices=n_center_indices,
)

# %% 7) Direct ptychography + display

upsample = 1.0
direct_ptycho_image, ssnr_ptycho = dataset.direct_ptychography(
    upsample=upsample, n_batches=8, return_snr=True, bright_field_mask_threshold=0.3)

direct_ptycho_image3 = direct_ptycho_image.clone()
p02, p98 = torch.quantile(direct_ptycho_image3.cpu(), torch.tensor([0.02, 0.98]))
print(f"2nd percentile: {p02:.3f}")
print(f"98th percentile: {p98:.3f}")
direct_ptycho_image3 -= p02  # display stretch

fig, ax = plt.subplots(figsize=(6, 6))
fig_direct_ptycho, ax_direct_ptycho = vis.show_2d(
    [direct_ptycho_image3],
    cbar=True,
    title=["Phase, BF reconstruction"],
    figax=(fig, ax),
)

# %% 8) Tilt-corrected dark field (tcDF)

tcDF, ssnr_tcdf = dataset.tilt_corrected_dark_field(
    n_dark_field_segments=16,
    verbosity=1,
    bright_field_mask_threshold=bright_field_mask_threshold,
    upsample=upsample,
    return_snr=True,
    snr_blur_sigma=0.0,
)
fig, ax = plt.subplots(figsize=(6, 6))
fig_tcDF, ax_tcDF = vis.show_2d(
    [
        tcDF,
    ],
    cbar=True,
    title=["tcDF"],
    figax=(fig, ax),
)

# %% 9) Fused full field

fff, phase_weighted, tcdf_weighted = dataset.fused_full_field(verbosity=2)
