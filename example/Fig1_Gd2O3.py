"""
Gd₂O₃ 4D-STEM example (60 kV): BF crop, aberration fit, ptychography, tcDF, fused field.

Run cell-by-cell where ``# %%`` is supported (VS Code / Cursor, Spyder, PyCharm), or
reuse chunks in a notebook.

**Before you run**

- **GPU**: ``device = cuda`` by default; use ``cpu`` only if you accept slower runs and
  fix any device mismatches.
- **Data**: Set ``DATA_PATH`` to your 4D stack ``(ny, nx, qy, qx)`` as a NumPy ``.npy``.
- **Metadata**: Adjust ``E``, ``semiangle_cutoff``, ``dr``, and defocus guess in the
  aberration block for your acquisition.

**Pipeline (numbered ``# %%`` sections)**

1. Imports, device, beam parameters, load data.
2. ``Metadata4dstem`` + ``Dataset4dstem``; print total intensity.
3. Bright-field crop; quick summed-DP preview.
4. BF radius → calibrate ``dk`` / ``dalpha``; copy sampling back to ``dataset``.
5. ``determine_aberrations_`` (hyperparameters grouped in one cell).
6. Direct ptychography (upsampled); percentile stretch for display.
7. Tilt-corrected dark field + SNR map.
8. ``fused_full_field``.
"""

import matplotlib
import numpy as np
import torch

matplotlib.use("inline")  # Notebook-friendly backend
from scatterem2.vis.visualization_utils import add_scalebar_to_ax
import matplotlib.pyplot as plt

import scatterem2.vis as vis
from scatterem2.vis.visualization import show_2d_array

from scatterem2.utils.data.aberrations import Aberrations
from scatterem2.utils.data.data_classes import Metadata4dstem
from scatterem2.utils.data.datasets import Dataset4dstem

# %% 1) Device, beam parameters, load 4D .npy

device = torch.device("cuda")
E = 60e3  # eV
semiangle_cutoff = 30e-3  # rad
dr = 0.43  # Å, scan step

DATA_PATH = "/home/shengbo/data/TS_1-selected/fig1_gd2o3.npy"
data = np.load(DATA_PATH)

# %% 2) Metadata + dataset

meta = Metadata4dstem(
    energy=E,
    semiconvergence_angle=float(semiangle_cutoff),
    rotation=0,
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
    name="dataset_full",
    signal_units="arb. units",
    meta=meta,
    transform_to_amplitudes=False,
    device=device,
    normalize=True,
)

total_intensity = dataset.array.sum()
print(f"Total intensity: {total_intensity}")
print(dataset)

# %% 3) Bright-field crop

data_bf = dataset.crop_brightfield_(
    thresh_lower=0.1,
    thresh_upper=0.99,
)
vis.show_2d(data_bf.array[:25, :25].sum((0, 1)), cbar=True, title="data")

# %% 4) k-space calibration from bright-field disk

rBF, c = data_bf.bright_field_radius_and_center_(
    thresh_lower=0.1, thresh_upper=0.99, plot_rbf=True, method="area"
)
# rBF += 1
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

# %% 5) Aberration fit

bright_field_mask_threshold = 0.1
correction_method = "total-variation"
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
dataset.meta.aberrations.array[0] = -50
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

# %% 6) Direct ptychography + display

upsample = 2.0

direct_ptycho_image, ssnr_ptycho = dataset.direct_ptychography(
    upsample=upsample, n_batches=15, return_snr=True, verbosity=2)


fig, ax = plt.subplots(figsize=(6, 6))
fig_direct_ptycho, ax_direct_ptycho = vis.show_2d(
    [direct_ptycho_image],
    cbar=True,
    title=["Phase, BF reconstruction"],
    figax=(fig, ax),
)

# %% 7) Tilt-corrected dark field (tcDF) + SNR

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
vis.show_2d(torch.fft.fftshift(ssnr_tcdf), cbar=True, title="ssnr_tcdf")

# %% 8) Fused full field

fused, phase_weighted, tcdf_weighted = dataset.fused_full_field(verbosity=2)