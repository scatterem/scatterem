"""
Biological 4D-STEM example: bright-field crop, aberration fitting, and reconstructions.

This script mirrors the workflow used for a bio-related figure (ptychography-style
direct reconstruction, tilt-corrected dark field, and fused full field). Run it
cell-by-cell in an editor that supports ``# %%`` markers (VS Code / Cursor, Spyder,
PyCharm Scientific Mode), or paste sections into a notebook.

**Before you run**

- **GPU**: Uses ``torch.device("cuda")``. For CPU-only, set ``device`` to
  ``torch.device("cpu")`` (slower; some paths may expect CUDA tensors—check errors).
- **Data**: Point ``DATA_PATH`` to your 4D-STEM array ``(ny, nx, qy, qx)`` in NumPy
  ``.npy`` format. The path below is an example from the original author’s machine.
- **Microscope metadata**: ``E``, ``semiangle_cutoff``, ``dr``, ``rotation``, and
  pre-filled aberration guesses should match your experiment.

**Pipeline (each numbered ``# %%`` section)**

1. Device, beam parameters, and load the 4D ``.npy`` stack.
2. Build ``Metadata4dstem`` and ``Dataset4dstem`` from the array.
3. Bright-field crop; calibrate reciprocal-space sampling (``dk``, ``dalpha``).
4. Fit aberrations (many hyperparameters in one block—tune for your data).
5. Direct ptychography reconstruction and a simple intensity display.
6. Tilt-corrected dark field (tcDF).
7. Fused full field (combines modalities; ``verbosity`` prints progress).

Outputs are mostly Matplotlib figures; intermediate prints show ``dk``, Nyquist, etc.
"""

import matplotlib

matplotlib.use("inline")  # Notebook-friendly backend; safe for script + inline plots

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

# %% 1) Setup: device, beam energy, convergence, real-space pixel size, load 4D data

device = torch.device("cuda")
E = 300e3  # eV
semiangle_cutoff = 7.0e-3  # rad, semiconvergence angle (defines bright-field disk scale)
dr = 28.7  # Å, scan step (both fast and slow axes here)

# Replace with your file: shape (ny, nx, M, M) — scan y, scan x, diffraction y, x
DATA_PATH = "/home/shengbo/data/muller_bio/42_transpose.npy"
data = np.load(DATA_PATH)

# %% 2) Metadata + dataset: links physical units to the array

wavelength = energy2wavelength(E)

meta = Metadata4dstem(
    energy=E,
    semiconvergence_angle=float(semiangle_cutoff),
    rotation=-84.46,
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

# %% 3) Bright-field crop and k-space calibration from the BF disk radius

data_bf = dataset.crop_brightfield_(
    thresh_lower=0.1,
    thresh_upper=0.99,
)
vis.show_2d(data_bf.array[:25, :25].sum((0, 1)), cbar=True, title="data")

rBF, c = data_bf.bright_field_radius_and_center_(
    thresh_lower=0.5, thresh_upper=0.8, plot_rbf=True, method="area"
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

# %% 4) Aberration fit: mask threshold, optimizer knobs, ROI, initial Zernike-style guesses

bright_field_mask_threshold = 0.8
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
num_iterations = 5000
lr = 1

roi_shape = (450, 450)
roi_center = "center"
upsample = 1.0
n_center_indices = 25

dataset.meta.aberrations.array[0] = -21815.68
dataset.meta.aberrations.array[1] = -143.0
dataset.meta.aberrations.array[2] = -467.0

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
dataset.meta.aberrations.array

# %% 5) Direct ptychography (upsampled phase/intensity-style image) + quick display

upsample = 4.0
direct_ptycho_image, ssnr_ptycho = dataset.direct_ptychography(
    upsample=upsample, n_batches=8, return_snr=True, bright_field_mask_threshold=0.85)

direct_ptycho_image3 = direct_ptycho_image.clone()
p02, p98 = torch.quantile(direct_ptycho_image3.cpu(), torch.tensor([0.02, 0.98]))
direct_ptycho_image3 -= p02  # crude contrast stretch (floor at 2nd percentile) 

fig, ax = plt.subplots(figsize=(12, 6), dpi = 300)
vis.show_2d(
    direct_ptycho_image3,
    figax=(fig, ax),
    cmap="gray",
    cbar=True,
)

# %% 6) Tilt-corrected dark field (tcDF)

tcDF, ssnr_tcdf = dataset.tilt_corrected_dark_field(
    n_dark_field_segments=32,
    verbosity=0,
    bright_field_mask_threshold=0.3,
    upsample=upsample,
    return_snr=True,
    snr_blur_sigma=0.0,
)
fig, ax = plt.subplots(figsize=(6, 6))
fig_tcDF, ax_tcDF = vis.show_2d(
    [
        tcDF.T,
    ],
    cbar=True,
    title=["tcDF"],
    figax=(fig, ax),
)

# %% 7) Fused full field (phase- and tcDF-weighted components)

fff, phase_weighted, tcdf_weighted = dataset.fused_full_field(verbosity=2)
