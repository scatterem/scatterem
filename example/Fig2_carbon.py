"""
Carbon 4D-STEM example (300 kV): reconstructions vs. a loaded “ground truth” potential.

Runs the usual bright-field calibration, aberration fit, direct ptychography (Nyquist),
tcDF, and fused full field. A later cell loads a reference potential array and builds a
four-panel figure with **SSIM** vs. ground truth and **power-spectrum insets** (needs
``scikit-image``).

Run cell-by-cell using ``# %%`` markers.

**Before you run**

- **GPU**: CUDA by default.
- **Data**: ``DATA_PATH`` — 4D stack. ``POTENTIAL_PATH`` — 2D reference (same script
  assumes comparable shape after processing; adjust cropping if needed).
"""

import matplotlib

matplotlib.use("inline")  # Notebook-friendly backend

import matplotlib.pyplot as plt
import numpy as np
import torch
import math

from scatterem2.vis.visualization import show_2d_array
from scatterem2.vis.visualization_utils import add_scalebar_to_ax

import scatterem2.vis as vis

from scatterem2.utils.data.aberrations import Aberrations
from scatterem2.utils.data.data_classes import Metadata4dstem
from scatterem2.utils.data.datasets import Dataset4dstem


from scatterem2.utils.stem import energy2wavelength

from pathlib import Path

# %% 1) Device, beam parameters, load 4D stack

device = torch.device("cuda")
E = 300e3  # eV
semiangle_cutoff = 19.68e-3  # rad
dr = 0.25  # Å

DATA_PATH = "/home/shengbo/data/vlp/carbon_72.npy"
data = np.load(DATA_PATH).astype(np.float32)

# %% 2) Wavelength + metadata + dataset

wavelength = energy2wavelength(E)




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

dataset_full = Dataset4dstem.from_array(
    array=data,
    origin=np.array((ny / 2, nx / 2, M / 2, M / 2), dtype=np.float32),
    name="dataset_full",
    signal_units="arb. units",
    meta=meta,
    transform_to_amplitudes=False,
    device=device,
    normalize=True,
)
print(dataset_full)

dataset = dataset_full  # .bin_detector(bin_factor=2)

total_intensity = dataset.array.sum()
print(f"Total intensity: {total_intensity}")
print(dataset)

# %% 3) Bright-field crop

data_bf = dataset.crop_brightfield_(
    thresh_lower=0.1,
    thresh_upper=0.99,
)
vis.show_2d(data_bf.array[:25, :25].sum((0, 1)), cbar=True, title="data")

print(f"fluence = {dataset.fluence} e-/A^2")

# %% 4) k-space calibration

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

# %% 6) Direct ptychography (Nyquist)

direct_ptycho_image, ssnr_ptycho = dataset.direct_ptychography(
    upsample="nyquist", verbosity=1, return_snr=True, n_batches=n_batches)

direct_ptycho_image2 = dataset.direct_ptychography_phase_image
p02, p98 = torch.quantile(direct_ptycho_image2.cpu(), torch.tensor([0.02, 0.98]))
direct_ptycho_image2 -= p02  # display stretch

fig_bf_analytic2, ax_bf_analytic2 = vis.show_2d(
    [direct_ptycho_image2],
    cbar=True,
    title=["Phase, BF reconstruction"],
)

# %% 7) Tilt-corrected dark field (tcDF)

tcDF, ssnr_tcdf = dataset.tilt_corrected_dark_field(
    n_dark_field_segments=16,
    verbosity=0,
    bright_field_mask_threshold=0.85,
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

# %% 8) Fused full field

fff, phase_weighted, tcdf_weighted = dataset.fused_full_field(verbosity=2)

# %% 9) Load reference potential (ground truth for metrics / figure)

POTENTIAL_PATH = "/home/shengbo/data/vlp/carbon72_potential.npy"
pot = np.load(POTENTIAL_PATH)

# %% 10) SSIM + power spectra + scale bars (4-panel figure)

from skimage.metrics import structural_similarity as ssim

from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def power_spectrum(img, eps=1e-12):
    F = np.fft.fftshift(np.fft.fft2(img))
    P = np.abs(F) ** 2
    return P

def normalize01(a):
    a = np.asarray(a)
    amin = np.nanmin(a); amax = np.nanmax(a)
    return np.zeros_like(a, np.float32) if amax == amin else ((a - amin) / (amax - amin)).astype(np.float32)

pp_np  = np.asarray(pot)
fff_np = fff.detach().cpu().numpy()
directptycho = direct_ptycho_image2.cpu().numpy()
tcdf_np = tcDF.cpu().numpy()
dpw_np = phase_weighted.cpu().numpy()
tcdfw_np = tcdf_weighted.cpu().numpy()

pp_n  = normalize01(pp_np)
fff_n = normalize01(fff_np)
directptycho_n = normalize01(directptycho)
tcdf_n = normalize01(tcdf_np)
dpw_n = normalize01(dpw_np)
tcdfw_n = normalize01(tcdfw_np)

fdp = directptycho_n.copy()
ftcdf = tcdf_n.copy()

offset_down_pp   = -30      
band_height_pp   = 30     

offset_down_fff  = -30   
band_height_fff  = 30
line_width       = 1.5

row_pp_co, row_fff_co = 150, 150
row_pp_o, row_fff_o = 54, 54
thickness = 1
thicknessf = thickness  
sigma = 0.1
scale_factor = 1


fig, ax = plt.subplots(1, 4, figsize=(16, 8))
show_2d_array(fdp, figax=(fig, ax[0]))
img1 = pp_n
img2 = fdp
ssim_dp = ssim(img1, img2, data_range=img2.max() - img2.min())
print(ssim_dp)
psdp = power_spectrum(fdp)
axins = inset_axes(
    ax[0],
    width="30%",   # relative to parent axis
    height="30%",
    loc="upper right",
    borderpad=0.8
)
show_2d_array(psdp[10:-10, 10:-10], figax=[fig, axins])
axins.set_xticks([])
axins.set_yticks([])
delta_k = 1.0 / (ssnr_ptycho.shape[0] * dr / upsample)

scalebar_length = 1

add_scalebar_to_ax(
    ax=axins,
    array_size=20,
    sampling=delta_k,
    length_units=scalebar_length,
    units="Å",
    width_px=1,
    pad_px=1,
    color="white",
    loc="lower right",
)

ax[0].set_title(f"Direct Ptychography, SSIM:{ssim_dp:.3f}")
ax[0].axis('off')

show_2d_array(ftcdf, figax=(fig, ax[1]))
img2 = ftcdf
ssim_tcdf = ssim(img1, img2, data_range=img2.max() - img2.min())
print(ssim_tcdf)
ax[1].set_title(f"tcDF, SSIM:{ssim_tcdf:.3f}")

pstcdf = power_spectrum(ftcdf)
axins = inset_axes(
    ax[1],
    width="30%",   # relative to parent axis
    height="30%",
    loc="upper right",
    borderpad=0.8
)
show_2d_array(pstcdf[10:-10, 10:-10], figax=[fig, axins])
axins.set_xticks([])
axins.set_yticks([])
delta_k = 1.0 / (ssnr_ptycho.shape[0] * dr / upsample)

scalebar_length = 1

add_scalebar_to_ax(
    ax=axins,
    array_size=20,
    sampling=delta_k,
    length_units=scalebar_length,
    units="Å",
    width_px=1,
    pad_px=1,
    color="white",
    loc="lower right",
)

ax[1].axis('off')

show_2d_array(pp_n, figax=(fig, ax[3]))
ax[3].set_title(f"Ground truth potential")
pspp = power_spectrum(pp_n)
axins = inset_axes(
    ax[3],
    width="30%", 
    height="30%",
    loc="upper right",
    borderpad=0.8
)
show_2d_array(pspp[10:-10, 10:-10], figax=[fig, axins])
axins.set_xticks([])
axins.set_yticks([])
delta_k = 1.0 / (ssnr_ptycho.shape[0] * dr / upsample)

scalebar_length = 1

add_scalebar_to_ax(
    ax=axins,
    array_size=20,
    sampling=delta_k,
    length_units=scalebar_length,
    units="Å",
    width_px=1,
    pad_px=1,
    color="white",
    loc="lower right",
)
ax[3].axis('off')

show_2d_array(fff_n, figax=(fig, ax[2]))
img2 = fff_n
ssim_fff = ssim(img1, img2, data_range=img2.max() - img2.min())
print(ssim_fff)
ax[2].set_title(f"Fused Full Field, SSIM:{ssim_fff:.3f}")

psff = power_spectrum(fff_n)
axins = inset_axes(
    ax[2],
    width="30%",  
    height="30%",
    loc="upper right",
    borderpad=0.8
)
show_2d_array(psff[10:-10, 10:-10], figax=[fig, axins])
axins.set_xticks([])
axins.set_yticks([])
delta_k = 1.0 / (ssnr_ptycho.shape[0] * dr / upsample)

scalebar_length = 1

add_scalebar_to_ax(
    ax=axins,
    array_size=20,
    sampling=delta_k,
    length_units=scalebar_length,
    units="Å",
    width_px=1,
    pad_px=1,
    color="white",
    loc="lower right",
)

ax[2].axis('off')

sampling = dr
scalebar_length = 5  # Å
add_scalebar_to_ax(
    ax=ax[3],
    array_size=20,  
    sampling=sampling,
    length_units=scalebar_length,          
    units="Å",
    width_px=1,                
    pad_px=1,
    color="white",
    loc="lower right",          
)

add_scalebar_to_ax(
    ax=ax[2],
    array_size=20,
    sampling=sampling,
    length_units=scalebar_length,
    units="Å",
    width_px=1,
    pad_px=1,
    color="white",
    loc="lower right",
)

add_scalebar_to_ax(
    ax=ax[0],
    array_size=20,
    sampling=sampling,
    length_units=scalebar_length,
    units="Å",
    width_px=1,
    pad_px=1,
    color="white",
    loc="lower right",
)
add_scalebar_to_ax(
    ax=ax[1],
    array_size=20,
    sampling=sampling,
    length_units=scalebar_length,
    units="Å",
    width_px=1,
    pad_px=1,
    color="white",
    loc="lower right",
)

for a in ax:
    a.set_xlabel("x (pixels)"); a.set_ylabel("y (pixels)")

plt.tight_layout(); plt.show()
# %%