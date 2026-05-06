"""
Carbon nanotube 4D-STEM example (80 kV): BF calibration, reconstructions, parallax figure.

An initial ``dk`` guess is **overwritten** after ``bright_field_radius_and_center_`` —
the BF disk sets reciprocal sampling. The script ends with a four-panel figure plus
power-spectrum insets comparing parallax vs. fused full field.

Run cell-by-cell using ``# %%`` markers.

**Before you run**

- **GPU**: CUDA by default.
- **Data**: ``DATA_PATH`` (4D stack). ``PARALLAX_PATH`` — external parallax image,
  cropped here to match the figure layout.
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

# %% 1) Device, beam parameters, load stack (dk below is updated in §4)

device = torch.device("cuda")
E = 80e3  # eV
semiangle_cutoff = 25e-3  # rad
dr = 0.316  # Å
dk = 1 / 2 / dr  # provisional; replaced after BF calibration

DATA_PATH = "/home/shengbo/data/vlp/nanotube171.npy"
data = np.load(DATA_PATH).astype(np.float16)

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

# %% 3) Bright-field crop

data_bf = dataset.crop_brightfield_(
    thresh_lower=0.1,
    thresh_upper=0.99,
)
vis.show_2d(data_bf.array[:25, :25].sum((0, 1)), cbar=True, title="data")

print(f"fluence = {dataset.fluence} e-/A^2")

# %% 4) k-space calibration (updates dk)

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

bright_field_mask_threshold = 0.5
correction_method = "total-variation" #"bright-field-shifts" 
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

# %% 6) Direct ptychography (Nyquist upsampling)

upsample = "nyquist"
direct_ptycho_image, ssnr_ptycho = dataset.direct_ptychography(
    upsample=upsample, n_batches=15, return_snr=True, bright_field_mask_threshold=0.8)


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

# %% 7) Tilt-corrected dark field (tcDF)

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

# %% 8) Fused full field

fff, phase_weighted, tcdf_weighted = dataset.fused_full_field(verbosity=2)

# %% 9) Load parallax image + build 4-panel figure and power-spectrum insets

PARALLAX_PATH = "/home/shengbo/data/vlp/pptube171.npy"
pp = np.load(PARALLAX_PATH)
pp = pp[16:-16, 16:-16]

from scipy.ndimage import gaussian_filter1d
def normalize01(a):
    a = np.asarray(a)
    amin = np.nanmin(a); amax = np.nanmax(a)
    return np.zeros_like(a, np.float32) if amax == amin else ((a - amin) / (amax - amin)).astype(np.float32)

pp_np  = 2 - np.asarray(pp)
# p02, p98 = np.percentile(pp_np, (10, 90))
# pp_n = np.clip((pp_np - p02) / (p98 - p02), 0, 1)
fff_np = fff.detach().cpu().numpy()
directptycho = direct_ptycho_image.cpu().numpy()
tcdf_np = tcDF.cpu().numpy()

pp_n  = normalize01(pp_np)
fff_n = normalize01(fff_np)
directptycho_n = normalize01(directptycho)
tcdf_n = normalize01(tcdf_np)

offset_down_pp   = -80      # pixels below the red line to start the ribbon (pp image)
band_height_pp   = 80      # ribbon thickness in pixels (pp image)

offset_down_fff  = -80      # fff is 2x, so roughly double to keep same *physical* size
band_height_fff  = 80
line_width       = 1.5

row_pp, row_fff = 300, 300
thickness = 3
thicknessf = thickness  
sigma = 10

fig, ax = plt.subplots(1, 4, figsize=(16, 8))
show_2d_array(directptycho_n, figax=(fig, ax[0]))
ax[0].set_title("Direct Ptychography")
ax[0].axis('off')
show_2d_array(tcdf_n, figax=(fig, ax[1]))
ax[1].set_title("Tilt-Corrected Dark Field")
ax[1].axis('off')
show_2d_array(pp_n, figax=(fig, ax[3]))
ax[3].set_title("Parallax Reconstruction")

ax[3].hlines(y=row_pp, xmin=0, xmax=pp_n.shape[1]-1, colors='r', linestyles='--', linewidth=2)

x0 = np.arange(pp_n.shape[1])
y = pp_n[row_pp:row_pp+thickness].astype(np.float32).mean(0)
ys = gaussian_filter1d(y, sigma=sigma, mode="nearest")
ysn = (ys - ys.min())
base0 = row_pp + offset_down_pp
y0 = base0 + (1.0 - ysn) * band_height_pp  
y0 = np.clip(y0, 0, pp_n.shape[0]-1)       


ax[3].axhspan(base0, base0 + band_height_pp, alpha=0.12, color='white', linewidth=0)
ax[3].plot(x0, y0, linewidth=line_width, color='b', zorder=3)
ax[3].axis('off')



show_2d_array(fff_n, figax=(fig, ax[2]))

ax[2].set_title("Fused Full Field")
ax[2].hlines(y=row_fff, xmin=0, xmax=fff_n.shape[1]-1, colors='r', linestyles='--', linewidth=2)

x1 = np.arange(fff_n.shape[1])
y = fff_n[row_fff:row_fff+thicknessf].astype(np.float32).mean(0)
ys = gaussian_filter1d(y, sigma=sigma, mode="nearest")

ysn = (ys - ys.min())
base1 = row_fff + offset_down_fff
y1 = base1 + (1.0 - ysn) * band_height_fff
y1 = np.clip(y1, 0, fff_n.shape[0]-1)

ax[2].axhspan(base1, base1 + band_height_fff, alpha=0.12, color='white', linewidth=0)
ax[2].plot(x1, y1, linewidth=line_width, color='b', zorder=3)
ax[2].axis('off')

sampling = 0.316
scalebar_length = 20  

add_scalebar_to_ax(
    ax=ax[3],
    array_size=20,   
    sampling=sampling,
    length_units=scalebar_length,          
    units="Å",
    width_px=4,                
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
    width_px=4,
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
    width_px=4,
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
    width_px=4,
    pad_px=1,
    color="white",
    loc="lower right",
)

for a in ax:
    a.set_xlabel("x (pixels)"); a.set_ylabel("y (pixels)")



# ---------- helpers ----------
def hann2d(ny, nx):
    wy = np.hanning(ny)[:, None]
    wx = np.hanning(nx)[None, :]
    return wy * wx

def power_spectrum(img, sampling_A_per_px):
    """Return power spectrum P and the frequency pixel size Δk (Å^-1 / px)."""
    img = np.asarray(img, np.float32)
    ny, nx = img.shape
    win = hann2d(ny, nx).astype(np.float32)
    x = (img - np.median(img)) * win
    F = np.fft.fftshift(np.fft.fft2(x))
    P = (np.abs(F) ** 2) / (win**2).sum()

    # frequency pixel size (Å^-1 per pixel)
    delta_k = 1.0 / (nx * sampling_A_per_px)  # assuming square pixels (nx=ny)
    return P, delta_k

# ---------- compute spectra with matched physical sampling ----------
sampling_pp  = 0.316         # Å/px

# Bin the FFF back to PP sampling for a fair comparison
# fff_binned = downscale_local_mean(fff_n, (2, 2))  # 1024->512

P_pp,  dk_pp  = power_spectrum(pp_n,        sampling_pp)
P_ff,  dk_ff  = power_spectrum(fff_n,  sampling_pp)  # NOTE: sampling_pp after binning

# Shared robust display range for the log power spectra
L_pp = np.log1p(P_pp); L_ff = np.log1p(P_ff)
vmin = np.percentile(np.r_[L_pp.ravel(), L_ff.ravel()], 5)
vmax = np.percentile(np.r_[L_pp.ravel(), L_ff.ravel()], 99.7)

# ---------- make insets (top-right) and add scale bars in Å^-1 ----------
# Size/placement of the inset inside each main axes: [left, bottom, width, height] in axes-fraction
inset_rect = [0.68, 0.68, 0.32, 0.32]  # top-right corner
crop_pixels = 180
L_pp = L_pp[crop_pixels:-crop_pixels, crop_pixels:-crop_pixels]  # crop for better visibility
L_ff = L_ff[crop_pixels:-crop_pixels, crop_pixels:-crop_pixels]
# Parallax inset
ax_ps_pp = ax[3].inset_axes(inset_rect)
im0 = ax_ps_pp.imshow(L_pp, cmap='gray', vmin=vmin, vmax=vmax)
ax_ps_pp.set_xticks([]); ax_ps_pp.set_yticks([])

from scatterem2.vis.visualization_utils import add_scalebar_to_ax
add_scalebar_to_ax(
    ax=ax_ps_pp,
    array_size=L_pp.shape[1],       # spectrum width in pixels
    sampling=dk_pp,                 # Δk = Å^-1 per pixel
    length_units=None,              # let helper pick a "nice" k-length
    units="Å⁻¹",
    width_px=2,
    pad_px=0.5,
    color="white",
    loc="upper right",
)

# FFF inset (using binned spectrum so it matches PP bandwidth)
ax_ps_ff = ax[2].inset_axes(inset_rect)
im1 = ax_ps_ff.imshow(L_ff, cmap='gray', vmin=vmin, vmax=vmax)
ax_ps_ff.set_xticks([]); ax_ps_ff.set_yticks([])
# ax_ps_ff.set_title("PS", fontsize=9, pad=2)
add_scalebar_to_ax(
    ax=ax_ps_ff,
    array_size=L_ff.shape[1],
    sampling=dk_pp,                 # same Δk since we binned to PP sampling
    length_units=0.2,
    units="Å⁻¹",
    width_px=2,
    pad_px=0.5,
    color="white",
    loc="upper right",
)

plt.tight_layout(); plt.show()
