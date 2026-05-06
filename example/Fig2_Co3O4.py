"""
Co₃O₄ (VLP) 4D-STEM example (200 kV): standard reconstructions plus a 4-panel figure.

After the usual BF calibration, aberration fit, direct ptychography, tcDF, and fused
full field, the script loads a **parallax** reconstruction from disk and builds a
publication-style figure (direct ptychography, tcDF, fused, parallax) with overlaid
line profiles and scale bars.

Run cell-by-cell using ``# %%`` markers.

**Before you run**

- **GPU**: CUDA by default.
- **Data**: ``DATA_PATH`` — 4D ``.npy`` stack. ``PARALLAX_PATH`` — parallax / phase image
  for side-by-side comparison (same approximate field of view as reconstructions).
- **Extra deps**: ``scipy``, ``scikit-image`` (for the final figure block).

**Pipeline**

1. Load 4D data; beam parameters.
2. Wavelength, metadata, ``Dataset4dstem``.
3. Bright-field crop.
4. Calibrate ``dk`` from BF disk.
5. Aberration fit.
6. Direct ptychography (Nyquist upsampling); phase image display.
7. tcDF + side-by-side with phase.
8. Fused full field.
9. Load parallax array.
10. Multi-panel figure: normalized images, profile ribbons, scale bars.
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

# %% 1) Device, beam parameters, load 4D stack

device = torch.device("cuda")
E = 200e3  # eV
semiangle_cutoff = 21e-3  # rad
dr = 0.2  # Å

DATA_PATH = "/home/shengbo/data/vlp/co3o4_data.npy"
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

# %% 4) k-space calibration

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

# %% 6) Direct ptychography (Nyquist)

direct_ptycho_image, ssnr_ptycho = dataset.direct_ptychography(
    upsample="nyquist", verbosity=1, return_snr=True, n_batches=n_batches)

print(dataset.meta)
direct_ptycho_image2 = dataset.direct_ptychography_phase_image


p02, p98 = torch.quantile(direct_ptycho_image2.cpu(), torch.tensor([0.02, 0.98]))
print(f"2nd percentile: {p02:.3f}")
print(f"98th percentile: {p98:.3f}")
direct_ptycho_image2 -= p02  # display stretch

fig_bf_analytic2, ax_bf_analytic2 = vis.show_2d(
    [direct_ptycho_image2],
    cbar=True,
    title=["Phase, BF reconstruction"],
)

# %% 7) Tilt-corrected dark field + comparison to phase

tcDF, ssnr_tcdf = dataset.tilt_corrected_dark_field(
    n_dark_field_segments=16,
    verbosity=0,
    bright_field_mask_threshold=bright_field_mask_threshold,
    upsample=upsample,
    return_snr=True,
    snr_blur_sigma=0.0,
)

fig, ax = plt.subplots(1, 2, figsize=(8, 8))
fig_bf_analytic2, ax_bf_analytic2 = vis.show_2d(
    [tcDF, direct_ptycho_image2],
    cbar=True,
    title=["tcDF", "Phase, BF reconstruction"],
    figax=(fig, ax),
)

print(tcDF.shape, direct_ptycho_image2.shape)

# %% 8) Fused full field

fff, phase_weighted, tcdf_weighted = dataset.fused_full_field(verbosity=2)

# %% 9) Load parallax (external) reconstruction for comparison

PARALLAX_PATH = "/home/shengbo/data/vlp/sim1ppn.npy"
pp = np.load(PARALLAX_PATH).astype(np.float32)

# %% 10) Four-panel figure: profiles, blended phase/tcDF columns, scale bars

from scipy.ndimage import gaussian_filter1d
from scatterem2.vis.visualization_utils import add_scalebar_to_ax
from skimage.transform import downscale_local_mean
from scatterem2.vis.visualization import show_2d_array

def normalize01(a):
    a = np.asarray(a)
    amin = np.nanmin(a); amax = np.nanmax(a)
    return np.zeros_like(a, np.float32) if amax == amin else ((a - amin) / (amax - amin)).astype(np.float32)

pp_np  = np.asarray(pp)
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
fdp[:, 92:] = dpw_n[:, 92:]

ftcdf = tcdf_n.copy()
ftcdf[:, 92:] = tcdfw_n[:, 92:]

offset_down_pp   = -30      
band_height_pp   = 30      # ribbon thickness in pixels (pp image)

offset_down_fff  = -30      # fff is 2x, so roughly double to keep same *physical* size
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
ax[0].set_title("Direct Ptychography")


ax[0].hlines(y=row_fff_co, xmin=0, xmax=fdp.shape[1]-1, colors='yellow', linestyles='--', linewidth=2)
ax[0].hlines(y=row_fff_o, xmin=0, xmax=fdp.shape[1]-1, colors='yellow', linestyles='--', linewidth=2)

x1 = np.arange(fdp.shape[1])

yco = fdp[row_fff_co:row_fff_co+thicknessf].astype(np.float32).mean(0)
ysco = gaussian_filter1d(yco, sigma=sigma, mode="nearest")
ysnco = (ysco - ysco.min()) * scale_factor
base1co = row_fff_co + offset_down_fff
y1co = base1co + (1.0 - ysnco) * band_height_fff
y1co = np.clip(y1co, 0, fdp.shape[0]-1)

# ax[2].axhspan(base1co, base1co + band_height_fff, alpha=0.12, color='white', linewidth=0)
ax[0].plot(x1, y1co, linewidth=line_width, color='b', zorder=3)


yo = fdp[row_fff_o:row_fff_o+thicknessf].astype(np.float32).mean(0)
yso = gaussian_filter1d(yo, sigma=sigma, mode="nearest")
ysno = (yso - yso.min()) * scale_factor
base1o = row_fff_o + offset_down_fff
y1o = base1o + (1.0 - ysno) * band_height_fff
y1o = np.clip(y1o, 0, fdp.shape[0]-1)

ax[0].plot(x1, y1o, linewidth=line_width, color='r', zorder=3)

ax[0].axis('off')


show_2d_array(ftcdf, figax=(fig, ax[1]))
ax[1].set_title("tcDF")

ax[1].hlines(y=row_fff_co, xmin=0, xmax=ftcdf.shape[1]-1, colors='yellow', linestyles='--', linewidth=2)
ax[1].hlines(y=row_fff_o, xmin=0, xmax=ftcdf.shape[1]-1, colors='yellow', linestyles='--', linewidth=2)

x1 = np.arange(ftcdf.shape[1])

yco = ftcdf[row_fff_co:row_fff_co+thicknessf].astype(np.float32).mean(0)
ysco = gaussian_filter1d(yco, sigma=sigma, mode="nearest")
ysnco = (ysco - ysco.min()) * scale_factor
base1co = row_fff_co + offset_down_fff
y1co = base1co + (1.0 - ysnco) * band_height_fff
y1co = np.clip(y1co, 0, ftcdf.shape[0]-1)

ax[1].plot(x1, y1co, linewidth=line_width, color='b', zorder=3)


yo = ftcdf[row_fff_o:row_fff_o+thicknessf].astype(np.float32).mean(0)
yso = gaussian_filter1d(yo, sigma=sigma, mode="nearest")
ysno = (yso - yso.min()) * scale_factor
base1o = row_fff_o + offset_down_fff
y1o = base1o + (1.0 - ysno) * band_height_fff
y1o = np.clip(y1o, 0, ftcdf.shape[0]-1)

# ax[2].axhspan(base1o, base1o + band_height_fff, alpha=0.12, color='white', linewidth=0)
ax[1].plot(x1, y1o, linewidth=line_width, color='r', zorder=3)
ax[1].axis('off')



show_2d_array(pp_n, figax=(fig, ax[3]))
ax[3].set_title("Parallax Reconstruction")

ax[3].hlines(y=row_pp_co, xmin=0, xmax=pp_n.shape[1]-1, colors='yellow', linestyles='--', linewidth=2)
ax[3].hlines(y=row_pp_o, xmin=0, xmax=pp_n.shape[1]-1, colors='yellow', linestyles='--', linewidth=2)

x0 = np.arange(pp_n.shape[1])
yco = pp_n[row_pp_co:row_pp_co+thickness].astype(np.float32).mean(0)
ysco = gaussian_filter1d(yco, sigma=sigma, mode="nearest")

# normalize to [0,1] so the band mapping is contrast-agnostic
ysnco = (ysco - ysco.min()) * scale_factor
# map into a band entirely *below* the red line
base0co = row_pp_co + offset_down_pp
y0co = base0co + (1.0 - ysnco) * band_height_pp  # high intensity closer to base (top of band)
y0co = np.clip(y0co, 0, pp_n.shape[0]-1)       # stay in image


# draw the curve
ax[3].plot(x0, y0co, linewidth=line_width, color='b', zorder=3)

yo = pp_n[row_pp_o:row_pp_o+thickness].astype(np.float32).mean(0)
yso = gaussian_filter1d(yo, sigma=sigma, mode="nearest")
ysno = (yso - yso.min()) * scale_factor
# map into a band entirely *below* the red line
base0o = row_pp_o + offset_down_pp
y0o = base0o + (1.0 - ysno) * band_height_pp  # high intensity closer to base (top of band)
y0o = np.clip(y0o, 0, pp_n.shape[0]-1)       # stay in image

# draw the curve
ax[3].plot(x0, y0co, linewidth=line_width, color='b', zorder=3)
ax[3].plot(x0, y0o, linewidth=line_width, color='r', zorder=3)
ax[3].axis('off')




show_2d_array(fff_n, figax=(fig, ax[2]))
ax[2].set_title("Fused Full Field")

ax[2].hlines(y=row_fff_co, xmin=0, xmax=fff_n.shape[1]-1, colors='yellow', linestyles='--', linewidth=2)
ax[2].hlines(y=row_fff_o, xmin=0, xmax=fff_n.shape[1]-1, colors='yellow', linestyles='--', linewidth=2)

x1 = np.arange(fff_n.shape[1])

yco = fff_n[row_fff_co:row_fff_co+thicknessf].astype(np.float32).mean(0)
ysco = gaussian_filter1d(yco, sigma=sigma, mode="nearest")
ysnco = (ysco - ysco.min()) * scale_factor
base1co = row_fff_co + offset_down_fff
y1co = base1co + (1.0 - ysnco) * band_height_fff
y1co = np.clip(y1co, 0, fff_n.shape[0]-1)

ax[2].plot(x1, y1co, linewidth=line_width, color='b', zorder=3)


yo = fff_n[row_fff_o:row_fff_o+thicknessf].astype(np.float32).mean(0)
yso = gaussian_filter1d(yo, sigma=sigma, mode="nearest")
ysno = (yso - yso.min()) * scale_factor
base1o = row_fff_o + offset_down_fff
y1o = base1o + (1.0 - ysno) * band_height_fff
y1o = np.clip(y1o, 0, fff_n.shape[0]-1)

ax[2].plot(x1, y1o, linewidth=line_width, color='r', zorder=3)

ax[2].axis('off')


sampling = dr
scalebar_length = 5  
add_scalebar_to_ax(
    ax=ax[3],
    array_size=20, 
    sampling=sampling,
    length_units=scalebar_length,         
    units="Å",
    width_px=2,              
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
    width_px=2,
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
    width_px=2,
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
    width_px=2,
    pad_px=1,
    color="white",
    loc="lower right",
)

for a in ax:
    a.set_xlabel("x (pixels)"); a.set_ylabel("y (pixels)")

plt.tight_layout(); plt.show()
