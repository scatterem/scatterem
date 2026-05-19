# %%
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
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# %%

data = np.load('/tank/storage/shengbo/vlp/amorphous_carbon_bg30_thick70_defn150_publish.npy').astype(np.float32)

print(f"data.shape = {data.shape}")

plt.figure()
plt.imshow(np.mean(data, axis=(2, 3)), cmap="inferno")
plt.colorbar()

N, _, M, _ = data.shape

E = 300e3
skip = 1

wavelength = energy2wavelength(E)
semiangle_cutoff = 19.68e-3

device = torch.device("cuda")

dr = 0.2

meta = Metadata4dstem(
    energy=E,
    semiconvergence_angle=float(semiangle_cutoff),
    rotation=0,
    defocus_guess=float(0),
    sample_thickness_guess=0,
    vacuum_probe=None,
    sampling=(dr * skip, dr * skip, 1, 1),
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
# %
data_bf = dataset.crop_brightfield_(
    thresh_lower=0.1,
    thresh_upper=0.99,
)
vis.show_2d(data_bf.array[:25, :25].sum((0, 1)), cbar=True, title="data")

print(f"fluence = {dataset.fluence} e-/A^2")
# %%

rBF, c = data_bf.bright_field_radius_and_center_(
    thresh_lower=0.1, thresh_upper=0.99, plot_rbf=True, method="area"
)
dalpha = semiangle_cutoff / rBF
dk = dalpha / data_bf.meta.wavelength

data_bf.meta.sampling = (dr * skip, dr * skip, dk, dk)
data_bf.sampling = (dr * skip, dr * skip, dk, dk)
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
# %%
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

roi_shape = (150, 150)
roi_center = "center"
upsample = 1.0
n_center_indices = 25
dataset.meta.aberrations.array[0] = -50.0
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
# %%
upsample = 1.0
direct_ptycho_image, ssnr_ptycho = dataset.direct_ptychography(
    upsample=upsample, verbosity=1, return_snr=True, n_batches=8)

# %%
tcDF, ssnr_tcdf = dataset.tilt_corrected_dark_field(
    n_dark_field_segments=6,
    verbosity=0,
    bright_field_mask_threshold=bright_field_mask_threshold,
    upsample=upsample,
    return_snr=True,
    snr_blur_sigma=2.0,
)

# %%
fff, phase_weighted, tcdf_weighted = dataset.fused_full_field(verbosity=2)

# %%
from scatterem2.utils.utils import radial_average2
import matplotlib.pyplot as plt

gt = np.load('/tank/storage/shengbo/vlp/amorphous_carbon_bg30_thick70_gt.npy').astype(np.float32)


def normalize01(a):
    a = np.asarray(a)
    amin = np.nanmin(a);
    amax = np.nanmax(a)
    return np.zeros_like(a, np.float32) if amax == amin else ((a - amin) / (amax - amin)).astype(np.float32)


dp_img = direct_ptycho_image.cpu().numpy()
tcdf_img = tcDF.cpu().numpy()
fff_img = fff.cpu().numpy()

dptop_n = normalize01(dp_img)
tcdftop_n = normalize01(tcdf_img)
ffftop_n = normalize01(fff_img)
gttop_n = normalize01(gt)

p0_05 = (30, 45)
p1_05 = (80, 45)

p0_08 = (10, 200)
p1_08 = (80, 200)

p0_10 = (30, 125)
p1_10 = (220, 125)

p0_12 = (150, 65)
p1_12 = (240, 65)

p0_15 = (140, 195)
p1_15 = (245, 195)

dash_width = 3

fig, ax = plt.subplots(1, 4, figsize=(20, 8), dpi=100)

cutedge = 1
ax[0].imshow(dptop_n, cmap="gray", vmin=0, vmax=1)
ax[0].axis('off')
ax[1].imshow(tcdftop_n, cmap="gray", vmin=0, vmax=1)
ax[1].axis('off')
ax[2].imshow(ffftop_n, cmap="gray", vmin=0, vmax=1)

ax[2].plot([p0_05[0], p1_05[0]], [p0_05[1], p1_05[1]], "b--", linewidth=3)
ax[2].plot([p0_08[0], p1_08[0]], [p0_08[1], p1_08[1]], "b--", linewidth=3)
ax[2].plot([p0_10[0], p1_10[0]], [p0_10[1], p1_10[1]], "b--", linewidth=3)
ax[2].plot([p0_12[0], p1_12[0]], [p0_12[1], p1_12[1]], "b--", linewidth=3)
ax[2].plot([p0_15[0], p1_15[0]], [p0_15[1], p1_15[1]], "b--", linewidth=3)

ax[2].axis('off')
ax[3].imshow(gttop_n, cmap="gray", vmin=0, vmax=1)

ax[3].plot([p0_05[0], p1_05[0]], [p0_05[1], p1_05[1]], "r--", linewidth=3)
ax[3].plot([p0_08[0], p1_08[0]], [p0_08[1], p1_08[1]], "r--", linewidth=3)
ax[3].plot([p0_10[0], p1_10[0]], [p0_10[1], p1_10[1]], "r--", linewidth=3)
ax[3].plot([p0_12[0], p1_12[0]], [p0_12[1], p1_12[1]], "r--", linewidth=3)
ax[3].plot([p0_15[0], p1_15[0]], [p0_15[1], p1_15[1]], "r--", linewidth=3)

ax[3].axis('off')

sampling = dr
scalebar_length = 10  # Å
width_px = 8
for i in range(3):
    add_scalebar_to_ax(
        ax=ax[i],
        array_size=20,
        sampling=sampling,
        length_units=scalebar_length,
        units="Å",
        width_px=width_px,
        pad_px=1,
        color="white",
        loc="lower right",
    )

plt.tight_layout()
# %%