# Quickstart

## The whole pipeline

The few lines below take a 4D-STEM array to the fused image. `array` is your
data, a NumPy array of shape `(scan_y, scan_x, det_y, det_x)`.

```python
from scatterem import Dataset4DStem
from scatterem.reconstruction import run
from scatterem.reconstruction.fusion.ffstem import FusedFullFieldConfig
from scatterem.reconstruction.ptychography.direct import AberrationFitConfig

dataset = Dataset4DStem.from_array(
    array,                          # (scan_y, scan_x, det_y, det_x)
    energy=300e3,                   # eV
    semiconvergence_angle=19.7e-3,  # rad
    scan_step=0.25,                 # Angstrom
    device="cuda",
)
dataset.calibrate_reciprocal_from_bright_field()  # measure dk; from_array sets a placeholder

fit = run(
    dataset,
    AberrationFitConfig(correct_order=2, bright_field_mask_threshold=dataset.bright_field_mask_threshold),
)
dataset.meta = fit.applied_to(dataset.meta)

fused = run(dataset, FusedFullFieldConfig(n_dark_field_segments=32)).fused
```

That is the whole pipeline; the sections below go through it step by step.
Each channel is also available on its own, through
`run(dataset, DirectPtychographyConfig(...))` and
`run(dataset, TiltCorrectedDarkFieldConfig(...))`. All three share one
calibration, so they land on the same grid.

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/scatterem/scatterem/blob/main/notebooks/ffstem_demo.py)

Run it now on a free GPU: the molab notebook installs the package, fetches one
dataset from Zenodo and reconstructs it, with nothing to set up locally.

## From an array

The container needs the acquisition geometry, because every reconstruction here is
calibrated — the fusion weights are functions of spatial frequency, so the code has
to know what a pixel is worth.

```python
import numpy as np
from scatterem import Dataset4DStem

array = np.load("my_4dstem.npy")   # (scan_y, scan_x, det_y, det_x)

dataset = Dataset4DStem.from_array(
    array,
    energy=300e3,                   # accelerating voltage, eV
    semiconvergence_angle=19.7e-3,  # convergence semi-angle, rad
    scan_step=0.25,                 # scan pitch, Angstrom
    device="cuda",
)
dataset.calibrate_reciprocal_from_bright_field()  # measure dk
```

`scan_step` may be a scalar or a `(dy, dx)` pair. You must calibrate the
reciprocal pixel size `dk` yourself. `from_array` sets `dk` to a `(1.0, 1.0)`
placeholder, and `run()` warns if the placeholder is still there. Call
`calibrate_reciprocal_from_bright_field()`, which measures `dk` from the
bright-field disk, or pass a known `reciprocal_step=` to `from_array`. See
[Calibration](calibration.md): most reconstruction problems start there.

## Fit the aberrations

The reconstruction needs a decent aberration estimate. The default `"combined"`
fit gets its start from the bright-field shifts, so it needs no start value:

```python
from scatterem.reconstruction import run
from scatterem.reconstruction.darkfield import TiltCorrectedDarkFieldConfig
from scatterem.reconstruction.fusion.ffstem import FusedFullFieldConfig
from scatterem.reconstruction.ptychography.direct import (
    AberrationFitConfig,
    DirectPtychographyConfig,
)

fit = run(
    dataset,
    AberrationFitConfig(correct_order=2, bright_field_mask_threshold=dataset.bright_field_mask_threshold),
)
dataset.meta = fit.applied_to(dataset.meta)
```

To use only the autofocus, which maximises image sharpness, give it a start
value:

```python
dataset.meta.aberrations.array[0] = -50.0   # C10, Angstrom; defocus == -C10
fit = run(
    dataset,
    AberrationFitConfig(
        correction_method="autofocus",
        correct_order=2,
        bright_field_mask_threshold=dataset.bright_field_mask_threshold,
    ),
)
dataset.meta = fit.applied_to(dataset.meta)
```

:::{admonition} Give the autofocus a defocus
:class: important

With `correction_method="autofocus"` the fit starts from `meta.aberrations`. If
C10, C12a and C12b are all zero (a known Cs alone does not count), a
bright-field-shift fit gives the start. If that fit fails
and `meta.defocus_guess` is 0, the autofocus has nothing to descend from, and it
raises `ValueError` rather than return a result that depends only on the start.
The default `"combined"` fit does not read `meta.aberrations`. For the
autofocus, any estimate within a factor of two or so
of the truth is enough; the sign matters more than the magnitude.

Index 0 is `C10`, and `defocus == -C10`.
:::

`correct_order=2` fits up to second order (defocus, two-fold astigmatism, coma,
three-fold astigmatism). Raise it to reach spherical aberration; the higher orders
need more signal to constrain.

The autofocus holds the scan rotation **fixed**, so `meta.rotation` must already be
right on the way in. If you do not know it, see [Calibration](calibration.md).

## Reconstruct

```python
threshold = dataset.bright_field_mask_threshold

ptycho     = run(dataset, DirectPtychographyConfig(bright_field_mask_threshold=threshold))
dark_field = run(dataset, TiltCorrectedDarkFieldConfig(bright_field_mask_threshold=threshold))
fused      = run(dataset, FusedFullFieldConfig(n_dark_field_segments=32)).fused
```

`DirectPtychographyConfig` and `AberrationFitConfig` default the bright-field mask
threshold to 0.5; the paper's path uses the dataset's threshold (0.3), so pass it
explicitly as shown.

Each config takes `upsample`, which defaults to `"nyquist"` — the factor that brings the
output pitch to the resolution the data can actually support, rather than a round
number someone picked. Pass a float to override.

Each `run` returns a result and writes nothing on the dataset.

`n_dark_field_segments` sets how many detector segments the dark-field channel
of the fusion uses. It is a scientific parameter: it changes the wedges, the
per-segment shifts and the half-set split. Give it explicitly. If you do not,
the fusion uses `n_batches` (32) and warns.

To fuse again without a new calculation of the channels, use the channels of
an earlier fusion result `r`: pass `reuse={"ptycho": r.ptycho, "tcdf": r.tcdf}`.
Do not pass the `ptycho` and `dark_field` results from above. The fusion
calculates its channels with its own settings (`return_snr=True`, and
`n_batches=32` for the ptychography channel). It refuses a result that has
different settings.

## Look at the result

```python
from scatterem.data.sampling import reconstruction_sampling
from scatterem.vis.visualization import show_2d

fig, ax = show_2d(fused, sampling=reconstruction_sampling(dataset, fused))
fig.savefig("fused.png", dpi=150, bbox_inches="tight")
```

Passing the sampling is what makes the scale bar read in Ångström instead of
`a.u.` — and it must be the *reconstruction's* sampling, not the dataset's. The
output is upsampled, so `dataset.sampling` would overstate the bar by the upsample
factor: four-fold on the paper's Fig. 4.

## Save and reload

```python
dataset.save("carbon.h5")
same = Dataset4DStem.load("carbon.h5", device="cuda")
```

The file records which fields it contains. If you later add or rename a field on
the class, loading an old file **fails** and names the difference, rather than
handing you a half-populated object. See
[`scatterem.utils.store`](api.md#saving-and-loading).

## The named datasets

The four specimens from the paper download themselves, verified by md5:

```python
from scatterem import You2026Carbon

dataset = You2026Carbon(root="~/ffstem-data", download=True, device="cuda")
```

They arrive already calibrated. See [Figure scripts](figures.md).
