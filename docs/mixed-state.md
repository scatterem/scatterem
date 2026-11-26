# Mixed-state ptychography

This page covers the reconstruction of a partially coherent probe: a 2D object
and a probe of several incoherent modes, from one 4D-STEM scan. It then shows
how to read the probe's orbital angular momentum (OAM) spectrum and its OAM
density matrix. The method is the one of S. You *et al.*, *Transverse
quantum-state characterization of programmable electron optics*,
[arXiv:2608.05749](https://doi.org/10.48550/arXiv.2608.05749).

## The idea in one paragraph

A probe that is not fully coherent is a density matrix, not a wave function.
Ptychography can measure it in a finite-rank form: a small set of orthogonal
modes $\psi_m$, each with its own power. Each mode passes through the same
object, and the detector adds the intensities of the modes:
$I(\mathbf{k}) = \sum_m |\mathcal{F}[\psi_m \cdot O]|^2$. The solver refines
the modes, the object and the scan positions together, so that the computed
amplitudes match the measured ones. From the modes you get the purity
$\sum_m p_m^2$ of the state ($p_m$ is the power fraction of mode $m$), and its
projection onto OAM eigenstates.

## The parts

| Part | What it does |
|---|---|
| `PlanarPtychography` | The model: probe modes `(M, 1, N, N)`, one object slice, sub-pixel scan offsets and an optional detector mask |
| `solve_ptychography(..., solver="secant_gd")` | Minibatch gradient descent with one Newton step size per parameter group; no learning rate to tune |
| `PlanarConstraints` | Per-pass constraints: mode orthogonalisation, a radial support, an aperture-plane probe support and an absorption bound |
| `PassStopRule`, `PassStopCallback` | When to stop a solve, and which pass to return |
| `scatterem.diagnostics.metrics` | Probe gauges, aperture plane, OAM spectrum, mean OAM, OAM density matrix |

All of them are in `scatterem.reconstruction.ptychography.iterative.second_order`,
except the diagnostics. The loss is `normalized_amplitude_rmse_loss` from
`scatterem.nn.losses`: the amplitude RMSE divided by the mean measured
amplitude, so it has no unit.

## A first reconstruction

The example below makes a synthetic data set and reconstructs it. A vortex probe
(a disk aperture with the phase $e^{i l \theta}$) in two incoherent modes
illuminates a weak phase object. The script `examples/vortex_secant_gd.py` of
the repository is a longer version of it.

```python
import torch
from scatterem.reconstruction.ptychography.iterative.second_order import (
    PassStopCallback,
    PassStopRule,
    PlanarConstraints,
    PlanarPtychography,
    solve_ptychography,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRID, DK, STEP, SCAN = 48, 0.05, 0.8, 16   # detector px, 1/A per px, A, positions per side


def vortex(n, charge, radius):
    """Real-space probe of a disk aperture with the phase exp(i l theta)."""
    y = torch.arange(n, dtype=torch.float64) - n / 2
    yy, xx = torch.meshgrid(y, y, indexing="ij")
    aperture = (torch.hypot(yy, xx) < radius) * torch.exp(1j * charge * torch.atan2(yy, xx))
    probe = torch.fft.ifft2(torch.fft.ifftshift(aperture))
    return torch.fft.fftshift(probe).to(torch.complex64)


# The truth: 90 % of the power in l = 1, 10 % in l = 2.
true_probe = torch.stack([vortex(GRID, 1, 8), 0.35 * vortex(GRID, 2, 8)])[:, None].to(device)
blank = torch.ones(SCAN, SCAN, GRID, GRID, device=device)
truth = PlanarPtychography.from_scan(blank, STEP, DK, true_probe, rotation_deg=3.0)
gen = torch.Generator().manual_seed(0)
phase = torch.randn(1, 1, *truth.object.shape, generator=gen)
phase = torch.nn.functional.avg_pool2d(phase, 5, stride=1, padding=2)[0, 0]
with torch.no_grad():
    truth.object.copy_(torch.complex(1.5 * phase, 0.02 * phase.abs()).to(device))
    intensities = truth(torch.arange(SCAN * SCAN, device=device)) ** 2
intensities = intensities.reshape(SCAN, SCAN, GRID, GRID)   # what the detector records
```

The reconstruction has two stages. Stage 1 finds a coherent probe and a first
object with one mode. Stage 2 expands that probe into several modes and refines
everything again, with the modes orthogonalised after each pass.

```python
# Stage 1: one mode, from the aperture you expect.
seed = vortex(GRID, 1, 8)[None, None].to(device)
model = PlanarPtychography.from_scan(intensities, STEP, DK, seed, rotation_deg=3.0)
batches = model.batches(32)
solve_ptychography(model, batches, n_iter=20, solver="secant_gd", optimize_positions=True)

# Stage 2: three modes, 90 % of the power in mode 0.
model, ratio = model.expand_modes(3, mode0_frac=0.9)
model.constraints = PlanarConstraints(orthogonalize=True)
callback = PassStopCallback(model, PassStopRule(), return_last=True)
log = solve_ptychography(
    model,
    batches,
    n_iter=30,
    solver="secant_gd",
    probe_mode_groups=True,       # one step size per probe mode
    optimize_positions=True,
    callback=callback,
)
callback.restore_last()           # the state of the last accepted pass
print(f"{len(log.losses)} passes, loss {log.losses[-1]:.4f}")
```

`from_scan` takes the counts `(Ny, Nx, My, Mx)` with the bright-field disk in
the centre, the scan step in angstroms, the detector pixel in inverse
angstroms, and the seed probe in real space. It scales the probe to the
measured intensity and starts from a vacuum object. `expand_modes` keeps the
product of probe power and object transmission, because the data fix only that
product.

## Read the probe

The last mode of a mixed-state solve is free: it takes the part of the data
that the model cannot describe (detector noise, a background). Leave it out of
every probe metric. Remove the two gauges of the probe first: the data do not
fix the probe's position (translation) or a linear phase ramp in its aperture
plane.

```python
from scatterem.diagnostics.metrics import (
    adjacent_coherence,
    mean_oam,
    oam_density_matrix,
    oam_spectrum,
    remove_probe_gauges,
    to_aperture_plane,
)
from scatterem.reconstruction.ptychography.iterative.second_order.stop_rule import mode_purity

probe = model.probe.detach()                             # (3, 1, N, N)
modes = remove_probe_gauges(probe, drop_last=True)       # (2, N, N), gauges removed
print(f"purity {mode_purity(probe):.3f}")                # truth: 0.81; the free mode is left out

aperture = to_aperture_plane(modes)                      # centred aperture plane
ls, spectrum = oam_spectrum(aperture, l_max=8)
print(f"mean OAM {mean_oam(ls, spectrum):+.2f} hbar")    # truth: about +1.1

ls, rho = oam_density_matrix(aperture, l_max=8)
g, i0, j = adjacent_coherence(rho)
print(f"coherence of l = {int(ls[i0])} and l = {int(ls[j])}: {g:.2f}")   # truth: 0 (incoherent)
```

The OAM spectrum $P_l$ is the power in each charge $l$, about the intensity
centroid of the aperture plane. The density matrix $\rho_{ll'}$ has this
spectrum on its diagonal. Its off-diagonal elements show the coherence between
OAM channels: zero for an incoherent mixture of OAM eigenstates, one for a
coherent superposition with a common radial profile.

## Measured data

The synthetic example skips the steps that a real scan needs. The paper's
reconstructions used these settings, on 512 × 512 positions and native
192 × 192 detector pixels:

| Setting | Value in the paper | scatterem call |
|---|---|---|
| Data | flat-field corrected counts; bright-field disk centred; descan removed; detector axes point-inverted (k to -k) | your preprocessing, then `from_scan` |
| Model grid | 288 px far field, the 192 px detector in its centre | `from_scan(..., model_grid=288)` |
| Detector mask | measured pixels only; the needle shadow of the device has zero weight | `detector_mask=`, with `ray_band_mask` for the shadow |
| Seed | an aperture from a first estimate on every second scan position, 96 % of the power in mode 0 | `probe_stack_from_aperture(aperture, 192, pmodes, 0.96)` |
| Modes | 9: 8 physical and 1 free | `expand_modes(9, mode0_frac=0.96)`, or a 9-mode seed |
| Passes | end a: 40 passes with one mode at batch 128, then 150 passes with 9 modes at batch 512; end b: 150 passes with 9 modes at batch 512 | `model.batches(...)`, `n_iter=` |
| Probe support | aperture-plane field set to zero outside the measured detector area and in the needle shadow, power kept | `PlanarConstraints(aperture_support=measured_bbox_support(...))` |
| Other constraints | mode orthogonalisation; radial support; absorption bound 2.5 | `PlanarConstraints(orthogonalize=True, radial_support=(1.2, 8.0), absorption_width=2.5)` |
| Positions | refined, 0.5 px trust bound per pass | `optimize_positions=True` (default bound) |
| Result | state of the last accepted pass | `PassStopCallback(..., return_last=True)` |

With these settings, the public code gave the same purity and mean OAM as the
paper's production code, within the run-to-run spread of each, on the same
measured scan. The seed of the paper came from a first-estimate step that is
not part of scatterem. For a vortex, a flat-phase disk is a poor seed: on
scan 1074 of the paper, its one-mode solve reached only +12.3 ħ of the
+20 ħ. The notebook `notebooks/mems_vortex_scan1074.py` measures a seed
from the shifts of the virtual bright-field images
(`pairwise_relative_shifts`, `synchronize_shifts`) and integrates the
aperture phase from them. It runs the whole recipe on a crop of that scan
(see the [Example gallery](gallery.md)).

Two rules from the paper's controls with simulated data:

1. **The mean OAM is a measurement; purity recovery needs a strongly
   scattering object and a large scan.** On simulated states, the
   reconstruction returned the mean OAM within 2 %. The purity is recovered
   only from a strongly scattering object and a large scan (the paper uses
   512 × 512 positions). With a weaker object or a smaller scan, the purity
   comes out too high, and it changes with the batch size, the start state
   and the number of passes. Report the purity with the scan size and these
   settings.
2. **Sample the aperture plane finely before you sort a vortex.** For a large
   charge $l$, bilinear ring sampling on the reconstruction grid gives a mean
   OAM that is too small (23.4 to 23.8 for $l = 24$). Zero-pad the real-space
   modes by a factor of 4 before `to_aperture_plane`, so that the aperture
   plane is sampled 4 times finer. Scale the ring radii (`r_min`, `r_max`) by
   the same factor, and set `l_max` well above the largest charge (the paper
   uses 120). On measured data, the paper also removes the two-fold
   astigmatism common to all modes before the sort, and uses rings over the
   whole aperture (Section V E); the notebook shows both steps.

```python
up = 4
pad = (GRID * up - GRID) // 2
fine = to_aperture_plane(torch.nn.functional.pad(modes, (pad, pad, pad, pad)))
ls, spectrum = oam_spectrum(fine, l_max=12, r_max=up * (GRID / 2 - 1))
print(f"mean OAM, 4x finer aperture plane: {mean_oam(ls, spectrum):+.2f} hbar")
```

## Reading further

- The paper: S. You *et al.*, arXiv:2608.05749 (2026),
  [doi:10.48550/arXiv.2608.05749](https://doi.org/10.48550/arXiv.2608.05749)
- The solver's step sizes: Carlsson, Nikitin, Troedsson and Wendt, *The bilinear
  Hessian for large scale optimization*,
  [arXiv:2502.03070](https://arxiv.org/abs/2502.03070)
- [API reference](api.md#second-order-ptychography)
- [How to cite](cite.md)
