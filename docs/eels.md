# Simulate STEM-EELS

This page covers the forward problem: given a structure, produce the core-loss
EELS signal that a STEM records. There is no EELS *reconstruction* here.

Three algorithms compute the same physics at very different cost. Pick the one
that gives the output you need:

| You want | Algorithm | Entry point |
|---|---|---|
| A reference to check the others against | conventional multislice | `multislice_transition_potential` |
| An energy-filtered 4D-STEM cube on a real scan | PRISM | `PrismEelsSimulator.eels_cube` |
| A 2D elemental map at full resolution | BiP-PRISM | `PrismEelsSimulator.eels_map` |
| Momentum-resolved output (qEELS, PACBED) | BiP-PRISM | `PrismEelsSimulator.qeels` |

The cube keeps the whole diffraction pattern at every scan position. It is the
general output and the expensive one. The map integrates over an EELS collection
aperture. The same two-matrix path can also keep individual detector beams for
qEELS and PACBED; see [BiP-PRISM](bip-prism.md). If you want an elemental map,
do not simulate a cube and sum it.

`PrismEelsSimulator` is a thin driver. It builds the kernel inputs from a
structure and calls `prism_eels_image` (map, qEELS) or
`prism_transition_potential` (cube). It adds no physics of its own. The same
driver also runs the elastic simulation; see [Elastic PRISM](prism.md).

## The short version

`structure` is a `scatterem.physics.atom_set.AtomSet`; [A structure](#a-structure)
below builds one.

```python
from scatterem.simulation.eels import PrismEelsSimulator

sim = PrismEelsSimulator(
    structure,                     # scatterem.physics.atom_set.AtomSet
    eV=100e3,
    semiconvergence_angle=25.0,    # mrad
    pixels=(128, 128),
    scan=(16, 16),
    num_slices=16,
    device="cuda",
)

ti_map = sim.eels_map(
    (22, 2, 1),                    # Ti L: (Z, n, l)
    detector_mrad=25.0,
    partition=dict(n_radial=4),
    partition_s2=dict(n_radial=4),
)
ti_map.data                        # (16, 16) elemental map
cbed = sim.elastic_cbed()          # elastic 4D-STEM by PRISM
```

Without GPAW, this example uses analytic hydrogenic states and emits a
`HydrogenicFallbackWarning`: the numbers are then not quantitative. See
[Transition potentials, and the GPAW question](#transition-potentials-and-the-gpaw-question)
and [The BiP-PRISM paper and v0.4](#bip-prism-paper-v04). The sections below
go through each step.

## A structure

`AtomSet` is an immutable NumPy container. It holds the side lengths of a
rectangular box in Ångström, the fractional coordinates of the atoms and their
atomic numbers.

```python
import numpy as np
from scatterem.physics.atom_set import AtomSet

a = 3.905
sto = AtomSet(
    np.array([a, a, a]),        # box sides (y, x, depth), Angstrom
    np.array([                  # fractional coordinates (y, x, depth)
        [0.0, 0.0, 0.0],        # Sr
        [0.5, 0.5, 0.5],        # Ti
        [0.5, 0.5, 0.0],        # O
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ]),
    np.array([38, 22, 8, 8, 8]),  # atomic numbers
    msd=None,                   # <u^2> per atom; None gives every atom 0.01 A^2
).replicated((4, 4, 8))         # 15.6 x 15.6 x 31.2 A, 640 atoms
```

Coordinate 0 runs along grid axis 0 (y), coordinate 1 along grid axis 1 (x),
and coordinate 2 along the beam. `read_atom_set` (same module) reads
Kirkland/Prismatic `.xyz`, POSCAR-style `.p1` (VESTA), muSTEM `.xtl`, `.cif`
and extended `.xyz` (`.extxyz`) files. A Kirkland `.xyz` file holds Cartesian
positions in Å and an rms thermal displacement; pass
`displacement_kind="mean_square"` if the file holds $\langle u^2 \rangle$.
The `.cif` and extended `.xyz` readers need the optional `ase` package. If you give lattice vectors as `(3, 3)` rows, `AtomSet` reduces the
cell to a rectangular box, because the multislice handles only that box.

:::{admonition} `msd` is $\langle u^2 \rangle$, one-dimensional
:class: note

`AtomSet.msd` holds the *one-dimensional* mean-square thermal
displacement in Å². It is not the Debye-Waller $B$ and not the three-dimensional
total. $B = 8\pi^2 \langle u^2 \rangle$. If you use the wrong convention, the
frozen-phonon smearing is wrong by $\sqrt{3}$.
:::

## The simulator

```python
import torch
from scatterem.simulation.eels import PrismEelsSimulator

sim = PrismEelsSimulator(
    sto,
    eV=100e3,                    # probe energy
    semiconvergence_angle=25.0,  # mrad
    pixels=(128, 128),           # real-space grid across the cell
    scan=(16, 16),               # scan positions, a raster over the cell
    num_slices=16,
    interpolation_factor=2,      # PRISM factor f
    device="cuda",
)
```

The grid is the whole cell, and the scan is a regular raster over the cell. The
probe has zero defocus by default; pass `defocus=` (Å) to change it. Pass
`dtype=torch.complex128` for double precision. For thermal averaging, pass
`num_frozen_phonons=` with `displacements=True` and a `seed`.

:::{admonition} `semiconvergence_angle` is in **milliradians** here
:class: important

`Dataset4DStem.from_array` on the reconstruction side takes radians. The
simulator takes mrad, and so does `detector_mrad` below. A factor of 1000 gives
a probe that fills the whole grid or a single pixel, so the error is easy to
see. But it costs a run.
:::

## A full-resolution elemental map

```python
ti_map = sim.eels_map(
    (22, 2, 1),                    # Ti L: (Z, n, l) of the ionized subshell
    epsilon=10.0,                  # eV above the ionization threshold
    detector_mrad=25.0,            # EELS collection semi-angle
    inelastic_crop=4,
    partition=dict(n_radial=4),    # partition S1
    partition_s2=dict(n_radial=4), # ... and S2: this is the "Bi" in BiP
)
ti_map.data            # (16, 16), real, non-negative
ti_map.energy_losses   # the per-transition losses summed into the map, eV
```

`edge` names the *bound* orbital that is ionized, as `(Z, n, l)`. Ti L is
`(22, 2, 1)`, O K is `(8, 1, 0)`, Fe L is `(26, 2, 1)`. Every atom of that `Z` in
the structure ionizes; there is no site selection. Each site ionizes with the
weight that its atom has in the elastic potential. If the structure has a
partial occupancy, that weight is the `site_fraction` of the site, else it is 1.
So a site with half occupancy gives half the signal of a full site in the same
specimen, and a site with `site_fraction=0` gives no signal. The kernel runs
once for each different nonzero `site_fraction` of the ionized element, so a
structure with many different fractions costs more.

`epsilon` is the energy of the ejected electron *above* the threshold. So it
selects a point along the edge, not an absolute energy loss. The onset is the
binding energy, and `energy_losses` reports the resulting losses.

If you remove both `partition` arguments, you get the exact dual matrix. Use it
as the oracle on a small cell. At full resolution it does not fit on one GPU.

The other keyword arguments of `eels_map` go to `prism_eels_image`:
`inelastic_crop`, `prism_mask`, `mag_preserve`, `focal_backprop`, `contract`,
`scan_chunk` and `reconstruction_workspace`. [BiP-PRISM](bip-prism.md) explains
what they do.

## A cube

```python
cube = sim.eels_cube((22, 2, 1), epsilon=10.0)
cube.data               # (16, 16, 128, 128), real, non-negative
cube.detector_sampling  # detector (dky, dkx) in 1/A
```

The diffraction patterns are corner-origin, like everything else in reciprocal
space here. Apply `torch.fft.fftshift` before you plot them.

For the cube, `partition=dict(n_radial=N)` partitions $\mathcal{S}_1$ only. The
exit leg keeps the full diffraction pattern.

## Momentum-resolved output and PACBED

```python
q = sim.qeels((22, 2, 1), epsilon=10.0, detector_mrad=25.0, inelastic_crop=4)
q.data                  # (16, 16, n_detector_beams)
q.q_pixels              # (n_detector_beams, 2), (qy, qx) in grid pixels
q_inv_A = q.q_pixels / torch.tensor(sim.gridsize, device=q.q_pixels.device)
pacbed = q.data.mean(dim=(0, 1))
```

`axis="both"` (the default) keeps each detector beam. `axis=0` or `axis=1`
returns a one-axis momentum projection, and `q_pixels` is then a sorted
one-dimensional coordinate array. If you sum the last axis of any resolved output,
you get the map with the same settings. Prefer `interpolation_factor=1` when you
keep detector coordinates, because a larger factor subsamples the detector.

## An energy window, and several edges

One `epsilon` is one point on the edge. To integrate an energy-filtered window,
sweep `epsilon` and integrate with the trapezoid rule:

```python
eps = np.linspace(5.0, 45.0, 8)
maps = torch.stack(
    [sim.eels_map((22, 2, 1), epsilon=e, detector_mrad=25.0).data for e in eps]
)
window = torch.trapezoid(maps, torch.as_tensor(eps, device=maps.device, dtype=maps.dtype), dim=0)
```

The continuum states are energy-normalised, so the trapezoid integral over
`epsilon` is the signal collected in that loss window. `epsilon` must be greater
than zero: the energy-normalised continuum diverges at the threshold.

For several edges, call the driver once per edge, for example `(22, 2, 1)` and
`(8, 1, 0)`. Each call rebuilds the transmission functions. If that cost
matters, build the inputs once with `sim.transmissions()`, `sim.probe_q()`,
`sim.scan_pixels()`, `sim.sites(Z)` and `sim.transition_potentials(edge,
epsilon)`, and call the kernels directly (see below).

## Transition potentials, and the GPAW question

The ionization operator $H_{n0}$ is built from atomic radial wavefunctions. Their
source decides whether your numbers are quantitative.

```python
from scatterem.simulation.eels import gpaw_available
gpaw_available()
```

The `wavefunctions` option selects the source. Pass it through
`transitions=dict(...)` on any driver method:

- `"auto"` (the default) uses GPAW when it is installed, as v0.3 did. Without
  GPAW, it falls back to `"hydrogenic"` and emits a `HydrogenicFallbackWarning`.
- `"gpaw"` takes bound states from an all-electron atomic DFT calculation, and
  continuum states from Numerov integration of the radial Schrödinger equation in
  that all-electron potential. This is the physically accurate path. The
  BiP-PRISM paper uses it wherever the absolute edge physics matters. A missing
  GPAW install is an error.
- `"hydrogenic"` uses a hydrogenic bound state and a free spherical-Bessel
  continuum. It runs the entire pipeline, and the test suite selects it
  explicitly. It is **not quantitatively accurate**: the selection rules and the
  shape of the pipeline are right, the cross-sections are not.

GPAW is an optional extra, because it is GPL-licensed and a large install. The
`eels` extra installs it. pip builds GPAW from source, and the build needs a
C++ compiler (see [Installation](install.md)). The line below pins
scatterem 0.4.0, the release that gives the numbers on this page:

```bash
CC=g++ pip install "scatterem[eels]==0.4.0" gpaw-data
```

:::{admonition} Without GPAW, the numbers change
:class: warning

If GPAW is not installed, the default falls back to the hydrogenic states, and
the EELS calls warn:

```text
HydrogenicFallbackWarning: wavefunctions="auto": GPAW is not installed, so the
analytic hydrogenic bound and free continuum states are used. These EELS
numbers are not quantitative and differ from the GPAW results. Install the
eels extra to use GPAW. v0.4 uses a new absolute first-Born scale
(Angstrom^2/eV). After a best scalar fit, the v0.4 map shapes differ from v0.3
by 18-61 % with these hydrogenic states. With GPAW they differ by 0.03-0.6 %
for the K, L23 and M45 edges tested and by 1.4 % for Fe L1. Pass
wavefunctions="hydrogenic" to choose the analytic states explicitly.
```

If you run the examples on this page without GPAW, you see this warning. For
quantitative numbers, install `scatterem[eels]`. To test the pipeline with the
analytic states and no warning, pass
`transitions=dict(wavefunctions="hydrogenic")`. To fail when GPAW is missing,
pass `transitions=dict(wavefunctions="gpaw")`.
:::

(bip-prism-paper-v04)=
### The BiP-PRISM paper and v0.4

v0.4 builds the transition potentials from a rewritten first-Born model. The
forward model (`prism_eels_image` and its partitioned forms) is the same as in
v0.3: given the same transition potentials, it gives the same maps. So the
numbers of the paper move through the transition potentials and, when you
build the specimen with v0.4, through the new physics base layer (structure,
projected potentials, transmission functions and probe):

- **Absolute scale.** The transition potentials now carry the first-Born
  cross-section. For a unit plane wave, the summed intensity of the channels of
  one subshell is $d\sigma/dE$ in Å²/eV. The scale includes the $1/v^2$ factor
  of the interaction constant and the occupancy $N/(2l+1) = 2$ of each orbital
  of a closed subshell, and the flux factor $k_0/k_n$. v0.3 had no absolute
  scale. So no single factor converts v0.3 values to v0.4 values: the ratio
  depends on the edge and on the beam energy (see
  [Migrating from v0.3](migrating.md)).
- **GPAW states.** We compared v0.4 with v0.3 for each GPAW edge of the paper,
  at the beam energies of the paper (100 and 300 kV), 10 eV above threshold,
  after a best scalar fit. The map shapes change by 0.03-0.06 % for O K, Mg K and Al K, and by
  0.16-0.37 % for Ti L23, Fe L23, Cu L23, Sr L23 and La M45.
  Fe L1 (spinel PACBED, 300 kV) changes by 1.4 %. Its initial state is an s
  orbital, and the orthogonalised continuum changes its $l' = l = 0$ channel
  most. At 200 kV, over 256 kernel settings, C K, O K, Si L23 and Ti L23 change
  by at most 0.57 %. We did not re-run the GPAW figures of the paper.
- **Hydrogenic states.** The map shapes change by 18-61 %. The v0.3 hydrogenic
  maps were dominated by a spurious monopole term, which the orthogonalised
  continuum of v0.4 removes. The paper uses hydrogenic states in three studies.
  We re-ran them with v0.4 in two ways. The first re-run kept the paper's
  specimen setup (structure, transmissions, probe and scan from the paper's
  code) and changed only the transition potentials. The second re-run built
  the whole specimen setup with the v0.4 physics layer: `AtomSet` for the
  structure, `PrismEelsSimulator.transmissions` (`potential_stack` and
  `thin_slice_transmission`) for the potentials and transmissions, and
  `PrismEelsSimulator.probe_q` for the probe. The scan positions follow the
  paper's geometry on the v0.4 grid. The conclusions of all three studies
  still hold in both re-runs. The first re-run gives these changes:
  - Detector-phase ablation (Ti L, LAO/STO line scan): the relative errors of
    the four arms go from 47.48 / 12.86 / 9.36 / 2.12 % to
    48.09 / 13.59 / 9.00 / 2.02 %. The order of the arms and the
    conclusion do not change.
  - Oxygen-row qEELS test: the sum over the resolved momentum still equals the
    map to 4.3e-16 (v0.3: 2.7e-16), and equivalent columns still agree. The
    map and spectrum image shapes change by 43-60 %, and the energy-momentum
    stack by 41 %.
  - Map-scaling study (Ti L, SrTiO₃): the timings do not depend on the
    transition-potential values. The map shapes change by at most 0.9 %. The
    partitioning errors above 1 % change by at most 3 % of their value (for
    example 15.3 % to 15.0 % at 30 nm). Time stays linear in thickness. The
    partitioned peak memory of the paper's memory panel holds: 0.545 GB at a
    384² grid and 3.60 GB at 1024² (paper code on the same GPU: 0.549 and
    3.76 GB). Against v0.3, the time per call changes because of the matmul
    precision (see [the cost of a call](#the-cost-of-a-call-v03-and-v04)).
    Against the paper's pre-release code, the time per call changes by a
    factor of 0.3-1.6 at the same precision (`"medium"`), because of kernel
    changes made before v0.3. v0.4 is 1.6x slower at a 32×32 scan and 3.4x
    faster at a 1024² grid. A linear fit over the 8×8, 32×32 and 64×64 scans
    gives 10.1 ms (paper code) and 11.4 ms (v0.4) per scan position. The
    exact PRISM call takes 2.9x as long as the partitioned call in v0.4, and
    1.8-2.2x with the paper's kernels on the same GPU.
  - With the v0.4 specimen setup (second re-run, v0.4 transition potentials,
    RTX A6000), the numbers change again:
    - Detector-phase ablation: the four arms use the paper's kernels on the
      v0.4 setup, because only those kernels have the uncorrected first arm.
      The relative errors of the four arms are 48.06 / 13.31 / 9.82 / 2.21 %
      (paper setup: 48.09 / 13.59 / 9.00 / 2.02 %). The sign fix lowers the
      error 3.6x (published paper: 3.7x), the de-chirp 1.36x (published paper:
      1.37x), and 207 parents still give about 2 %. The v0.4 kernels give the
      same three corrected arms (13.30 / 9.80 / 2.21 %).
    - Oxygen-row qEELS test: the momentum sum equals the map to 4.1e-16, and
      equivalent columns agree to 3.9e-8 of the peak. The map changes shape
      by 1.7 % against the paper setup.
    - Map-scaling study: the time exponents in thickness are 1.90
      (conventional), 1.00 (exact) and 1.00 (partitioned). The partitioned
      peak memory is 0.551 GB at a 384² grid, 53x below the exact call
      (29.0 GB), and 3.60 GB at 1024². Per scan position, the partitioned
      call costs 27 ms and conventional multislice 2.2 s on this GPU. The
      timings agree with the paper setup on the same GPU to 6 %. The
      partitioning errors change by up to 10 % of their value (for example
      15.0 % to 16.5 % at 30 nm, and 11.2 % to 10.5 % at a 128² grid). They
      still grow with thickness and fall with the grid size. The maps change
      shape by 0.8-12 % against the paper setup. Most of this comes from the
      transmissions (7-8 %), because the v0.4 potentials use a B-spline
      deposit with a Fourier correction instead of a pixel average. The
      soft-edged v0.4 probe adds up to 6 %.
- **The exact published numbers came from the paper's pre-release code.** That
  code uses the v0.3 transition potentials. With the same transition
  potentials, its kernels agree with the v0.4 kernels to 4.5e-4 or better
  (relative L2, complex64). It also keeps the uncorrected detector of the
  first ablation arm, which v0.3 and v0.4 do not have, so the first arm above
  (48.09 %) comes from the paper's kernels with the v0.4 transition
  potentials. With v0.4 you get the numbers above, not the printed ones.

The code caches computed wavefunctions on disk, keyed by their physical
parameters, so a re-run does not repeat the atomic calculations. Set
`SCATTEREM_EELS_CACHE` to put the cache somewhere with room.

Two more `transitions` options are worth knowing. `order` sets how many
angular-momentum channels are summed. The default 1 keeps the monopole and both
dipole channels, as the published treatments do. `potential_scale` multiplies
the atomic potential in which the GPAW continuum states are computed. It does
not scale the centrifugal term. The default 1.0 is the unscaled Kohn-Sham
potential of the neutral atom, and 0 gives free continuum states. A change
moves mostly the $l' = l$ monopole s-wave.

## Calling the kernels directly

The driver methods return the inputs they use. You can pass them to the kernels
yourself, for example to reuse one set of transmission functions for several
edges:

```python
from scatterem.simulation.eels import prism_eels_image

T = sim.transmissions()
tp = sim.transition_potentials((22, 2, 1), epsilon=10.0)

ti = prism_eels_image(
    sim.probe_q(), T, tp, sim.sites(22), sim.scan_pixels(),
    wavelength=sim.wavelength,
    gridsize=sim.gridsize,
    slice_distance=sim.slice_distance,
    detector_mrad=25.0,
    interpolation_factor=2,
    inelastic_crop=4,
    scan_shape=sim.scan,
    partition=dict(n_radial=4),
    partition_s2=dict(n_radial=4),
)
```

## What it costs, and the levers

Four options matter most. In rough order of how much they save:

**`partition` / `partition_s2`** (`dict(n_radial=N)`) build each scattering matrix
on a sparse set of parent beams instead of every beam. This is the memory lever.
On a large map it saves an order of magnitude or more, but only when the matrices
fill the card. On a small problem it saves nothing and costs a little time. It is
also approximate, and the detector side is the demanding one; see
[accuracy](bip-prism.md#accuracy-and-the-one-caveat-worth-knowing).

**`interpolation_factor`** ($f$) subsamples the aperture beams by $f^2$. It is
valid only if the probe fits inside a $1/f$ fraction of the cell around the atom.
If it does not, neighbouring periodic copies of the probe leak in. For the 4D-STEM
cube it is the *only* lever on the exit leg, because partitioning there changes
only $\mathcal{S}_1$. It also changes the absolute scale; see below.

**`inelastic_crop`** restricts the per-atom contraction to a
`(Ny, Nx) // inelastic_crop` window. $H_{n0}$ has sub-ångström support, so a
generous crop costs nothing in accuracy and saves the largest intermediate of
the contraction. `None` uses the full grid and is exact.

**`num_frozen_phonons`** with `displacements=True` averages over thermal
configurations. The cost is linear in the count. One configuration is enough for
a smoke test and wrong for anything quantitative. In each configuration, one
thermal draw moves the atoms of the elastic potential and the ionized atoms, so
each transition potential sits on its displaced atom. This is also true with
`seed=None`, where each configuration draws a new seed.

**`reconstruction_workspace`** (bytes) bounds the extra memory that a
partitioned call may use to reconstruct the full grid once per slice instead of
once per atom window. `None` (default) allows 2 GiB, and `0` turns the reuse off,
which is the behaviour of the BiP-PRISM paper and of v0.3. The reuse applies only
when the crop windows of one slice together cover the grid *and* the full-grid
reconstruction fits the budget. At the paper's map-scaling settings
(`inelastic_crop=16`, 50 mrad detector) it never applies, so both values give
the same time and peak memory. The output does not depend on the value beyond
rounding.

Two things add cost. The scattering matrix is built once, but each scan
position still needs its own contraction. With `prism_mask=False`, a 64×64 scan
takes about four times as long as a 32×32 scan (see the table below). A larger
detector angle increases $n_{\text{det}}$ and so the $\mathcal{S}_2$ build.

(the-cost-of-a-call-v03-and-v04)=
### The cost of a call: v0.3 and v0.4

The PRISM-EELS kernels of v0.4 compute the same forward model as v0.3, but a
float32 call can take longer, for one reason: `import scatterem` no longer sets
`torch.set_float32_matmul_precision("medium")`. So the complex64 contractions
run at full float32 precision instead of TF32. To get the v0.3 speed and the
v0.3 rounding, call `scatterem.configure(matmul_precision="medium")` before the
simulation. The TF32 rounding moves a complex64 map by about 1.6e-4 (relative
L2). complex128 calls do not change. The table gives the measured times on one
RTX PRO 6000, in seconds per call, with the paper's map-scaling inputs (complex64, `contract="probe"`,
`prism_mask=False`, partitioned `n_radial=4`):

| Configuration | v0.3 | v0.4 | v0.4 with `"medium"` |
|---|---|---|---|
| 128², 8×8 scan | 1.20 | 1.35 | 1.08 |
| 128², 32×32 scan | 12.3 | 17.9 | 12.2 |
| 128², 64×64 scan | 47.4 | 69.9 | 47.2 |
| 384², 8×8 scan | 2.52 | 3.69 | 2.27 |
| 1024², 8×8 scan | 16.5 | 24.6 | 14.3 |

With the same precision, v0.4 is as fast as v0.3 or faster.

## Checking your setup

The approximations compose. Check them one at a time, on a cell small enough
that the exact answer is affordable. At $f = 1$, PRISM must reproduce the
conventional multislice to rounding:

```python
from scatterem.physics.propagation import fourier_shift_stack
from scatterem.simulation.eels import multislice_transition_potential

ref = PrismEelsSimulator(
    sto, eV=100e3, semiconvergence_angle=25.0, pixels=(128, 128), scan=(8, 8),
    num_slices=16, interpolation_factor=1, dtype=torch.complex128, device="cuda",
)
edge, epsilon = (22, 2, 1), 10.0

probes = fourier_shift_stack(torch.fft.ifft2(ref.probe_q()), ref.scan_pixels())
exact = multislice_transition_potential(
    probes, ref.transmissions(), ref.transition_potentials(edge, epsilon), ref.sites(22),
    wavelength=ref.wavelength, gridsize=ref.gridsize, slice_distance=ref.slice_distance,
)
prism = ref.eels_cube(edge, epsilon=epsilon).data.reshape(exact.shape)

rel = float((prism - exact).norm() / exact.norm())
```

A large `rel` here means that something is wrong with the setup, not with the
method.

:::{admonition} `interpolation_factor` changes the absolute scale of the cube
:class: important

A larger $f$ keeps the pattern but scales the intensity of the cube by roughly
$f^2$. So a raw relative error against an $f = 1$ reference is dominated by this
factor and tells you almost nothing. **Hold $f$ fixed when you compare
anything.** Take absolute cross-sections at $f = 1$, or calibrate the scale
against it.
:::

Partitioning is a separate approximation. Compare `partition=None` with your
chosen `n_radial` at a fixed $f$. Treat the pattern correlation and the total
signal as two different questions. For the *map*, the detector-side partition is
much more demanding than the probe side. [BiP-PRISM](bip-prism.md#accuracy-and-the-one-caveat-worth-knowing)
works through that case.

## Reading further

- [BiP-PRISM](bip-prism.md): what the two scattering matrices are, why both can
  be partitioned, and where the approximation breaks down
- [Elastic PRISM](prism.md): the elastic 4D-STEM simulation from the same driver
- [API reference](api.md): every symbol above, generated from the source
- The paper: P. Pelz, *J. Phys. Mater.* (2026),
  [doi:10.1088/2515-7639/aebdd0](https://doi.org/10.1088/2515-7639/aebdd0)
