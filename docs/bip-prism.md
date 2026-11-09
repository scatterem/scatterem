# BiP-PRISM

## What makes core-loss EELS expensive

A core-loss STEM-EELS simulation is a double-channelling problem. The probe has to
be propagated dynamically *to* an atom that ionizes, the ionization event has to be
applied there, and the inelastically scattered wave has to be propagated
dynamically *from* that atom to the detector. Both legs are multislice.

Written out, the conventional transition-potential multislice does this for every
combination of scan position, ionized atom and transition channel. With $P$ scan
positions, $N_a$ atoms of the ionized element and $N_t$ channels, that is
$P N_a N_t$ exit-wave propagations. The scan is the painful factor: it is the one
that grows when you ask for the atomic-resolution map you actually wanted.

## The scattering matrix removes the scan

PRISM's idea is that the entrance leg does not depend on where the probe is. A
converged probe is a sum of plane waves over the aperture, and multislice is
linear, so propagating each of the $B$ aperture beams once gives a basis
$\mathcal{S}_1$ from which the probe at *any* scan position is recovered as a
linear combination — the coefficients are the aperture's phase ramps. One
propagation of $B$ columns replaces $P$ propagations of one probe.

The exit leg yields to the same argument, and this is the step that makes
detector-integrated maps cheap. Everything after the transition potential is
linear too, so

$$
\psi_{\text{exit}}(\boldsymbol{\rho}) \;=\;
\mathcal{S}_2\Big[\,H_{n0}\sum_b c_b(\boldsymbol{\rho})\,\mathcal{S}_{1,b}\Big]
\;=\;
\sum_b c_b(\boldsymbol{\rho})\;\mathcal{S}_2\big[\,H_{n0}\,\mathcal{S}_{1,b}\big] .
$$

The expensive operator acts on the $B$ columns, not on the $P$ probes, and the
scan is synthesised afterwards by the same coefficients that formed it. For a
detector-integrated map $\mathcal{S}_2$ can be built once per slice as a
*transpose* multislice of the detector-accepted output beams — by reciprocity,
running the detector backwards is the same calculation as running the source
forwards — and then reused across every atom in that slice. The per-scan exit
propagation disappears entirely. This is the dual-matrix algorithm of Brown,
Ciston & Ophus.

## What is left is memory

Having traded time for structure, the cost reappears as storage. $\mathcal{S}_1$
is $B$ complex arrays on the full $N \times N$ grid; $\mathcal{S}_2$ is $n_{\text{det}}$
of them, and the detector is usually the wider aperture of the two. Both are
resident. For the 23 196-atom FePt nanoparticle of the
[BiP-PRISM paper](https://doi.org/10.1088/2515-7639/aebdd0) (936² grid,
PRISM factor 4), the exact dual-matrix calculation needs 63.18 GB of GPU
memory. That is more than any consumer card, and more than most cluster cards.

## Partitioning both matrices

Beam partitioning attacks exactly this. Neighbouring beams in the aperture differ
only by a small tilt, so their columns, once de-tilted, are locally similar. Build
the matrix on a sparse set of $B_p \ll B$ *parent* beams, and reconstruct the
column you need for any beam by natural-neighbour interpolation of the parents —
de-tilt, combine with the interpolation weights, re-tilt.

The reconstruction is only ever needed **locally**, and that is the point. The
ionization operator $H_{n0}(\mathbf{r} - \boldsymbol{\tau})$ has sub-ångström
support around the atom at $\boldsymbol{\tau}$, so it *is* the crop: the columns
only have to be right inside a small window, never across the whole grid.

**BiP-PRISM** — bi-partitioned PRISM — applies this to both matrices at once:
$\mathcal{S}_1$ on parent aperture beams, $\mathcal{S}_2$ on parent detector
beams, each reconstructed on the crop window around the ionized atom immediately
before the contraction

$$
\big[\mathcal{S}H_{n0}\big]_{db}
\;=\;
\sum_{\mathbf{r}\,\in\,\Omega_\tau}
\mathcal{S}_{2,d}(\mathbf{r})\,H_{n0}(\mathbf{r}-\boldsymbol{\tau})\,\mathcal{S}_{1,b}(\mathbf{r}) .
$$

Both reconstructions resolve *before* the contraction, on the window, and the
window is small — so the resident matrices shrink from $B$ and $n_{\text{det}}$
columns to $B_p$ each, while the arithmetic that consumes them is unchanged.

:::{admonition} Resolved detector output
:class: note

`prism_transition_potential` retains the full exit diffraction pattern with
ordinary multislice on the exit leg, so its `partition` option applies to
$\mathcal{S}_1$ alone. `prism_eels_image` reconstructs both matrices and can
also retain detector coordinates: use `qeels_axis=0` or `1` for a momentum
projection, or `qeels_axis="both"` for individual detector beams.
`PrismEelsSimulator.qeels` calls it this way. Averaging
these resolved patterns over probe positions gives core-loss-filtered PACBED.
Summing the retained detector intensities recovers the elemental map.
:::

## Accuracy, and the one caveat worth knowing

Partitioning approximates the scattering columns on the inelastic crop window.
For a double-partitioned calculation, both the probe and detector reconstruction
errors matter. Increase `partition` and `partition_s2` parent counts independently
and compare with an unpartitioned calculation at the same grid, PRISM factor,
energy window and detector aperture. This isolates partitioning error;
convergence of the grid, slice thickness and crop is a separate check.

Since version 0.3, the code uses the corrected carrier signs of the detector
reconstruction. Detector
columns start from conjugate plane waves, so their de-tilt and re-tilt signs
are opposite to those of probe columns. It also removes each parent's known
quadratic vacuum propagation phase (detector de-chirp, on by default) before interpolation and restores the target
beam's phase afterwards. These scalar factors work directly on the crop window.
With all beams selected as parents, reconstruction recovers the unpartitioned
columns; in vacuum, the corrected detector interpolation is exact even with
sparse parents, up to floating-point error.

`mag_preserve=True` additionally uses the interpolated magnitudes and the phase
of the complex natural-neighbour sum on crop windows. This is a heuristic for
limiting amplitude loss, not a guarantee of absolute cross-section accuracy.
Set `mag_preserve=False` for plain complex interpolation. For a detector-phase
ablation, `partition_s2=dict(n_radial=4, dechirp=False)` keeps the corrected
carrier signs but disables scalar vacuum-phase removal.

## Focal referencing

For `prism_eels_image` and `PrismEelsSimulator.eels_map`, `focal_backprop=None`
is the default: interpolate at the current ionization
plane. This differs from zero incident-probe defocus, which sets the probe's
entrance phase. Optional `focal_backprop="centroid"` references the probe parents
by half the traversed depth, rounded to a slice, then propagates the interpolated
columns forward to the ionization plane. A numeric value specifies the fraction
of the traversed depth. Windowed forward propagation uses padding; its
truncation must be converged separately. When every beam is a parent, the
reference-plane round trip is skipped to preserve the exact cropped columns.
No reference plane is assumed to minimize interpolation error universally.

The corrected large-map examples use zero incident defocus with focal
back-propagation disabled. The single-column regime survey uses the explicit
centroid setting. Scalar detector dechirping is independent of this spatial
reference-plane operation.

The elastic operator `PrismScanOp` has its own default,
`focal_backprop="centroid"`. There, "centroid" is the scattering-weighted
centroid of the specimen, not half the traversed depth. See
[Elastic PRISM](prism.md).

## Memory and runtime

Partitioning reduces resident scattering-matrix storage by replacing full beam
sets with parent sets. The per-atom coupling and scan synthesis still cost time;
reconstruction overhead can outweigh build savings on small problems. Measure
both runtime and peak memory for the specimen and output mode of interest.

`contract="columns"` forms the beam-to-beam coupling before scan synthesis;
`contract="probe"` forms the local probe first. Both compute the same observable
within floating-point precision. The default, `contract="auto"`, picks the
order for each atom from the array shapes. The test suite checks that
`"auto"` agrees with both forced orders to a relative tolerance of 1e-11 in
float64. v0.3 used `"columns"`; pass it to force that order. `scan_chunk` bounds the number of scan positions
processed together in the resolved-output and probe-contraction paths; it does
not change the output definition.

`reconstruction_workspace` (bytes) bounds the extra memory that a partitioned
call may use to reconstruct one full-grid matrix per slice and crop it for
every atom, instead of reconstructing each atom's window. It applies only when
the windows of one slice cover the grid and the full-grid reconstruction fits
the budget. `None` (default) allows 2 GiB. `0` turns the reuse off: this is the
windowed path of the paper and of v0.3, with the smallest peak memory. The
output does not depend on the value beyond rounding. At the paper's
map-scaling settings the reuse never applies. There, both values give 0.545 GB
peak memory and 3.7 s at a 384² grid, and 3.60 GB and 24.6 s at 1024²
(complex64, RTX PRO 6000, full float32 matmul precision).

The float32 speed depends on the matmul precision. v0.3 set TF32 (`"medium"`)
at import; v0.4 does not. `scatterem.configure(matmul_precision="medium")`
restores the v0.3 speed; see
[the cost of a call](eels.md#the-cost-of-a-call-v03-and-v04).

## Reading further

- The paper: P. Pelz, *The bi-partitioned PRISM (BiP-PRISM) algorithm for fast
  and scalable core-loss STEM-EELS simulations*, J. Phys. Mater. (2026),
  [doi:10.1088/2515-7639/aebdd0](https://doi.org/10.1088/2515-7639/aebdd0)
- Transition-potential multislice for EELS: Dwyer, *Ultramicroscopy* **104**, 141 (2005),
  [doi:10.1016/j.ultramic.2005.03.005](https://doi.org/10.1016/j.ultramic.2005.03.005)
- The dual scattering matrix: Brown, Ciston & Ophus, *Phys. Rev. Research* **1**,
  033186 (2019),
  [doi:10.1103/PhysRevResearch.1.033186](https://doi.org/10.1103/PhysRevResearch.1.033186)
- PRISM: Ophus, *Adv. Struct. Chem. Imaging* **3**, 13 (2017),
  [doi:10.1186/s40679-017-0046-1](https://doi.org/10.1186/s40679-017-0046-1)
- Beam partitioning: Pelz *et al.*, *Microsc. Microanal.* **27**, 835 (2021),
  [doi:10.1017/S1431927621012083](https://doi.org/10.1017/S1431927621012083)
- Frozen phonons and specimen setup: Kirkland, *Advanced Computing in Electron
  Microscopy*, 2nd ed. (2010)
