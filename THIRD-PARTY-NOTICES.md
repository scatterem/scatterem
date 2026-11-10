# Third-party notices

scatterem is licensed under the Apache License 2.0 (see `LICENSE`). Parts of
it keep names and layouts from the third-party projects below, one file derives
from torchvision, and one data table comes from a journal article. This file
carries their notices.

## quantem (acknowledgment)

The dataset container and display API of scatterem follow quantem
(https://github.com/electronmicroscopy/quantem, MIT). No quantem code text
remains; signatures and trivial accessors that match quantem's interface are
listed below.

The line numbers are those of the v0.4.1 files. Each item is a run of 5 or
more identical lines (comments and blank lines not counted). The release gate
recomputes these runs and fails when the list is wrong.

- `scatterem/vis/normalization.py` 90-100: the fields and defaults of
  `NormalizationConfig` (its constructor parameters).
- `scatterem/vis/visualization_utils.py` 243-248: the signature of
  `add_cbar_to_ax`.
- `scatterem/data/datasets/base.py` 88-93: the `__init__` signature of
  `Dataset`; 115-119: the signature of `Dataset.from_array`; 175-182,
  211-218: the `name` and `signal_units` properties (a getter and a setter
  that converts to `str`).
- `scatterem/data/datasets/dataset_4dstem.py` 168-172: the `__init__`
  signature of `Dataset4DStem`; 743-747: the signature of
  `Dataset4DStem.from_array`; 965-972: the `origin` property (a getter and a
  setter that calls `validate_ndinfo`).

## torchvision (BSD-3-Clause)

torchvision (https://github.com/pytorch/vision) is licensed under the BSD
3-Clause License. `scatterem/data/public/utils.py` derives from
`torchvision/datasets/utils.py`: the URL retrieval, the MD5 checks, the Google
Drive download, and the archive extraction and decompression helpers keep
torchvision's function names, structure and docstring text. The file header
names this origin. The torchvision licence follows.

```text
BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## py_multislice (names only)

py_multislice by Hamish G. Brown (https://github.com/HamishGBrown/py_multislice)
ships no licence. The physics base layer of scatterem
(`scatterem.physics.periodic_sampling`, `atom_amplitudes`, `atom_set`,
`beam_partition`, `potential_stack` and the modules it uses, `thin_slice`)
is a clean-room rewrite.
Two teams did each rewrite. A specification team that knew the former code
wrote a behavioural contract. A separate blind team wrote the code from that
contract and the published literature, in a sandbox that held no
py_multislice code and none of the replaced modules. The sandboxes also held
some unrelated scatterem helper modules (the physical constants, the Fourier
helpers, a calibration helper and plotting modules), in their v0.3 form;
v0.4 replaces the constants and the Fourier helpers too (see the abTEM and
py4DSTEM sections). The acceptance tests
compared the new code with recorded outputs of the former code. After the
blind run, other agents, who were not blind, renamed the names that the
provenance audits flagged. For this, they compared the names with
py_multislice, abTEM, prismatic and the former modules.

The new API keeps no name, keyword, default, unit string or argument order of
py_multislice or of the replaced modules, with two exceptions. The NumPy and
PyTorch convention names `shape`, `dtype` and `device` stay, and so do the
generic names `msd` (mean squared displacement), `depth` and `jitter`, which
the former potential code used in the same roles. Only the structure and
potential teams had a separate Python environment. The grid and
scattering-factor team used the development environment, which also held
abTEM and the former scatterem package. An audit of the work records found no
use of them.

One interface outside the physics base layer keeps py_multislice names:
`scatterem.simulation.eels.build_transition_potentials` keeps the beam energy
name `eV`. `subshell_transitions(Z, n, l, epsilon, ..., order=1)` keeps the
parameter order and the `order=1` default of py_multislice `get_transitions`,
and the `mlprime` field of `Transition` has the name of the py_multislice
parameter.

The forward STEM-EELS functions in `scatterem.simulation.eels`
(`multislice_transition_potential`, `prism_transition_potential`,
`prism_eels_image`) implement the published algorithm of H. G. Brown, J.
Ciston and C. Ophus, Phys. Rev. Research 1, 033186 (2019). py_multislice
implements the same algorithm. A provenance review in v0.4 found no code in
common. The optional `site_threshold` of
`multislice_transition_potential` compares the same two intensities as the
`threshhold` option of py_multislice. Its module docstring derives the rule.

## abTEM (interface only)

abTEM (https://github.com/abTEM/abTEM) is licensed under the GNU General
Public License v3.0. The EELS physics of scatterem was rewritten clean-room in
v0.4, and no code, expression or docstring of abTEM is used. The
specification, implementation and fixer agents worked in a git worktree of the
scatterem repository, whose history holds the former abTEM-derived code. They
used the development environment, which also held abTEM 1.0.9. They could
read and run the EELS tests, some of which carried abTEM-shaped expressions,
and the orchestrator named abTEM-specific choices of the former code for them
to avoid. An audit of their work records found no read of the git history,
of abTEM or of the former code bodies (exposed, not used). Some names and
defaults follow abTEM:

- from `SubshellTransitions`: the `epsilon`, `order` (default 1), `xc`
  (default `"PBE"`) and `lprimes` parameters of
  `scatterem.simulation.eels.subshell_transitions`, the `mlprime` field of
  `Transition`, and the `lprime` and `epsilon` parameters of the continuum
  wavefunction functions;
- from `calculate_bound_radial_wavefunction` and
  `calculate_continuum_radial_wavefunction`: the parameters `(Z, n, l)` in
  this order and the `xc="PBE"` default of
  `scatterem.simulation.eels.physics.radial.bound_wavefunction`, and the
  `xc="PBE"` default of `radial.continuum_wavefunction`;
- the function names `natural_neighbor_weights`
  (`scatterem.utils.stem.beamlets`),
  `validate_units` (`scatterem.data.validation`) and the method name
  `_build_s_matrix` (`scatterem.nn.scan_exit_wave.prism`).

The physical constants and the electron wavelength, the wave-aberration
function, its Warp twin and the aperture masks (`scatterem.physics.constants`,
`scatterem.physics.optics` and `scatterem.physics.kernels.lens_wavefront_wp`)
are clean-room rewrites. Two teams did the rewrite, as for the other
clean-room modules. The blind team worked in a sandbox that held
none of the replaced modules and no abTEM, py4DSTEM, py_multislice, prismatic
or quantem code, with a separate Python environment in which none of these
packages imports. The acceptance tests compared the new code with recorded
outputs of the former code. The blind team used labels of its own for the 12
aberration slots. After acceptance, the scatterem author decided that v0.4
has one set of aberration labels. A non-blind agent, who knew the former
modules and abTEM, then changed the slot labels of `LENS_LAYOUT` and
`AMPLITUDE_LAYOUT` in `scatterem.physics.optics` to the Krivanek names.
These are standard notation (O. L. Krivanek, N. Dellby and A. R. Lupini,
Ultramicroscopy 78, 1 (1999)):

- `LENS_LAYOUT`: `"C10"`, `"C12a"`, `"C12b"`, `"C21a"`, `"C21b"`, `"C23a"`,
  `"C23b"`, `"C30"`, `"C32a"`, `"C32b"`, `"C34a"`, `"C34b"`;
- `AMPLITUDE_LAYOUT`: `"C10"`, `"C12"`, `"C21"`, `"C23"`, `"C30"`, `"C32"`,
  `"C34"`, then `"phi12"`, `"phi21"`, `"phi23"`, `"phi32"`, `"phi34"`.

These label strings are the same as those of the former modules and of
abTEM. `LENS_LAYOUT` also has the slot order of the former
`COEFFICIENT_NAMES`; this order follows from the term order (order `n`, then
multiplicity `m`, then the cosine part before the sine part). The order of
`AMPLITUDE_LAYOUT` (all magnitudes, then all angles) is not the interleaved
order (`"C12"`, `"phi12"`, `"C21"`, `"phi21"`, ...) of the former
`cartesian_to_polar` and of the abTEM aberration fields. It is the order of
`POLAR_NAMES` of the scatterem `Aberrations` container, which is now the same
tuple. Apart from these
label strings, no name, keyword, default or argument order of the former
modules or of abTEM is kept, except the NumPy and PyTorch convention names
`shape`, `dtype` and `device`. The former mask functions also had a `shape`
parameter for the output grid, in the same role.

## py4DSTEM (interface only)

py4DSTEM (https://github.com/py4dstem/py4DSTEM) is licensed under the GNU
General Public License v3.0. The centre-of-mass calibration in
`scatterem.data.calibration` was rewritten clean-room in v0.4 from a
behavioural specification, and no code, expression or docstring of py4DSTEM
is used. The implementation agent worked in a git worktree of the scatterem
repository, whose history holds the former py4DSTEM-derived code, and used the
development environment. Its tests included one that compares the outputs
with recorded outputs of the former code. It also read the former
scan-rotation search, which v0.4 replaces too (see below). An audit of its
work records found no read of the git history, of py4DSTEM or of the former
centre-of-mass code (exposed, not used). Its interface stays from py4DSTEM so that callers keep it: the
names and coefficient order of `plane`, `parabola` and `bezier_two`; the
names, parameters, defaults and return order of `fit_2D` (`function`, `data`,
`data_mask=None`, `popt=None`, `robust=False`, `robust_steps=3`,
`robust_thresh=2`; a 4-tuple return) and of `fit_origin` (`data`,
`mask=None`, `robust`, `robust_steps=3`, `robust_thresh=2`, the default
`"plane"` and the `"bezier_two"` option). Two `fit_origin` parameters have
new spellings of py4DSTEM's `fitfunction` and `returnfitp`: `fit_function`
and `return_fit_params`. quantem (MIT, above) has the same `fit_function`
spelling in its `fit_origin`; the
name `fftshift_checkerboard`; the order of the
12 aberration coefficients `C10` to `C34b`; the parameter names of `show_2d`;
the name, the parameters (`dp_mask`,
`fit_function="plane"`, `com_shifts`, `com_measured`,
`vectorized_calculation=True`) and the six-array return order of
`_calculate_intensities_center_of_mass`.

The scan-rotation search, its figures and the bright-field shift estimators
(`scatterem.data.calibration.find_scan_rotation`,
`scatterem.vis.rotation_figures`,
`scatterem.reconstruction.ptychography.direct.measure_bf_parallax`) are
clean-room rewrites with a new API. So are the SSB Warp kernels
(`scatterem.nn.functional.warp.ptychography`) and their aperture predicate
(`scatterem.utils.warp.probe_cone_pass`). No py4DSTEM name, parameter, default or argument order of these parts is kept.
The scan-rotation and bright-field team worked in a sandbox but used the
development environment, which also held the former scatterem package. The
SSB kernel team worked in a branch of the scatterem repository, whose history
holds the former kernels. An audit of the work records found no read of the
former code in either case. The SSB team saw the message of the commit that
removed the former kernels, which names four removed helpers (none of these
names is in the new code), and compared its outputs with recorded outputs of
the former kernels (exposed, not used).

The bright-field disk measurement
(`scatterem.data.pupil_gauge.measure_pupil`) and the DFT frequency grid and
ring profile (`scatterem.utils.spectral_mesh`) are clean-room rewrites. Two
teams did the rewrite. A specification
team that knew the former code wrote a behavioural contract. A separate blind
team wrote the code from that contract and the published literature, in a
sandbox that held none of the replaced modules and no py4DSTEM, quantem,
abTEM or py_multislice code, with a separate Python environment in which
none of these packages imports. The acceptance tests compared the new code
with recorded outputs of the former code and with exact references. After
the blind run, other agents, who were
not blind, renamed the names that the provenance audits flagged. For this,
they compared the names with py4DSTEM, quantem, abTEM, py_multislice and the
former modules. No name, keyword, default, option string or argument order
of the former modules is kept, except the PyTorch convention names `dtype`
and `device`.

## Lobato and Van Dyck (2014), Table 1 (data)

`scatterem/physics/_lobato2014_table.py` holds the elastic electron
scattering-factor coefficients for Z = 1 to 103 from Table 1 of I. Lobato and
D. Van Dyck, "An accurate parameterization for scattering factors, electron
densities and electrostatic potentials for neutral atoms that obey all
physical constraints", Acta Cryst. A70 (2014) 636-649,
https://doi.org/10.1107/S205327331401643X. The coefficients are numerical
facts, reproduced with this citation. Cite the article when you use them.
