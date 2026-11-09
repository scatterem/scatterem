# Third-party notices

scatterem is licensed under the Apache License 2.0 (see `LICENSE`). Parts of
it keep names and layouts from the third-party projects below, one file derives
from torchvision, and one data table comes from a journal article. This file
carries their notices.

## quantem (MIT)

quantem (https://github.com/electronmicroscopy/quantem) is licensed under the
MIT License. Parts of scatterem derive from quantem: `scatterem/vis/visualization.py`,
`scatterem/vis/visualization_utils.py`, `scatterem/vis/_display_input.py`,
`scatterem/vis/normalization.py` (the `NORMALIZATION_PRESETS` table and
`NormalizationConfig`), `scatterem/data/validation.py`,
`scatterem/data/datasets/base.py`, `scatterem/data/datasets/dataset_4dstem.py`
and `scatterem/data/datasets/virtual_bright_field.py`.
They keep quantem's API names, signatures, field lists, docstring text or
figure handling. Every run of 5 or more identical lines (comments and blank
lines not counted) in this release is listed below, with the line numbers of
the v0.4.0 files. The release gate recomputes these runs and fails when the
list is wrong.

- `scatterem/vis/normalization.py` 90-100: the fields and defaults of
  `NormalizationConfig`.
- `scatterem/data/datasets/base.py` 85-93, 100-104, 106-114: the `__init__`
  signature of the dataset container, its attribute assignments, and the
  signature of the `from_array` class method; 170-177, 206-213: the `name` and
  `signal_units` property pairs.
- `scatterem/data/datasets/dataset_4dstem.py` 168-172: the constructor
  arguments; 184-191: the arguments passed to the base-class constructor;
  739-747: the signature of `from_array`; 883-888, 921-925: keyword arguments
  passed when a new dataset is built; 965-972: the `origin` property pair.
- `scatterem/data/datasets/virtual_bright_field.py` 176-183: the arguments
  passed to the base-class constructor; 312-317: the `cls(...)` call of the
  constructor class method.
- `scatterem/vis/visualization.py` 86-91, 106-111, 358-362: numpydoc
  parameter text of the 2D display functions; 350-356: a signature; 420-427:
  the figure and axes handling; 478-486: the layout and return lines.
- `scatterem/vis/visualization_utils.py` 48-52: parameter text of the scale
  bar; 245-250: the signature of `add_cbar_to_ax`.

The quantem licence follows.

```text
MIT License

Copyright (c) 2025 ophusgroup

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

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

## py_multislice (former basis, replaced in v0.4)

py_multislice by Hamish G. Brown (https://github.com/HamishGBrown/py_multislice)
ships no licence. scatterem v0.3 and earlier based its physics layer on it:
the frequency grids and band limits, the scattering factors, the structure
container and its file readers, the projected-potential builder and the
transmission function. v0.4 replaces these modules with clean-room rewrites
(`scatterem.physics.periodic_sampling`, `atom_amplitudes`, `atom_set`,
`beam_partition`, `potential_stack` and the modules it uses, `thin_slice`).
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
parameter. v0.4 renames the band-limit keyword of
`build_transition_potentials` from `bandwidth_limit` to `bandlimit`; its
default stays 2/3.

The forward STEM-EELS functions in `scatterem.simulation.eels`
(`multislice_transition_potential`, `prism_transition_potential`,
`prism_eels_image`) implement the published algorithm of H. G. Brown, J.
Ciston and C. Ophus, Phys. Rev. Research 1, 033186 (2019). py_multislice
implements the same algorithm. A provenance review in v0.4 found no code in
common. v0.4 changes the function name `transition_potential_multislice` to
`multislice_transition_potential`; the list above gives the names that stay.
The optional `site_threshold` of
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
defaults stay from earlier scatterem releases and follow abTEM:

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

The antialiasing aperture of v0.3 and earlier (`antialias_aperture`,
`antialias_cutoff_gpts`, the cutoff 0.6666666 and the rim width 0.01)
followed abTEM. v0.4 replaces it with the clean-room
`scatterem.physics.periodic_sampling.CircularPassBand`, which keeps none of
these names or values.

In v0.3 the physical constants and the electron wavelength, the
wave-aberration function with its polar-to-Cartesian conversion, its Warp
twin and the aperture masks were rewritten from the textbook relations by an
author who had seen former code that followed abTEM (a copy of
`energy2wavelength` and of the `cartesian2polar` block), and were checked for
parity against it. v0.4 replaces these modules with clean-room rewrites
(`scatterem.physics.constants`, `scatterem.physics.optics` and
`scatterem.physics.kernels.lens_wavefront_wp`). Two teams did the rewrite, as
for the other clean-room modules. The blind team worked in a sandbox that held
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

v0.3 and earlier based the scan-rotation search, its figures and the
bright-field shift estimators on py4DSTEM. v0.4 replaces them with clean-room
rewrites (`scatterem.data.calibration.find_scan_rotation`,
`scatterem.vis.rotation_figures`,
`scatterem.reconstruction.ptychography.direct.measure_bf_parallax`) with a new
API, and the SSB Warp kernels
(`scatterem.nn.functional.warp.ptychography`) and their aperture predicate
(`scatterem.utils.warp.probe_cone_pass`) with a clean-room rewrite. No
py4DSTEM name, parameter, default or argument order of these parts is kept.
The scan-rotation and bright-field team worked in a sandbox but used the
development environment, which also held the former scatterem package. The
SSB kernel team worked in a branch of the scatterem repository, whose history
holds the former kernels. An audit of the work records found no read of the
former code in either case. The SSB team saw the message of the commit that
removed the former kernels, which names four removed helpers (none of these
names is in the new code), and compared its outputs with recorded outputs of
the former kernels (exposed, not used).

v0.3 measured the bright-field disk with a threshold-area fit (its module
`disk_fit`) and with a RANSAC circle fit. The threshold-area
fit replaced a port of the py4DSTEM `get_probe_size`, and its author had seen
that port. The same release had
the DFT frequency grid and the ring profile of its module `grids`. v0.4
replaces these modules with clean-room rewrites
(`scatterem.data.pupil_gauge.measure_pupil` and
`scatterem.utils.spectral_mesh`). Two teams did the rewrite. A specification
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
and `device`. The `thresh_lower`, `thresh_upper` and `N` parameters of the
bright-field methods of `Dataset4DStem`, which followed `get_probe_size`, are
removed.

## Lobato and Van Dyck (2014), Table 1 (data)

`scatterem/physics/_lobato2014_table.py` holds the elastic electron
scattering-factor coefficients for Z = 1 to 103 from Table 1 of I. Lobato and
D. Van Dyck, "An accurate parameterization for scattering factors, electron
densities and electrostatic potentials for neutral atoms that obey all
physical constraints", Acta Cryst. A70 (2014) 636-649,
https://doi.org/10.1107/S205327331401643X. The coefficients are numerical
facts, reproduced with this citation. Cite the article when you use them.
