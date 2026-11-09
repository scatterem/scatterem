#!/usr/bin/env python3
"""Generate ``docs/api.qmd`` from the installed package's docstrings.

Deliberately not a documentation framework. The alternatives (quartodoc, sphinx)
each pull in a dependency tree to do something this package needs one page of, and
they need their own configuration language on top of the one Quarto already has.

More to the point: this reads the **live** objects, so the reference cannot drift
from the code. If a documented function is renamed, this fails; if its signature
changes, the page changes. A hand-written API page would silently rot.

Run from the repository root:

    python docs/_generate_api.py
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

#: What the reference covers, in reading order. Explicit rather than discovered:
#: the public surface is a decision, not whatever happens to be importable.
SECTIONS: list[tuple[str, str, list[str]]] = [
    (
        "Package settings",
        "scatterem",
        ["configure"],
    ),
    (
        "Container",
        "scatterem.data.datasets",
        ["Dataset", "Dataset4DStem", "DatasetVirtualBrightField4DStem", "with_meta"],
    ),
    (
        "Running a reconstruction",
        "scatterem.reconstruction",
        ["run", "Result", "ReconstructionConfig", "Provenance", "register_runner"],
    ),
    (
        "Direct ptychography and the aberration fit",
        "scatterem.reconstruction.ptychography.direct",
        [
            "AberrationFitConfig",
            "AberrationFitResult",
            "DirectPtychographyConfig",
            "DirectPtychographyResult",
        ],
    ),
    (
        "Bright-field parallax",
        "scatterem.reconstruction.ptychography.direct",
        [
            "measure_bf_parallax",
            "fit_aberrations_and_rotation_to_bright_field_shifts",
            "aberrations_and_rotation_from_bright_field_shifts",
        ],
    ),
    (
        "Scan rotation and centre-of-mass calibration",
        "scatterem.data.calibration",
        [
            "find_scan_rotation",
            "ScanRotation",
            "RotationSearch",
            "fit_origin",
            "fit_2D",
            "plane",
            "parabola",
            "bezier_two",
        ],
    ),
    (
        "Scan-rotation figures",
        "scatterem.vis.rotation_figures",
        ["angle_search_figure", "plot_scan_maps"],
    ),
    (
        "Tilt-corrected dark field",
        "scatterem.reconstruction.darkfield",
        [
            "TiltCorrectedDarkFieldConfig",
            "TiltCorrectedDarkFieldResult",
            "tilt_corrected_dark_field_depth_section",
        ],
    ),
    (
        "Fused full field",
        "scatterem.reconstruction.fusion.ffstem",
        ["FusedFullFieldConfig", "FusedFullFieldResult"],
    ),
    (
        "Reconstruction functions",
        "scatterem.reconstruction",
        ["direct_ptychography", "tilt_corrected_dark_field", "fused_full_field"],
    ),
    (
        "Named datasets",
        "scatterem.data.public",
        ["You2026Carbon", "You2026Co3O4", "You2026AuLowDose", "You2026Gd2O3"],
    ),
    (
        "Metadata",
        "scatterem.data.acquisition",
        ["AcquisitionMetadata"],
    ),
    (
        "Aberration coefficients and scan metadata",
        "scatterem.data",
        ["Aberrations", "SMeta"],
    ),
    (
        "Aberrations and the probe",
        "scatterem.physics.optics",
        [
            "LENS_LAYOUT",
            "LENS_TERMS",
            "AMPLITUDE_LAYOUT",
            "pack_lens",
            "unpack_lens",
            "lens_wavefront",
            "lens_wavefront_slope",
            "lens_to_amplitude",
            "lens_from_amplitude",
            "RIM_WIDTH_PX",
            "smooth_disc",
            "smooth_annulus",
            "illumination_disc",
        ],
    ),
    (
        "Calibration helpers",
        "scatterem.data.pupil_gauge",
        ["measure_pupil", "PupilEstimate", "ESTIMATORS"],
    ),
    (
        "Physics",
        "scatterem.physics.constants",
        [
            "debroglie_at",
            "gamma_at",
            "moving_mass_at",
            "phase_per_potential_at",
            "bend_per_tesla_at",
            "EXACT_PLANCK",
            "EXACT_E_CHARGE",
            "EXACT_LIGHT_SPEED",
            "EXACT_AVOGADRO",
            "EXACT_BOLTZMANN",
            "CODATA18_ELECTRON_MASS",
            "CODATA18_DALTON",
            "CODATA18_FINE_STRUCTURE",
            "CODATA18_VACUUM_MU",
            "CODATA18_ELECTRON_RADIUS",
            "DERIVED_PLANCK_BAR",
            "DERIVED_ELECTRON_REST_EV",
            "DERIVED_AB_COUPLING",
        ],
    ),
    (
        "Structures",
        "scatterem.physics.atom_set",
        [
            "AtomSet",
            "read_atom_set",
            "has_partial_occupancy",
            "DEFAULT_TILT_SINE",
            "FULL_OCCUPANCY_TOLERANCE",
            "MAX_SUPERCELL_FACTOR",
        ],
    ),
    (
        "Slicing along the beam",
        "scatterem.physics.beam_partition",
        ["BeamPartition", "species_order", "edge_snap_margin", "EDGE_SNAP_ULPS"],
    ),
    (
        "Projected potentials",
        "scatterem.physics.potential_stack",
        ["potential_stack", "check_stack_request", "StackRequest", "resolve_stack_device"],
    ),
    (
        "Per-row replacements",
        "scatterem.physics.row_replacements",
        ["RowReplacements"],
    ),
    (
        "Thermal displacements",
        "scatterem.physics.thermal_jitter",
        ["ThermalJitter", "jitter_offsets", "colocated_rows", "COLOCATION_TOLERANCE"],
    ),
    (
        "Depth shares",
        "scatterem.physics.axial_shares",
        ["axial_shares", "AxialShares", "BEAM_SPREADS", "GAUSS_IMAGE_REACH"],
    ),
    (
        "Potential rolloff",
        "scatterem.physics.potential_rolloff",
        ["rolloff_mask", "check_rolloff", "DEFAULT_ROLLOFF"],
    ),
    (
        "Spectrum reuse",
        "scatterem.physics.spectrum_sweep",
        ["spectrum_sweep", "SweepLedger"],
    ),
    (
        "Potential routes",
        "scatterem.physics.route_settings",
        [
            "SpatialRouteSettings",
            "choose_route",
            "route_rolloff",
            "check_route",
            "check_cutoff",
            "ROUTE_NAMES",
            "DEFAULT_PROJECTED_CUTOFF",
        ],
    ),
    (
        "Transmission functions",
        "scatterem.physics.thin_slice",
        ["thin_slice_transmission", "phase_from_projection"],
    ),
    (
        "Scattering amplitudes",
        "scatterem.physics.atom_amplitudes",
        [
            "born_amplitude",
            "born_amplitude_array",
            "born_amplitudes",
            "potential_spectra",
            "gamma_wavelength",
            "validated_atomic_number",
            "POTENTIAL_PER_AMPLITUDE",
        ],
    ),
    (
        "Sampling grids and pass bands",
        "scatterem.physics.periodic_sampling",
        [
            "PeriodicGrid",
            "dft_sample_frequencies",
            "CircularPassBand",
            "apply_pass_band",
            "ellipse_mask",
            "lowpass_ellipse",
            "MaskCacheStats",
            "mask_cache_stats",
            "clear_mask_cache",
        ],
    ),
    (
        "STEM-EELS and PRISM driver",
        "scatterem.simulation.eels",
        ["PrismEelsSimulator", "PrismResult"],
    ),
    (
        "Transition potentials",
        "scatterem.simulation.eels",
        [
            "subshell_transitions",
            "build_transition_potentials",
            "Transition",
            "TransitionPotentials",
            "AtomicRadialWavefunction",
            "bound_wavefunction",
            "continuum_wavefunction",
            "gpaw_available",
            "cache_dir",
            "HydrogenicFallbackWarning",
        ],
    ),
    (
        "EELS algorithms",
        "scatterem.simulation.eels",
        [
            "multislice_transition_potential",
            "prism_transition_potential",
            "prism_transition_potential_hybrid",
            "prism_eels_image",
            "ScatteringMatrix",
            "PartitionedScatteringMatrix",
            "DetectorExitSMatrix",
        ],
    ),
    (
        "Elastic PRISM",
        "scatterem.nn.scan_exit_wave",
        [
            "ScanExitWaveOperator",
            "PrismScanOp",
            "MultisliceScanOp",
            "build_prism_smeta",
            "build_prism_smeta_for_probe",
        ],
    ),
    (
        "Propagation",
        "scatterem.nn.modules",
        ["Propagator", "SubpixShift"],
    ),
    (
        "Slice transmission in ptychography",
        "scatterem.nn.ptychography",
        ["multislice_transmit_slice"],
    ),
    (
        "Beamlet sampling",
        "scatterem.utils.stem",
        ["beamlet_samples", "natural_neighbor_weights", "fftshift_checkerboard"],
    ),
    (
        "Free-space propagation and probe shifts",
        "scatterem.physics.propagation",
        [
            "fourier_shift_stack",
            "propagate",
            "propagator_kernel",
            "propagator_phase",
            "angular_spectrum_kernel",
            "transfer_phase",
            "tilt_phase",
        ],
    ),
    (
        "Grids",
        "scatterem.utils.spectral_mesh",
        ["dual_mesh", "shell_means", "ShellMeans"],
    ),
    (
        "Visualisation",
        "scatterem.vis.visualization",
        ["show_2d", "show_2d_array"],
    ),
    (
        "EELS radial states",
        "scatterem.simulation.eels.physics.radial",
        [
            "resolve_wavefunctions",
            "hydrogenic_bound_wavefunction",
            "free_continuum_wavefunction",
        ],
    ),
    ("Crystal phases", "scatterem.crystal.phase", ["Phase"]),
    (
        "Elements",
        "scatterem.crystal.elements",
        ["atomic_number", "atomic_weight", "symbol"],
    ),
    (
        "Fourier helpers",
        "scatterem.physics.fourier",
        [
            "phase_ramp",
            "fourier_shift",
            "fourier_rescale",
            "dft_center",
            "fftshift_last2",
            "fftshift_rfft",
            "ifftshift_rfft",
            "rfft_shape",
        ],
    ),
    ("Axis-angle rotations", "scatterem.rotation.axis_angle", ["to_matrix"]),
    (
        "Quaternions",
        "scatterem.rotation.quaternion",
        [
            "from_axis_angle",
            "from_matrix",
            "from_rotation_vector",
            "multiply",
            "to_matrix",
        ],
    ),
    (
        "Display and animation",
        "scatterem.vis",
        ["show", "show_animated", "save_animated"],
    ),
    (
        "Colour maps and normalisation",
        "scatterem.vis",
        [
            "choose_colormap",
            "colormap_for_role",
            "register_colormaps",
            "ColourChoice",
            "DataTraits",
            "NormalizationConfig",
            "CustomNormalization",
        ],
    ),
    (
        "Figure style and scale bars",
        "scatterem.vis",
        [
            "use_style",
            "figure_params",
            "draw_scale_bar",
            "ScalebarConfig",
            "UncalibratedDataWarning",
        ],
    ),
    (
        "Colour",
        "scatterem.vis.complex_color",
        ["complex_to_rgba", "phase_wheel", "tile_to_rgba"],
    ),
    (
        "Scale bars",
        "scatterem.data.sampling",
        ["Sampling", "reconstruction_sampling"],
    ),
    (
        "Saving and loading",
        "scatterem.utils.store",
        [
            "Serializable",
            "load_serializable",
            "read_schema",
            "SerializationError",
            "SCHEMA_VERSION",
        ],
    ),
]

#: Methods worth documenting, per class. A class not listed here renders its own
#: docstring only. These surfaces are large and much of each is plumbing; the
#: entries are the ones a pipeline is actually driven through.
CLASS_METHODS: dict[str, list[str]] = {
    "AberrationFitResult": ["applied_to", "as_tuple"],
    "DirectPtychographyResult": ["as_tuple"],
    "TiltCorrectedDarkFieldResult": ["as_tuple"],
    "FusedFullFieldResult": ["as_tuple"],
    "Dataset4DStem": [
        "from_array",
        "calibrate_reciprocal_from_bright_field",
        "bright_field_radius_and_center",
        "save",
        "load",
    ],
    "PrismEelsSimulator": [
        "eels_map",
        "qeels",
        "eels_cube",
        "elastic_cbed",
        "probe_q",
        "scan_pixels",
        "transmissions",
        "sites",
        "transition_potentials",
    ],
    "AtomSet": ["with_fractional", "with_rotation", "replicated"],
    "BeamPartition": [
        "uniform",
        "from_thicknesses",
        "edge_values",
        "thicknesses",
        "subrange",
        "bin_index",
        "bin_index_clamped",
    ],
    "PeriodicGrid": ["from_pixel_sizes"],
    "CircularPassBand": ["mask", "min_crop_shape"],
}

#: Supporting modules: the building blocks that the methods above use. They are
#: public (no underscore, an ``__all__``), so the reference shows each name of
#: their ``__all__`` that no section above shows, under one heading per module.
SUPPORTING_MODULES: list[str] = [
    "scatterem.data.detector_warp",
    "scatterem.data.geometry",
    "scatterem.nn.regularizers.tv",
    "scatterem.physics.kernels.lens_wavefront_wp",
    "scatterem.reconstruction.darkfield.core",
    "scatterem.reconstruction.darkfield.drizzle",
    "scatterem.vis.colors",
    "scatterem.vis.normalization",
    "scatterem.vis.scale_bar",
    "scatterem.vis.show",
    "scatterem.vis.style",
    "scatterem.vis.visualization_utils",
]

#: Internal modules: shipped because exported modules import them, but not part
#: of the public API since v0.4. They keep their
#: ``__all__`` for their importers; the reference does not show them, and
#: :func:`_public_modules` skips them and every module below them. Their names
#: can change without notice. ``prism_project.basis`` became ``_basis`` and
#: needs no entry here.
INTERNAL_MODULES: tuple[str, ...] = (
    "scatterem.data.storage",
    "scatterem.utils.env",
    "scatterem.utils.lazy_plotting",
    "scatterem.utils.warp",
)


#: Modules whose whole ``__all__`` the reference must cover, in addition to every
#: module named in SECTIONS. A name counts as covered when the same object is
#: documented in any section (a re-export does not need a second entry).
#: :func:`_public_modules` adds every other public module of the package that
#: has an ``__all__``, so a new public module cannot stay out of the reference.
COMPLETE_MODULES: list[str] = ["scatterem.vis"]


def _public_modules() -> list[str]:
    """Every module with no underscore part and an ``__all__``, minus INTERNAL_MODULES."""
    import pkgutil

    import scatterem

    names = []
    for info in pkgutil.walk_packages(scatterem.__path__, "scatterem."):
        parts = info.name.split(".")
        if any(p.startswith("_") for p in parts) or "tests" in parts:
            continue
        if any(
            info.name == m or info.name.startswith(m + ".") for m in INTERNAL_MODULES
        ):
            continue
        module = importlib.import_module(info.name)
        if getattr(module, "__all__", None):
            names.append(info.name)
    return names


def _signature(obj) -> str:
    try:
        return f"{obj.__name__}{inspect.signature(obj)}"
    except (TypeError, ValueError):
        return getattr(obj, "__name__", str(obj))


def _docstring(obj) -> str:
    return inspect.cleandoc(obj.__doc__ or "*Undocumented.*")


#: Anchors already used on the page. Two functions with one name (the
#: ``to_matrix`` of two rotation modules) get the module in the second anchor.
_ANCHORS: set[str] = set()


def _anchor(obj) -> str:
    anchor = obj.__qualname__.replace(".", "-").lower()
    if anchor in _ANCHORS:
        anchor = f"{obj.__module__}.{obj.__qualname__}".replace(".", "-").lower()
    _ANCHORS.add(anchor)
    return anchor


def _render_callable(obj, level: int) -> list[str]:
    hashes = "#" * level
    return [
        f"{hashes} `{obj.__name__}` {{#{_anchor(obj)}}}",
        "",
        "```python",
        _signature(obj),
        "```",
        "",
        _docstring(obj),
        "",
    ]


def _render_constant(name: str, value, level: int) -> list[str]:
    return [
        f"{'#' * level} `{name}` {{#{name.lower()}}}",
        "",
        "```python",
        f"{name} = {value!r}",
        "```",
        "",
    ]


def _is_documentable(obj) -> bool:
    return inspect.isclass(obj) or callable(obj)


def _render_value(name: str, value, level: int) -> list[str]:
    """A constant, or the type of an object whose repr is long or not a value."""
    text = repr(value)
    if inspect.ismodule(value) or len(text) > 200:
        kind = "module" if inspect.ismodule(value) else type(value).__name__
        return [f"{'#' * level} `{name}` {{#{name.lower()}}}", "", f"A `{kind}`.", ""]
    return _render_constant(name, value, level)


def _check_complete(documented_ids: set[int], documented_names: set[tuple[str, str]]):
    """Fail when a module of the reference exports a name that it does not show.

    Every module named in SECTIONS or COMPLETE_MODULES is checked against its
    ``__all__``. A missing entry is a stale reference, so the build stops.
    """
    modules = dict.fromkeys(
        [m for _, m, _ in SECTIONS] + COMPLETE_MODULES + _public_modules()
    )
    missing = []
    for module_name in modules:
        module = importlib.import_module(module_name)
        for name in getattr(module, "__all__", ()):
            obj = getattr(module, name, None)
            if (module_name, name) in documented_names:
                continue
            if obj is not None and _is_documentable(obj) and id(obj) in documented_ids:
                continue
            missing.append(f"{module_name}.{name}")
    if missing:
        raise SystemExit(
            "These public names are in an __all__ but not in docs/_generate_api.py: "
            + ", ".join(missing)
            + ". Add each one to a section, or remove it from __all__."
        )


def _render_class(cls, level: int, methods: list[str] | None) -> list[str]:
    out = [f"{'#' * level} `{cls.__name__}`", "", _docstring(cls), ""]
    for name in methods or []:
        member = getattr(cls, name, None)
        if member is None:
            raise SystemExit(
                f"{cls.__name__}.{name} is documented but does not exist. Either "
                f"the name changed or the list in docs/_generate_api.py is stale -- "
                f"this fails rather than quietly shipping a reference to nothing."
            )
        target = member.__func__ if isinstance(member, classmethod) else member
        out += _render_callable(target, level + 1)
    return out


def build() -> str:
    lines = [
        "---",
        'title: "API reference"',
        "toc-depth: 3",
        "---",
        "",
        "Generated from the source by `docs/_generate_api.py`, so it cannot drift",
        "from the code. Everything below is public; anything not listed here is",
        "internal and may change without notice.",
        "",
    ]
    documented_ids: set[int] = set()
    documented_names: set[tuple[str, str]] = set()
    for title, module_name, names in SECTIONS:
        module = importlib.import_module(module_name)
        lines += [f"## {title}", "", f"`{module_name}`", ""]
        for name in names:
            if not hasattr(module, name):
                raise SystemExit(
                    f"{module_name}.{name} is documented but not importable. "
                    f"Fix the name or the list in docs/_generate_api.py."
                )
            obj = getattr(module, name)
            documented_names.add((module_name, name))
            if inspect.isclass(obj):
                documented_ids.add(id(obj))
                lines += _render_class(obj, 3, CLASS_METHODS.get(name))
            elif callable(obj):
                documented_ids.add(id(obj))
                lines += _render_callable(obj, 3)
            else:
                lines += _render_constant(name, obj, 3)
    lines += ["## Supporting modules", ""]
    for module_name in SUPPORTING_MODULES:
        module = importlib.import_module(module_name)
        rest = [
            n
            for n in module.__all__
            if (module_name, n) not in documented_names
            and not (_is_documentable(getattr(module, n)) and id(getattr(module, n)) in documented_ids)
        ]
        if not rest:
            continue
        lines += [f"### `{module_name}`", ""]
        for name in rest:
            obj = getattr(module, name)
            documented_names.add((module_name, name))
            if inspect.isclass(obj):
                documented_ids.add(id(obj))
                lines += _render_class(obj, 4, None)
            elif callable(obj):
                documented_ids.add(id(obj))
                lines += _render_callable(obj, 4)
            else:
                lines += _render_value(name, obj, 4)
    _check_complete(documented_ids, documented_names)
    return "\n".join(lines) + "\n"


def main() -> int:
    out = Path(__file__).parent / "api.qmd"
    text = build()
    out.write_text(text)
    n = text.count("\n### ")
    print(f"wrote {out} ({len(text.splitlines())} lines, {n} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
