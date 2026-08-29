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
        "Container",
        "scatterem.utils.data.datasets",
        ["Dataset4dstem", "DatasetVirtualBrightField4dstem"],
    ),
    (
        "Reconstruction",
        "scatterem.reconstruction",
        ["direct_ptychography", "tilt_corrected_dark_field", "fused_full_field"],
    ),
    (
        "Named datasets",
        "scatterem.datasets",
        ["You2026Carbon", "You2026Co3O4", "You2026AuLowDose", "You2026Gd2O3"],
    ),
    (
        "Metadata",
        "scatterem.utils.data.data_classes",
        ["Metadata4dstem"],
    ),
    (
        "STEM-EELS simulation",
        "scatterem.simulation",
        ["StemEelsSimulator", "StemEelsResult", "Structure"],
    ),
    (
        "Transition potentials",
        "scatterem.simulation",
        [
            "subshell_transitions",
            "build_transition_potentials",
            "TransitionPotentials",
            "bound_wavefunction",
            "continuum_wavefunction",
            "gpaw_available",
        ],
    ),
    (
        "EELS algorithms",
        "scatterem.simulation.eels",
        [
            "transition_potential_multislice",
            "prism_transition_potential",
            "prism_eels_image",
            "ScatteringMatrix",
            "PartitionedScatteringMatrix",
            "DetectorExitSMatrix",
        ],
    ),
    (
        "Multislice substrate",
        "scatterem.simulation",
        ["make_potential", "make_transmission_functions", "interaction_constant"],
    ),
    (
        "Aberrations",
        "scatterem.utils.aberration_basis",
        ["cartesian_chi", "cartesian_to_polar", "polar_to_cartesian"],
    ),
    (
        "Calibration helpers",
        "scatterem.utils.data.disk_fit",
        ["fit_bright_field_disk"],
    ),
    (
        "Physics",
        "scatterem.utils.physics",
        ["electron_wavelength"],
    ),
    (
        "Grids",
        "scatterem.utils.grids",
        ["fft_frequencies_2d", "radial_average"],
    ),
    (
        "Visualisation",
        "scatterem.vis.visualization",
        ["show_2d"],
    ),
    (
        "Colour",
        "scatterem.vis.complex_color",
        ["complex_to_rgba", "phase_wheel", "tile_to_rgba"],
    ),
    (
        "Scale bars",
        "scatterem.vis.reconstruction_sampling",
        ["reconstruction_sampling"],
    ),
    (
        "Saving and loading",
        "scatterem.io.store",
        ["Serializable", "read_schema"],
    ),
]

#: Methods worth documenting, per class. A class not listed here renders its own
#: docstring only. These surfaces are large and much of each is plumbing; the
#: entries are the ones a pipeline is actually driven through.
CLASS_METHODS: dict[str, list[str]] = {
    "Dataset4dstem": [
        "from_array",
        "determine_aberrations_",
        "direct_ptychography",
        "tilt_corrected_dark_field",
        "fused_full_field",
        "calibrate_reciprocal_from_bright_field",
        "bright_field_radius_and_center",
        "save",
        "load",
    ],
    "StemEelsSimulator": [
        "simulate",
        "simulate_window",
        "simulate_edges",
    ],
    "Structure": [
        "fromfile",
        "tile",
        "rotate",
    ],
}


def _signature(obj) -> str:
    try:
        return f"{obj.__name__}{inspect.signature(obj)}"
    except (TypeError, ValueError):
        return getattr(obj, "__name__", str(obj))


def _docstring(obj) -> str:
    return inspect.cleandoc(obj.__doc__ or "*Undocumented.*")


def _render_callable(obj, level: int) -> list[str]:
    hashes = "#" * level
    return [
        f"{hashes} `{obj.__name__}` {{#{obj.__qualname__.replace('.', '-').lower()}}}",
        "",
        "```python",
        _signature(obj),
        "```",
        "",
        _docstring(obj),
        "",
    ]


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
        "title: \"API reference\"",
        "toc-depth: 3",
        "---",
        "",
        "Generated from the source by `docs/_generate_api.py`, so it cannot drift",
        "from the code. Everything below is public; anything not listed here is",
        "internal and may change without notice.",
        "",
    ]
    for title, module_name, names in SECTIONS:
        module = importlib.import_module(module_name)
        lines += [f"## {title}", "", f"`{module_name}`", ""]
        for name in names:
            obj = getattr(module, name, None)
            if obj is None:
                raise SystemExit(
                    f"{module_name}.{name} is documented but not importable. "
                    f"Fix the name or the list in docs/_generate_api.py."
                )
            if inspect.isclass(obj):
                lines += _render_class(obj, 3, CLASS_METHODS.get(name))
            else:
                lines += _render_callable(obj, 3)
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
