"""Import rule for the modules of the clean-room unit w8.

The modules ``utils/spectral_mesh.py``, ``utils/_checks.py``,
``data/pupil_gauge.py``, ``data/_pupil_stages.py`` and
``data/_ring_regression.py`` may import only the standard library, numpy,
torch, ``scatterem.utils`` and ``scatterem.data``. An AST scan also sees
imports inside functions.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import scatterem

#: The installed or checked-out package, so the rule also holds for a wheel.
PACKAGE_ROOT = Path(scatterem.__file__).resolve().parent
ALLOWED_THIRD_PARTY = frozenset({"numpy", "torch"})
ALLOWED_SCATTEREM = ("scatterem.utils", "scatterem.data")


def _imports(path: Path) -> set[str]:
    """Absolute names of the modules that `path` imports."""
    package = ".".join(path.relative_to(PACKAGE_ROOT.parent).parent.parts)
    out = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            out.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[: len(base) - node.level + 1]
                name = ".".join(base + ([node.module] if node.module else []))
            else:
                name = node.module or ""
            out.add(name)
    return out


def _forbidden(module: str) -> bool:
    top = module.split(".")[0]
    if top == "__future__" or top in sys.stdlib_module_names:
        return False
    if top in ALLOWED_THIRD_PARTY:
        return False
    return not any(
        module == allowed or module.startswith(allowed + ".")
        for allowed in ALLOWED_SCATTEREM
    )


W8_MODULES = (
    "utils/spectral_mesh.py",
    "utils/_checks.py",
    "data/pupil_gauge.py",
    "data/_pupil_stages.py",
    "data/_ring_regression.py",
)


def _library_files() -> list[Path]:
    return [PACKAGE_ROOT / name for name in W8_MODULES]


def test_library_modules_import_only_allowed_modules():
    violations = [
        f"{path.relative_to(PACKAGE_ROOT)}: {module}"
        for path in _library_files()
        for module in sorted(_imports(path))
        if _forbidden(module)
    ]
    assert not violations, "forbidden imports:\n  " + "\n  ".join(violations)


def test_the_rule_detects_a_forbidden_import():
    """Control: a module that imports SciPy is flagged."""
    assert _forbidden("scipy.ndimage")
    assert _forbidden("scatterem.physics")
    assert not _forbidden("numpy.fft")
    assert not _forbidden("scatterem.data._pupil_stages")
