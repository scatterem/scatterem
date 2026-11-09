"""Tests of the library rules (brief TESTS.md, part E, and SPEC 4.E)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from scatterem.physics import optics as O

#: The installed or checked-out package, so the rules also hold for a wheel.
PACKAGE = Path(O.__file__).resolve().parent
#: The modules of this component. The rest of ``scatterem.physics`` has its own
#: layer rules.
MODULES = tuple(
    PACKAGE / name
    for name in (
        "constants.py",
        "optics.py",
        "_checks.py",
        "_lens_basis.py",
        "_smooth_masks.py",
        "kernels/lens_wavefront_wp.py",
    )
)


def imported_modules(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative import"
            found.add((node.module or "").split(".")[0])
    return found


def test_constants_module_imports_only_stdlib() -> None:
    allowed = set(sys.stdlib_module_names) | {"__future__"}
    assert imported_modules(PACKAGE / "constants.py") <= allowed
    assert not imported_modules(PACKAGE / "optics.py") <= allowed  # control


def _inputs():
    rng = np.random.default_rng(0)
    lens = torch.tensor(rng.standard_normal((2, 12)), requires_grad=True)
    fy = torch.tensor(rng.uniform(-1, 1, (4, 1)))
    fx = torch.tensor(rng.uniform(-1, 1, (1, 5)))
    return lens, fy, fx


@pytest.mark.parametrize(
    "func",
    [
        lambda c, y, x: O.lens_wavefront(c, y, x, lambda_a=0.02),
        lambda c, y, x: O.lens_wavefront_slope(c, y, x, lambda_a=0.02),
        lambda c, y, x: O.lens_to_amplitude(c),
        lambda c, y, x: O.lens_from_amplitude(c),
    ],
)
def test_tensor_functions_do_not_change_inputs(func) -> None:
    lens, fy, fx = _inputs()
    copies = [t.detach().clone() for t in (lens, fy, fx)]
    out = func(lens, fy, fx)
    out.sum().backward()
    for t, c in zip((lens, fy, fx), copies, strict=True):
        assert torch.equal(t.detach(), c)
    assert lens.grad is not None  # the graph reaches the input


def _function_lengths(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], "value", None), ast.Constant)
            ):
                body = body[1:]
            if body:
                yield node.name, body[-1].end_lineno - body[0].lineno + 1


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_module_and_function_sizes(path: Path) -> None:
    assert len(path.read_text().splitlines()) <= 500
    for name, length in _function_lengths(path):
        assert length <= 80, name


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_only_the_kernel_module_imports_warp(path: Path) -> None:
    imports = imported_modules(path)
    if PACKAGE / "kernels" in path.parents:
        assert "warp" in imports  # control: the scan sees the import
    else:
        assert "warp" not in imports, path.name
