"""The EELS ``wavefunctions`` default: ``"auto"``, as in v0.3.

``"auto"`` resolves to GPAW when it is importable and to the analytic
hydrogenic states otherwise. GPAW is an optional extra (``scatterem[eels]``),
so the fallback is allowed, but it emits one ``HydrogenicFallbackWarning`` per
call. GPAW availability is patched here; GPAW itself never runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from scatterem.simulation.eels import HydrogenicFallbackWarning
from scatterem.simulation.eels.physics.radial import (
    bound_wavefunction,
    continuum_wavefunction,
    resolve_wavefunctions,
)

_RADIAL = "scatterem.simulation.eels.physics.radial"


def test_resolve_auto_is_gpaw_when_it_is_importable(monkeypatch, recwarn):
    monkeypatch.setattr(f"{_RADIAL}.gpaw_available", lambda: True)
    assert resolve_wavefunctions("auto") == "gpaw"
    assert not [w for w in recwarn if w.category is HydrogenicFallbackWarning]


def test_resolve_auto_falls_back_to_hydrogenic_with_one_warning(monkeypatch):
    monkeypatch.setattr(f"{_RADIAL}.gpaw_available", lambda: False)
    with pytest.warns(HydrogenicFallbackWarning) as record:
        assert resolve_wavefunctions("auto") == "hydrogenic"
    assert len(record) == 1
    text = str(record[0].message)
    # The text states the measured map-shape change against v0.3, with the
    # hydrogenic range first, and names the eels extra.
    assert "eels extra" in text and "differ from v0.3" in text
    assert "18-61 % with these hydrogenic states" in text
    assert record[0].filename == __file__  # points at the caller, not radial.py


@pytest.mark.parametrize("choice", ["gpaw", "hydrogenic"])
def test_resolve_explicit_choices_never_warn(choice, monkeypatch, recwarn):
    monkeypatch.setattr(f"{_RADIAL}.gpaw_available", lambda: False)
    assert resolve_wavefunctions(choice) == choice
    assert not [w for w in recwarn if w.category is HydrogenicFallbackWarning]


def test_resolve_rejects_an_unknown_choice():
    with pytest.raises(ValueError, match="unknown wavefunctions 'nope'"):
        resolve_wavefunctions("nope")


def test_the_default_is_auto_on_every_radial_entry_point():
    import inspect

    from scatterem.simulation.eels import subshell_transitions

    for fn in (bound_wavefunction, continuum_wavefunction, subshell_transitions):
        assert inspect.signature(fn).parameters["wavefunctions"].default == "auto"


def test_subshell_transitions_warns_once_per_call_on_fallback(tmp_path, monkeypatch):
    """The default resolves once, so one subshell gives one warning, not one per l'."""
    from scatterem.simulation.eels import subshell_transitions

    monkeypatch.setenv("SCATTEREM_EELS_CACHE", str(tmp_path))
    monkeypatch.setattr(f"{_RADIAL}.gpaw_available", lambda: False)
    with pytest.warns(HydrogenicFallbackWarning) as record:
        auto = subshell_transitions(8, 2, 1, 5.0)  # l' = 0, 1, 2
    assert len(record) == 1
    explicit = subshell_transitions(8, 2, 1, 5.0, wavefunctions="hydrogenic")
    assert len(auto) == len(explicit)
    for a, b in zip(auto, explicit, strict=True):
        np.testing.assert_array_equal(a.bound.u, b.bound.u)
        np.testing.assert_array_equal(a.excited.u, b.excited.u)


def test_gpaw_is_not_imported_at_module_level():
    """GPAW is an optional GPL extra: importing the EELS package must not load it."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    import scatterem

    root = str(Path(scatterem.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": root}
    code = "import sys, scatterem.simulation.eels; print('GPAW', 'gpaw' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    # Warp may print a banner first; read only the marker line.
    assert "GPAW False" in out.stdout.splitlines()


def test_subshell_transitions_default_uses_gpaw_when_it_is_importable(
    tmp_path, monkeypatch, recwarn
):
    """The default dispatches to the GPAW builders; they are stubbed, not run."""
    from scatterem.simulation.eels import subshell_transitions
    from scatterem.simulation.eels.physics import radial

    monkeypatch.setenv("SCATTEREM_EELS_CACHE", str(tmp_path))
    monkeypatch.setattr(f"{_RADIAL}.gpaw_available", lambda: True)
    calls = []

    def bound(Z, n, ell, xc):
        calls.append(("bound", Z, n, ell, xc))
        return radial.hydrogenic_bound_wavefunction(Z, n, ell)

    def cont(Z, lp, eps, xc, ps):
        calls.append(("cont", Z, lp, eps, xc, ps))
        return radial.free_continuum_wavefunction(lp, eps)

    monkeypatch.setattr(f"{_RADIAL}._gpaw_bound", bound)
    monkeypatch.setattr(f"{_RADIAL}._gpaw_continuum", cont)
    trans = subshell_transitions(8, 1, 0, 5.0)  # l' = 0, 1
    assert calls == [
        ("bound", 8, 1, 0, "PBE"),
        ("cont", 8, 0, 5.0, "PBE", 1.0),
        ("cont", 8, 1, 5.0, "PBE", 1.0),
    ]
    assert len(trans) == 1 + 3  # (l'=0: 1 m') + (l'=1: 3 m')
    assert not [w for w in recwarn if w.category is HydrogenicFallbackWarning]


def test_fallback_warning_points_at_the_user_call(tmp_path, monkeypatch):
    """The warning names the user's line, also through the driver.

    Frames inside scatterem are skipped, so Python's default filter shows the
    warning once per user call site, not once per process.
    """
    from scatterem.physics.atom_set import AtomSet
    from scatterem.simulation.eels import PrismEelsSimulator, subshell_transitions

    monkeypatch.setenv("SCATTEREM_EELS_CACHE", str(tmp_path))
    monkeypatch.setattr(f"{_RADIAL}.gpaw_available", lambda: False)
    structure = AtomSet(
        np.array([4.0, 4.0, 4.0]),
        np.array([[0.5, 0.5, 0.5]]),
        [8.0],
        site_fraction=np.ones(1),
        msd=np.full(1, 0.006),
    )
    sim = PrismEelsSimulator(
        structure,
        eV=200e3,
        semiconvergence_angle=20.0,
        pixels=(16, 16),
        scan=(2, 2),
        device="cpu",
    )
    for call in (
        lambda: subshell_transitions(8, 1, 0, 5.0),
        lambda: sim.transition_potentials((8, 1, 0), 5.0),
    ):
        with pytest.warns(HydrogenicFallbackWarning) as record:
            call()
        assert len(record) == 1
        assert record[0].filename == __file__
