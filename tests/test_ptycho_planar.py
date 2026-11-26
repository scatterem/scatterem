"""The 2D mixed-state model ``PlanarPtychography`` and its second-order solve.

The 2D model is the path of ``solve_ptychography`` that needs no trainer.
These tests recover a small mixed state with the secant solver, check the
options the model refuses, the expansion from one mode to several, and the
per-pass constraints. A separate test pins the 2D model to the unified
``Ptychography`` module.
"""

from __future__ import annotations

import math

import pytest
import torch
from _planar_fixture import planar_truth as _truth

from scatterem.reconstruction.ptychography.iterative.second_order.driver import (
    solve_ptychography,
)
from scatterem.reconstruction.ptychography.iterative.second_order.planar import (
    PlanarPtychography,
)
from scatterem.reconstruction.ptychography.iterative.second_order.planar_constraints import (  # noqa: E501
    PlanarConstraints,
)
from scatterem.reconstruction.ptychography.iterative.second_order.stop_rule import (
    PassStopCallback,
    PassStopRule,
    state_change,
)


def test_secant_gd_lowers_the_loss_and_moves_the_state_to_the_truth() -> None:
    """The solve lowers the loss and moves the probe state toward the truth.

    It does not recover the mode powers of this small scene: the second mode
    keeps about 2 % of the power (13.8 % in the truth). The bound on the
    state distance is therefore relative to the seed.
    """
    truth, intensities = _truth()
    seed = truth.probe.detach().clone()
    seed[1] = 0.3 * torch.roll(seed[0], 2, dims=-2)  # wrong second mode
    model = PlanarPtychography.from_scan(
        intensities,
        0.4,
        0.1,
        seed,
        rotation_deg=3.0,
        constraints=PlanarConstraints(orthogonalize=True),
    )
    start_state = model.probe.detach().clone()
    batches = model.batches(16, seed=0)
    with torch.no_grad():
        start = sum(float(model.loss(b)) for b in batches) / len(batches)
    rule = PassStopRule(probe_tol=1e-6, min_iters=40)
    cb = PassStopCallback(model, rule, return_last=True)
    log = solve_ptychography(
        model,
        batches,
        n_iter=40,
        solver="secant_gd",
        probe_mode_groups=True,
        optimize_positions=True,
        callback=cb,
    )
    assert log.solver == "secant_gd"
    assert all(math.isfinite(v) for v in log.losses)
    assert cb.restore_last()
    with torch.no_grad():
        end = sum(float(model.loss(b)) for b in batches) / len(batches)
    assert end < 0.2 * start
    target = truth.probe.detach()
    moved = state_change(target, model.probe.detach())
    assert moved < 0.7 * state_change(target, start_state)


def test_unsupported_options_and_empty_batches_raise() -> None:
    truth, _ = _truth(4, 16)
    with pytest.raises(NotImplementedError, match="compile_forward"):
        solve_ptychography(
            truth, truth.batches(4), 1, solver="secant_gd", compile_forward=True
        )
    with pytest.raises(NotImplementedError, match="heldout"):
        solve_ptychography(truth, truth.batches(4), 1, heldout=object())
    with pytest.raises(ValueError, match="non-empty"):
        solve_ptychography(truth, [], 1)


def test_expand_modes_keeps_the_norm_and_rescales_the_object() -> None:
    truth, _ = _truth(4, 16)
    one = PlanarPtychography(
        truth.amplitudes,
        truth.positions,
        truth.offsets.detach(),
        truth.probe.detach()[:1] * 1.7,
        tuple(truth.object.shape),
        truth.norm,
    )
    with torch.no_grad():
        one.object.copy_(truth.object.detach())
    nine, ratio = one.expand_modes(4, 0.96)
    power = float((nine.probe.detach().abs() ** 2).sum())
    assert nine.probe.shape[0] == 4
    assert power == pytest.approx(nine.norm, rel=1e-5)
    assert ratio == pytest.approx(
        float((one.probe.detach().abs() ** 2).sum()) / one.norm
    )
    shift = nine.object.detach() - one.object.detach()
    expected = torch.full_like(shift.imag, -0.5 * math.log(ratio))
    assert torch.allclose(shift.imag, expected)
    assert torch.equal(nine.offsets.detach(), one.offsets.detach())


def test_constraints_keep_the_power_and_reset_opaque_pixels() -> None:
    truth, _ = _truth(6, 24)
    n = truth.probe.shape[-1]
    support = torch.zeros(n, n)
    support[n // 2 - 6 : n // 2 + 6, n // 2 - 6 : n // 2 + 6] = 1.0
    constraints = PlanarConstraints(
        orthogonalize=True,
        radial_support=(1.2, 8.0),
        aperture_support=support,
        absorption_width=2.5,
        absorption_margin=2,
    )
    with torch.no_grad():
        truth.object[5, 7] = complex(0.1, 40.0)  # an opaque pixel
    n_reset = constraints.apply(truth)
    after = truth.probe.detach()
    assert n_reset == 1
    assert float(truth.object[5, 7].imag) < 2.6
    # all modes but the free last one sit on the support
    dims = (-2, -1)
    spectrum = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(after[:-1], dim=dims)), dim=dims
    )
    outside = float((spectrum.abs() ** 2 * (1 - support)).sum())
    assert outside < 1e-6 * float((spectrum.abs() ** 2).sum())
