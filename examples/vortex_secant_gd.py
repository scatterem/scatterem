"""Mixed-state ptychography with a vortex probe, solved with ``secant_gd``.

Run it with

    python examples/vortex_secant_gd.py

It takes a few minutes on a CPU and uses a GPU when one is present.

What it shows
-------------
The reconstruction path of the MEMS programmable-electron-optics work
(S. You et al.) on synthetic data:

1. A vortex probe (a disk aperture with the phase ``exp(i l theta)``) in two
   incoherent modes illuminates a weak phase object on a rotated raster
   scan. :class:`PlanarPtychography` computes the far-field amplitudes.
2. A new model starts from one coherent mode and a vacuum object. The
   solver ``solve_ptychography(model, batches, solver="secant_gd")`` refines
   the probe, the object and the scan offsets for some passes.
3. :meth:`PlanarPtychography.expand_modes` turns the one-mode state into
   three modes, and the solve continues with the mode orthogonalisation
   after each pass. A :class:`PassStopCallback` returns the last accepted
   pass.

``solver="secant_gd"`` is minibatch gradient descent with per-group Newton
step sizes in the span of the group gradients, after Carlsson, Nikitin,
Troedsson & Wendt, "The bilinear Hessian for large scale optimization",
arXiv:2502.03070. Its curvature is a secant estimate from gradient
differences, so it needs no learning rate and no double backward pass.

What it does NOT show
---------------------
This is the method on synthetic data, not the paper's measurement. The paper's
reconstructions also use a detector mask over the needle band, an
aperture-plane probe support and an absorption bound
(:class:`PlanarConstraints`), with the settings of the paper's methods
section. Do not expect the numbers here to reproduce a figure.
"""

from __future__ import annotations

import math

import torch

from scatterem.reconstruction.ptychography.iterative.second_order import (
    PassStopCallback,
    PassStopRule,
    PlanarConstraints,
    PlanarPtychography,
    solve_ptychography,
)

GRID = 48  # detector pixels per side
DK = 0.05  # detector pixel, 1/A
STEP = 0.8  # scan step, A
SCAN = 16  # scan positions per side


def oam_probe(n: int, charge: int, radius: float) -> torch.Tensor:
    """Real-space probe of a disk aperture with the phase ``exp(i l theta)``."""
    y = torch.arange(n, dtype=torch.float64) - n / 2
    yy, xx = torch.meshgrid(y, y, indexing="ij")
    aperture = (torch.hypot(yy, xx) < radius).to(torch.complex128)
    aperture = aperture * torch.exp(1j * charge * torch.atan2(yy, xx))
    dims = (-2, -1)
    probe = torch.fft.ifft2(torch.fft.ifftshift(aperture, dim=dims))
    return torch.fft.fftshift(probe, dim=dims).to(torch.complex64)


def mean_loss(model: PlanarPtychography, batches: list) -> float:
    """The loss over all batches, no gradient."""
    with torch.no_grad():
        return sum(float(model.loss(b)) for b in batches) / len(batches)


def purity(probe: torch.Tensor) -> float:
    """``sum p_i^2`` of the normalised mode powers."""
    power = (probe.abs() ** 2).reshape(probe.shape[0], -1).sum(-1)
    p = power / power.sum()
    return float((p**2).sum())


def simulate(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Intensities of a two-mode vortex probe over a weak phase object."""
    p0 = oam_probe(GRID, charge=1, radius=GRID / 6)
    p1 = 0.35 * oam_probe(GRID, charge=2, radius=GRID / 6)
    probe = torch.stack([p0, p1])[:, None].to(device)
    blank = torch.ones(SCAN, SCAN, GRID, GRID, device=device)
    truth = PlanarPtychography.from_scan(blank, STEP, DK, probe, rotation_deg=3.0)
    gen = torch.Generator().manual_seed(0)
    h, w = truth.object.shape
    phase = torch.randn(1, 1, h, w, generator=gen)
    phase = torch.nn.functional.avg_pool2d(phase, 5, stride=1, padding=2)[0, 0]
    with torch.no_grad():
        truth.object.copy_(torch.complex(1.5 * phase, 0.02 * phase.abs()).to(device))
        every = torch.arange(SCAN * SCAN, device=device)
        intensities = truth(every) ** 2
    return intensities.reshape(SCAN, SCAN, GRID, GRID), truth.probe.detach()


def main() -> None:
    """Simulate, reconstruct with one mode, expand to three, reconstruct."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    intensities, true_probe = simulate(device)
    seed = oam_probe(GRID, charge=1, radius=GRID / 6)[None, None].to(device)
    model = PlanarPtychography.from_scan(
        intensities, STEP, DK, seed, rotation_deg=3.0
    )
    batches = model.batches(32)
    print(f"start loss {mean_loss(model, batches):.4f}")
    solve_ptychography(
        model,
        batches,
        n_iter=20,
        solver="secant_gd",
        optimize_positions=True,
    )
    print(f"one mode, 20 passes: loss {mean_loss(model, batches):.4f}")

    model, ratio = model.expand_modes(3, mode0_frac=0.9)
    model.constraints = PlanarConstraints(orthogonalize=True)
    print(f"three modes (probe power / norm was {ratio:.3f})")
    callback = PassStopCallback(model, PassStopRule(), return_last=True)
    log = solve_ptychography(
        model,
        batches,
        n_iter=40,
        solver="secant_gd",
        probe_mode_groups=True,
        optimize_positions=True,
        callback=callback,
    )
    callback.restore_last()
    print(f"three modes, {len(log.losses)} passes: loss {mean_loss(model, batches):.4f}")
    probe = model.probe.detach()
    print(f"purity {purity(probe[:-1]):.3f} (truth {purity(true_probe):.3f})")
    overlap = torch.vdot(probe[0].flatten(), true_probe[0].flatten()).abs()
    overlap = overlap / (probe[0].norm() * true_probe[0].norm())
    print(f"mode-0 overlap with the true mode 0: {float(overlap):.3f}")
    assert math.isfinite(mean_loss(model, batches))


if __name__ == "__main__":
    main()
