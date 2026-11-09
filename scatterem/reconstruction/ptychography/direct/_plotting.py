"""Diagnostic plotting for measured bright-field shifts and PCTF fusion weights."""

import numpy as np
import torch

from scatterem.utils.lazy_plotting import plt, vis


def plot_bright_field_shifts(
    query_points,
    shift_values,
    sampling,
    wavelength,
    opt_rotation,
    arrow_scale,
    suptitle=None,
):
    """Plot bright field shifts before and after rotation correction.

    Parameters
    ----------
    query_points : torch.Tensor
        Query points, shape ``(N, 2)``, ordered by radius in the bright
        field in increasing order.
    shift_values : torch.Tensor
        Shift values, shape ``(N, 2)``, ordered by radius in the bright
        field in increasing order.
    sampling : torch.Tensor
        Sampling, shape ``(2,)``.
    wavelength : float
        Wavelength in Angstroms.
    opt_rotation : float
        Rotation in degrees.
    arrow_scale : float
        Arrow scale.
    suptitle : str, optional
        Figure super-title. Default None.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    if suptitle is not None:
        fig.suptitle(suptitle)
    # ax2 = ax[0].twinx()  # Create right axis
    ax3 = ax[0].twiny()  # Create top axis
    shift_values = shift_values.cpu().numpy()
    # Draw arrows for each position
    q_query_points = query_points.cpu().numpy() * sampling[0].cpu().numpy()

    ax[0].quiver(
        q_query_points[:, 1],
        q_query_points[:, 0],
        shift_values[:, 1] * arrow_scale,
        shift_values[:, 0] * arrow_scale,
        fc="r",
        ec="r",
        label="Shift vectors",
    )
    ax[0].scatter(
        q_query_points[:, 1],
        q_query_points[:, 0],
        color="b",
        s=1,
        marker="x",
        label="Query points",
    )

    f = 1.2
    # Calculate k ranges
    k_min_x = q_query_points[:, 1].min()
    k_max_x = q_query_points[:, 1].max()
    k_min_y = q_query_points[:, 0].min()
    k_max_y = q_query_points[:, 0].max()

    # Calculate alpha ranges (alpha = k*lambda)
    alpha_min_x = k_min_x * wavelength * 1e3  # Convert to mrad
    alpha_max_x = k_max_x * wavelength * 1e3
    alpha_min_y = k_min_y * wavelength * 1e3
    alpha_max_y = k_max_y * wavelength * 1e3

    # Create 5 evenly spaced ticks centered at zero
    k_ticks_x = np.linspace(k_min_x, k_max_x, 5)
    k_ticks_y = np.linspace(k_min_y, k_max_y, 5)
    alpha_ticks_x = np.linspace(alpha_min_x, alpha_max_x, 5)
    alpha_ticks_y = np.linspace(alpha_min_y, alpha_max_y, 5)

    # Set limits and ticks
    ax[0].set_ylim(k_min_y * f, k_max_y * f)
    ax[0].set_xlim(k_min_x * f, k_max_x * f)
    # ax2.set_ylim(alpha_min_y * f, alpha_max_y * f)
    ax3.set_xlim(alpha_min_x * f, alpha_max_x * f)

    ax[0].set_title("Bright Field Shifts")
    ax[0].set_ylabel(r"$k_y$ [$\mathrm{\AA^{-1}}$]")
    ax[0].set_xlabel(r"$k_x$ [$\mathrm{\AA^{-1}}$]")

    ax[0].set_yticks(k_ticks_y)
    ax[0].set_xticks(k_ticks_x)
    # ax2.set_yticks(alpha_ticks_y)
    ax3.set_xticks(alpha_ticks_x)

    # ax2.set_yticklabels([f'{a:.1f}' for a in alpha_ticks_y])
    ax3.set_xticklabels([f"{a:.1f}" for a in alpha_ticks_x])

    # ax2.set_ylabel(r"$\alpha_x$ [mrad]")
    ax3.set_xlabel(r"$\alpha_x$ [mrad]")

    ax[0].legend()

    # Create twin axes
    ax2 = ax[1].twinx()  # Create right axis
    ax3 = ax[1].twiny()  # Create top axis
    # A local 2x2 rotation keeps this module's imports small.
    angle_rad = torch.deg2rad(-opt_rotation).item()
    rot_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )

    # Apply rotation to query points
    q_query_points_rot = np.dot(q_query_points, rot_matrix)

    ax[1].quiver(
        q_query_points_rot[:, 1],
        q_query_points_rot[:, 0],
        shift_values[:, 1] * arrow_scale,
        shift_values[:, 0] * arrow_scale,
        fc="r",
        ec="r",
        label=f"Shift vectors (bin {bin})",
    )
    ax[1].scatter(
        q_query_points_rot[:, 1],
        q_query_points_rot[:, 0],
        color="b",
        s=1,
        marker="x",
        label="Query points",
    )

    # Calculate alpha ranges (alpha = k*lambda)
    alpha_min_x = k_min_x * wavelength * 1e3  # Convert to mrad
    alpha_max_x = k_max_x * wavelength * 1e3
    alpha_min_y = k_min_y * wavelength * 1e3
    alpha_max_y = k_max_y * wavelength * 1e3

    # Create 5 evenly spaced ticks centered at zero
    k_ticks_x = np.linspace(k_min_x, k_max_x, 5)
    k_ticks_y = np.linspace(k_min_y, k_max_y, 5)
    alpha_ticks_x = np.linspace(alpha_min_x, alpha_max_x, 5)
    alpha_ticks_y = np.linspace(alpha_min_y, alpha_max_y, 5)

    # Set limits and ticks
    ax[1].set_ylim(k_min_y * f, k_max_y * f)
    ax[1].set_xlim(k_min_x * f, k_max_x * f)
    ax2.set_ylim(alpha_min_y * f, alpha_max_y * f)
    ax3.set_xlim(alpha_min_x * f, alpha_max_x * f)

    ax[1].set_title("Bright Field Shifts (Rotated)")
    # ax[1].set_ylabel(r"$k_y$ [$\mathrm{\AA^{-1}}$]")
    ax[1].set_xlabel(r"$k_x$ [$\mathrm{\AA^{-1}}$]")

    # ax[1].set_yticks(k_ticks_y)
    ax[1].set_yticks([])
    ax[1].set_yticklabels([])
    ax[1].set_xticks(k_ticks_x)
    ax2.set_yticks(alpha_ticks_y)
    ax3.set_xticks(alpha_ticks_x)

    ax2.set_yticklabels([f"{a:.1f}" for a in alpha_ticks_y])
    ax3.set_xticklabels([f"{a:.1f}" for a in alpha_ticks_x])

    ax2.set_ylabel(r"$\alpha_y$ [mrad]")
    ax3.set_xlabel(r"$\alpha_x$ [mrad]")

    # ax[1].legend()
    plt.tight_layout()
    plt.show()


def plot_pctf_fusion_weights(weight_tcDF: torch.Tensor) -> None:
    """Show the dark-field and phase weights of the PCTF-quantile fusion.

    Parameters
    ----------
    weight_tcDF : torch.Tensor
        The corner-centred dark-field weight map; the phase weight shown
        beside it is ``1 - weight_tcDF``.
    """
    fig, ax = plt.subplots(1, 2, figsize=(8, 8))
    vis.show_2d(
        # fftshift for DISPLAY only -- the returned weights stay corner-centred.
        [
            torch.fft.fftshift(weight_tcDF),
            torch.fft.fftshift(1 - weight_tcDF),
        ],
        cbar=True,
        title=["Weight Dark field", "Weight Phase"],
        figax=(fig, ax),
    )
    plt.show()


def show_bright_field_shift_maps(vBF, measured_shifts, fitted_shifts):
    """Show measured and fitted bright-field shifts as maps over the detector.

    Parameters
    ----------
    vBF : DatasetVirtualBrightField4DStem
        Provides the detector grid (``diffraction_pattern_mean_normalized``)
        and the bright-field pixel indices ``bright_field_inds``.
    measured_shifts : torch.Tensor
        Measured shifts, shape ``(N, 2)`` ``(row, col)``, one per bright-field
        pixel in ``bright_field_inds`` order.
    fitted_shifts : torch.Tensor
        Shifts of the fitted aberrations, same shape and order.
    """
    rows, cols = vBF.bright_field_inds[:, 0], vBF.bright_field_inds[:, 1]
    plots = []
    for shifts in (measured_shifts, fitted_shifts):
        for axis in (1, 0):
            grid = torch.zeros_like(vBF.diffraction_pattern_mean_normalized)
            grid[rows, cols] = shifts[:, axis]
            plots.append(grid)
    titles = [
        "CC Shift Values X",
        "CC Shift Values Y",
        "Fitted Shifts X",
        "Fitted Shifts Y",
    ]
    vis.show_2d(plots, cbar=True, title=titles, role="signed")
