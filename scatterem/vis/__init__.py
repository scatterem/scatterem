"""Visualization: 2D display, colour policy, normalization and scalebar configuration."""

from scatterem.data.sampling import Sampling as Sampling
from scatterem.vis.colors import ROLES as ROLES
from scatterem.vis.colors import ColourChoice as ColourChoice
from scatterem.vis.colors import DataTraits as DataTraits
from scatterem.vis.colors import choose_colormap as choose_colormap
from scatterem.vis.colors import colormap_for_role as colormap_for_role
from scatterem.vis.colors import register_colormaps as register_colormaps
from scatterem.vis.normalization import CustomNormalization as CustomNormalization
from scatterem.vis.normalization import NormalizationConfig as NormalizationConfig
from scatterem.vis.scale_bar import draw_scale_bar as draw_scale_bar
from scatterem.vis.show import save_animated as save_animated
from scatterem.vis.show import show as show
from scatterem.vis.show import show_animated as show_animated
from scatterem.vis.style import figure_params as figure_params
from scatterem.vis.style import use_style as use_style
from scatterem.vis.visualization import show_2d as show_2d
from scatterem.vis.visualization_utils import ScalebarConfig as ScalebarConfig
from scatterem.vis.visualization_utils import (
    UncalibratedDataWarning as UncalibratedDataWarning,
)

# Adds names such as "batlow" to matplotlib's registry. Changes no rcParams.
register_colormaps()

__all__ = [
    "ColourChoice",
    "CustomNormalization",
    "DataTraits",
    "NormalizationConfig",
    "Sampling",
    "ScalebarConfig",
    "UncalibratedDataWarning",
    "choose_colormap",
    "colormap_for_role",
    "draw_scale_bar",
    "figure_params",
    "register_colormaps",
    "save_animated",
    "show",
    "show_2d",
    "show_animated",
    "use_style",
]
