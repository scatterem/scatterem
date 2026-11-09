"""Volume to images: the central sections of a volume.

Axis convention: a volume is ``(Z, Y, X)``; spacings and positions follow the
same order. Everything here is torch and runs on the volume's device.
"""

from __future__ import annotations


from torch import Tensor

from scatterem.data.binning import binning_factors, block_mean


#: Axis names of a ``(Z, Y, X)`` volume, by index.
AXES: tuple[str, str, str] = ("z", "y", "x")


def central_sections(
    volume: Tensor, max_side: int = 1024
) -> dict[str, tuple[Tensor, tuple[int, int]]]:
    """The XY, XZ and YZ sections through the volume centre.

    Parameters
    ----------
    volume : torch.Tensor
        Real ``(Z, Y, X)`` volume.
    max_side : int
        Sections larger than this along a side are block-averaged.

    Returns
    -------
    dict
        ``"xy"`` (rows y, columns x), ``"xz"`` (rows z, columns x) and ``"yz"``
        (rows z, columns y), each mapped to ``(image, (row_factor,
        column_factor))``.
    """
    nz, ny, nx = volume.shape
    planes = {
        "xy": volume[nz // 2],
        "xz": volume[:, ny // 2, :],
        "yz": volume[:, :, nx // 2],
    }
    out: dict[str, tuple[Tensor, tuple[int, int]]] = {}
    for name, plane in planes.items():
        f = max(binning_factors(plane.shape, max_side))
        factors = (min(f, plane.shape[0]), min(f, plane.shape[1]))
        out[name] = (block_mean(plane, factors), factors)
    return out








