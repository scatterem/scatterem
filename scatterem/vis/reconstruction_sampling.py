"""The real-space pixel size of a reconstruction, for scale bars.

A reconstruction is up-sampled relative to the scan, so its pixel is *smaller*
than the scan step. Passing ``dataset.sampling`` to a plot therefore overstates a
scale bar by the upsample factor -- four-fold on the paper's Fig. 4, which is the
difference between a correct bar and one that is silently four times too long.

The four figure scripts each worked this out inline, in four copies of the same
five lines. This is that calculation, once.
"""

from __future__ import annotations

from typing import Any, Sequence

from scatterem.utils.data.sampling import Sampling


def reconstruction_sampling(
    dataset: Any,
    image: Any = None,
    *,
    shape: Sequence[int] | None = None,
    units: tuple[str, str] = ("Å", "Å"),
) -> Sampling:
    """``Sampling`` describing one pixel of a reconstructed image.

    The scan covered ``scan_px * scan_step`` of specimen; the reconstruction
    represents that same extent with ``image_px`` pixels, so::

        pixel_size = scan_step * scan_px / image_px

    which needs no knowledge of the upsample factor that was requested -- it reads
    the factor off the result. That matters because ``upsample="nyquist"`` picks
    the factor from the data, so the caller may not know what it was.

    Args:
        dataset: the dataset the reconstruction came from.
        image: the reconstruction. Its last two dimensions give the output shape.
        shape: output shape, if you would rather pass it than the image.
        units: axis unit labels for the returned ``Sampling``.

    Returns:
        A :class:`~scatterem.utils.data.sampling.Sampling` for the two real-space
        axes.

    Raises:
        ValueError: neither ``image`` nor ``shape`` was given, or the shape is not
            two-dimensional.
    """
    if shape is None:
        if image is None:
            raise ValueError("pass either the reconstructed image or its shape")
        shape = tuple(image.shape)[-2:]
    if len(shape) != 2:
        raise ValueError(f"expected a 2D output shape; got {tuple(shape)}")

    scan_shape = tuple(dataset.array.shape)[:2]
    scan_step = (float(dataset.sampling[0]), float(dataset.sampling[1]))

    return Sampling(
        pixel_size=(
            scan_step[0] * int(scan_shape[0]) / int(shape[0]),
            scan_step[1] * int(scan_shape[1]) / int(shape[1]),
        ),
        units=units,
    )
