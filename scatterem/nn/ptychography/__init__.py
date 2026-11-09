"""Per-slice transmission kernels shared by the multislice and PRISM operators.

Written for the public release. It ships :mod:`._transmit` and
:mod:`._gather_per_slice` only. ``multislice_transmit_slice`` is re-exported
because the multislice and PRISM operators import it from here.
"""

from ._transmit import multislice_transmit_slice as multislice_transmit_slice

__all__ = ["multislice_transmit_slice"]
