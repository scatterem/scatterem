"""Named datasets that download themselves on first use.

Written for the public release, which carries only the four FF-STEM specimens
(Zenodo record 18008901, CC-BY-4.0). The private tree's version also exposes
loaders for other groups' published data.
"""

from .public.you2026 import (
    You2026AuLowDose,
    You2026Carbon,
    You2026Co3O4,
    You2026Gd2O3,
)

__all__ = [
    "You2026AuLowDose",
    "You2026Carbon",
    "You2026Co3O4",
    "You2026Gd2O3",
]
