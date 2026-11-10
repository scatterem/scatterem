"""Named datasets that download themselves on first use.

This release carries the four FF-STEM specimens (Zenodo record 18008901,
CC-BY-4.0). They download from a Hugging Face mirror of the record, with
Zenodo as fallback, and are md5-checked.
"""

from .you2026 import (
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
