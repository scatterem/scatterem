"""Whether a multi-rank ``torch.distributed`` group is active.

The second-order solve refuses a multi-rank group: it has no gradient
all-reduce. Callers look the test up as ``_ranks.multi_rank()``, so a test can
patch it in one place.
"""

from __future__ import annotations

import torch.distributed as dist


def multi_rank() -> bool:
    """True when a ``torch.distributed`` group of more than one rank is active.

    Returns
    -------
    bool
        ``dist.is_available() and dist.is_initialized()`` and a world size
        above 1.
    """
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
