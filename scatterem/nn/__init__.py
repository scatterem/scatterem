"""The neural-network subpackage.

This package re-exports nothing: each consumer imports the module it needs
directly.
"""

import warp as _wp

# Initialise Warp here, once, for every module below this package: they hand
# torch tensors to Warp through `wp.from_torch`, which does not initialise Warp
# by itself. The top-level `scatterem/__init__.py` does not call it, but it
# imports `scatterem.reconstruction`, so `import scatterem` starts Warp.
_wp.init()
