"""Core-loss STEM-EELS simulation: multislice, PRISM, and BiP-PRISM.

Exports the algorithms the BiP-PRISM paper describes: conventional
transition-potential multislice, PRISM-EELS, and the bi-partitioned map.
"""

from .multislice_eels import *  # noqa: F401,F403  -- Algorithm 1
from .prism_eels import *  # noqa: F401,F403  -- Algorithms 2-4
from .prism_eels_image import *  # noqa: F401,F403  -- Algorithm 5 (BiP-PRISM)
from .radial import *  # noqa: F401,F403
from .simulator import *  # noqa: F401,F403
from .transition_potentials import *  # noqa: F401,F403
