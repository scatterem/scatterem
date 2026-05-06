import warnings
from importlib.util import find_spec

import torch
import warp

from .io import load, print_file, read_4dstem, read_emdfile_to_4dstem

warnings.filterwarnings(
    "ignore", message="The 'train_dataloader' does not have many workers"
)


torch.set_float32_matmul_precision("medium")
warp.init()

if find_spec("astra"):
    from scatterem2.tomography import *

__version__ = "0.1.0"
