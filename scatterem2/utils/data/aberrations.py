import torch
from torch import Tensor

# from scatterem2.utils.data.datasets import RasterScanningDiffractionDataset


class Aberrations:
    def __init__(self, array: Tensor = torch.zeros((12,))) -> None:
        self.array: Tensor = array

    # @abstractmethod
    # def determine_from(self, dataset: RasterScanningDiffractionDataset, meta4d: Metadata4D) -> "Aberrations":
    #     pass
