from abc import ABC, abstractmethod
import numpy as np
from pymocker.catalogues.halo import HaloCatalogue


class Positioner(ABC):
    def get_pos(
        self, halo_cat: HaloCatalogue, n_tracers: np.array[int], **kwargs
    ) -> np.array[float]:
        """Abstract class describing a positioner

        Args:
            halo_cat (HaloCatalogue): halo catalogue to be populated
            n_tracers (np.array[int]): number of tracers to draw for each halo

        Returns:
            np.array[float]: positions of the desired tracers
        """
        pass


class IdentityPositioner(Positioner):
    def get_pos(self, halo_cat: HaloCatalogue, **kwargs) -> np.array[float]:
        """Sample tracers only where halos are

        Args:
            halo_cat (HaloCatalogue): halo catalogue to be populated

        Returns:
            np.array[float]: positions of the desired tracers
        """
        return halo_cat.pos
