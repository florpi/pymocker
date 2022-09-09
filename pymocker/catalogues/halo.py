from pymocker.catalogues.base import Catalogue
import numpy as np
from typing import Optional


class HaloCatalogue(Catalogue):
    def __init__(
        self,
        pos: np.array,
        vel: np.array,
        mass: np.array,
        hid: Optional[np.array] = None,
        concentration: Optional[np.array] = None,
        boxsize: Optional[float] = None,
        redshift: Optional[float] = None,
    ):
        """Catalogue of dark matter halos

        Args:
            pos (np.array): array of positions of size (N, 3)
            vel (np.array): array of velocities of size (N, 3)
            mass (np.array): array of halo masses
            hid (Optional[np.array], optional): halo id
            concentration (Optional[np.array], optional): halo concentration, used for NFW profiles. Defaults to None.
            boxsize (Optional[float], optional): size of the simulated box. Defaults to None.
            redshift (Optional[float], optional): redshift of simulated snapshot. Defaults to None.
        """
        self.pos = pos
        self.vel = vel
        self.mass = mass
        if hid is None:
            self.hid = np.arange(len(pos))
        else:
            self.hid = hid
        # TODO: Add option to go from halo id to halo idx through dictionary
        self.concentration = concentration
        self.boxsize = boxsize
        self.redshift = redshift
        self.attrs_to_frame = [
            "mass",
        ]

    @classmethod
    def from_forge(cls,):

        return cls(
            pos=pos,
            vel=vel,
            mass=mass,
            boxsize=boxsize,
            redshift=redshift,
        )
