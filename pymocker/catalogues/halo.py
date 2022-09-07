from pymocker.catalogues.base import Catalogue
import numpy as np
from typing import Optional

class HaloCatalogue(Catalogue):
    def __init__(
        self,
        pos: np.array,
        vel: np.array,
        mass: np.array,
        boxsize: Optional[float] = None,
        redshift: Optional[float] = None,
    ):
        self.pos = pos
        self.vel = vel
        self.mass = mass
        self.boxsize = boxsize
        self.redshift = redshift