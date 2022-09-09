from pymocker.catalogues.base import Catalogue
import numpy as np
from typing import Optional


class GalaxyCatalogue(Catalogue):
    def __init__(
        self,
        pos: np.array,
        vel: np.array,
        host_id: np.array,
        gal_type: np.array,
        boxsize: Optional[float] = None,
        redshift: Optional[float] = None,
    ):
        self.pos = pos
        self.vel = vel
        self.host_id = host_id
        self.gal_type = gal_type
        self.boxsize = boxsize
        self.redshift = redshift
