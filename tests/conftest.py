from pymocker.catalogues.halo import HaloCatalogue
from pymocker.positioners import NFWPositioner
import numpy as np
import pytest


@pytest.fixture(name="halo_cat", scope="session")
def create_random_halo_catalogue():
    n = 20
    pos = np.random.random((n, 3))
    vel = np.random.random((n, 3))
    mass = np.random.random((n,))
    concentration = np.random.random((n,))
    boxsize = 1.0
    redshift = 0.0
    return HaloCatalogue(
        pos=pos,
        vel=vel,
        mass=mass,
        concentration=concentration,
        boxsize=boxsize,
        redshift=redshift,
    )


@pytest.fixture(name="nfw_pos", scope="session")
def create_nfw():
    return NFWPositioner()
