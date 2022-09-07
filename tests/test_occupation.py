import numpy as np
from pymocker.occupation import Zheng07Centrals, Zheng07Sats
from pymocker.galaxy import VanillaGalaxy


def test_satellites_only_if_centrals():
    galaxy = VanillaGalaxy()
    halo_mass = np.logspace(12, 16, 30)
    cens = Zheng07Centrals()
    sats = Zheng07Sats()
    n_centrals = cens.get_mean_occ(halo_mass, galaxy=galaxy)
    n_sats = sats.get_mean_occ(halo_mass, n_centrals=n_centrals, galaxy=galaxy)

    assert sum(n_centrals) > 0
    assert sum(n_sats) > 0

    assert sum(n_sats > n_centrals) > 0
