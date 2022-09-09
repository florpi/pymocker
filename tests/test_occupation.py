import numpy as np
from scipy.stats import binned_statistic
from pymocker.occupation import Zheng07Centrals, Zheng07Sats
from pymocker.galaxy import VanillaGalaxy


def test_mean_centrals(halo_cat):
    galaxy = VanillaGalaxy()
    halo_mass = np.logspace(
        np.log10(1.05 * np.min(halo_cat.mass)), np.log10(np.max(halo_cat.mass)), 10
    )

    cens = Zheng07Centrals()
    n_centrals = cens(halo_cat=halo_cat, galaxy=galaxy)
    mean_centrals, _, _ = binned_statistic(
        halo_cat.mass,
        n_centrals,
        bins=halo_mass,
    )
    bin_center = 0.5 * (halo_mass[1:] + halo_mass[:-1])
    expected_mean_centrals = cens.get_mean_occ(
        halo_mass=bin_center,
        galaxy=galaxy,
    )
    np.testing.assert_allclose(mean_centrals, expected_mean_centrals, atol=0.1)


def test_mean_satellites(halo_cat):
    galaxy = VanillaGalaxy()
    halo_mass = np.logspace(
        np.log10(1.05 * np.min(halo_cat.mass)), np.log10(np.max(halo_cat.mass)), 30
    )
    sats = Zheng07Sats()
    cens = Zheng07Centrals()
    n_sats = sats(halo_cat=halo_cat, galaxy=galaxy)
    mean_sats, _, _ = binned_statistic(
        halo_cat.mass,
        n_sats,
        bins=halo_mass,
    )
    bin_center = 0.5 * (halo_mass[1:] + halo_mass[:-1])
    expected_mean_centrals = cens.get_mean_occ(
        halo_mass=bin_center,
        galaxy=galaxy,
    )
    expected_mean_satellites = sats.get_mean_occ(
        halo_mass=bin_center,
        galaxy=galaxy,
        n_centrals=expected_mean_centrals,
    )
    np.testing.assert_allclose(mean_sats, expected_mean_satellites, rtol=0.15)


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
