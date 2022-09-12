from operator import ge
from pymocker.catalogues.halo import HaloCatalogue
import numpy as np
import pandas as pd
import pytest

# TODO: Add small sample of each simulation to test in any machine?


def test__read_abacus():
    halos = HaloCatalogue.from_abacus(node=0, phase=0, redshift=0.575, boxsize=2000)
    assert (halos.pos[:, :].min() >= 0) and halos.pos[:, :].max() <= 2000

    halos = HaloCatalogue.from_abacus(node=0, phase=3000, redshift=0.575, boxsize=500)
    assert (halos.pos[:, :].min() >= 0) and halos.pos[:, :].max() <= 500


def test__read_forge():
    halos = HaloCatalogue.from_forge(node=1, phase=0, snapshot=16, boxsize=1500.0)
    assert (halos.pos[:, :].min() >= 0) and halos.pos[:, :].max() <= 1500.0

    halos = HaloCatalogue.from_forge(node=3, phase=0, snapshot=16, boxsize=500)
    assert (halos.pos[:, :].min() >= 0) and halos.pos[:, :].max() <= 500
    print(halos.param_dict)

    assert halos.param_dict == {
        "Omega_m": 0.10721,
        "S8": 0.73513,
        "h": 0.61090,
        "|f_R0|": 3.31067e-06,
        "sigma8": 1.22974,
        "B0": -3.60014e-06,
    }
