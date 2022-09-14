from operator import ge
from pymocker.catalogues.halo import HaloCatalogue
import numpy as np
import pandas as pd
import pytest

# TODO: Add small sample of each simulation to test in any machine?


def test__read_abacus():
    halos = HaloCatalogue.from_abacus(node=1, phase=0, redshift=0.575, boxsize=2000)
    assert (halos.pos[:, :].min() >= 0) and halos.pos[:, :].max() <= 2000
    assert halos.param_dict == {
        "omega_b": 0.02242,
        "omega_cdm": 0.1134,
        "h": 0.7030,
        "n_s": 0.9638,
        "alpha_s": 0.0,
        "w0_fld": -1.0,
        "wa_fld": 0.0,
        "sigma8_m": 0.776779
    }

    halos = HaloCatalogue.from_abacus(node=0, phase=3000, redshift=0.575, boxsize=500)
    assert (halos.pos[:, :].min() >= 0) and halos.pos[:, :].max() <= 500
    assert halos.param_dict == {
        "omega_b": 0.02237,
        "omega_cdm": 0.1200,
        "h": 0.6736,
        "n_s": 0.9649,
        "alpha_s": 0.0,
        "w0_fld": -1.0,
        "wa_fld": 0.0,
        "sigma8_m": 0.807952
    }


def test__read_forge():
    halos = HaloCatalogue.from_forge(node=0, phase=0, snapshot=16, boxsize=1500.0)
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
