from operator import ge
from pymocker.catalogues.halo import HaloCatalogue
import numpy as np
import pandas as pd
import pytest


def test__read_abacus():
    hc = HaloCatalogue
    halos = hc.from_abacus(
        node=0,
        phase=0,
        redshift=0.575,
        boxsize=2000
    )
    assert((halos.pos[:,:].min() >= 0) and halos.pos[:,:].max() <= 2000)

    hc = HaloCatalogue
    halos = hc.from_abacus(
        node=0,
        phase=3000,
        redshift=0.575,
        boxsize=500
    )
    assert((halos.pos[:,:].min() >= 0) and halos.pos[:,:].max() <= 500)


