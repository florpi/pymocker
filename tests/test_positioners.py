import numpy as np
from pymocker.positioners import IdentityPositioner


def test__central_identity(halo_cat):
    central_positioner = IdentityPositioner()
    n_halos = len(halo_cat)
    n_tracers = np.random.randint(low=0, high=5, size=(n_halos,))
    central_pos = central_positioner.get_pos(
        halo_cat=halo_cat,
    )
    np.testing.assert_equal(central_pos, halo_cat.pos)
