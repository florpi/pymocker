import numpy as np
from pymocker.positioners import IdentityPositioner


class TestCentralPositioners:
    def test__central_identity(self, halo_cat):
        central_positioner = IdentityPositioner()
        central_pos = central_positioner.get_pos(
            halo_cat=halo_cat,
        )
        np.testing.assert_equal(central_pos, halo_cat.pos)
