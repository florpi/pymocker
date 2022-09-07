import numpy as np
from pymocker.positioners import IdentityPositioner


class TestCentralPositioners:
    def test__central_identity(self, halo_cat):
        central_positioner = IdentityPositioner()
        central_pos = central_positioner.get_pos(
            halo_cat=halo_cat,
        )
        np.testing.assert_equal(central_pos, halo_cat.pos)


class TestSatellitePositioners:
    def test__inverse_cdf(
        self,
        halo_cat,
        nfw_pos,
    ):
        q = np.random.random(size=(len(halo_cat),))
        c = halo_cat.concentration
        p = (np.log(1.0 + q * c) - q * c / (1 + q * c)) / (np.log(1 + c) - c / (1 + c))
        reconstructed_q = nfw_pos.apply_inverse_cdf(p, halo_cat=halo_cat)
        np.testing.assert_allclose(q, reconstructed_q, rtol=0.1)

    def test__sample_radial_positions(
        self,
        halo_cat,
        nfw_pos,
    ):
        # Test that we actually sample an NFW profile
        pass

    """
    def test__satellite(self, halo_cat):
        n_tracers = np.random.randint(low=0, high=100, size=(len(halo_cat),))
    """
