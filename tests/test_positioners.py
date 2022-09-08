import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius
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
        c = 1.0
        halo_cat.concentration = c
        r = nfw_pos.sample_scaled_radial_positions(
            halo_cat=halo_cat,
            n_tracers=100,
        )
        # Test that we actually sample an NFW profile
        sorted_r = np.sort(r)
        estimated_cdf = 1.0 * np.arange(len(r)) / (len(r) - 1)
        estimated_cdf_at_r = ius(sorted_r, estimated_cdf)(r)
        true_cdf = (np.log(1.0 + r * c) - r * c / (1 + r * c)) / (
            np.log(1 + c) - c / (1 + c)
        )
        np.testing.assert_allclose(true_cdf, estimated_cdf_at_r, atol=0.05)

    def test_r_to_3d(self, halo_cat, nfw_pos):
        c = 1.0
        halo_cat.concentration = c
        halo_cat.radius = 1.0
        r = nfw_pos.sample_scaled_radial_positions(
            halo_cat=halo_cat,
            n_tracers=100,
        )
        x, y, z = nfw_pos.convert_r_to_3d_pos(r, halo_cat)
        r_recovered = np.sqrt(x**2 + y**2 + z**2)
        np.testing.assert_allclose(r, r_recovered, rtol=0.001)
