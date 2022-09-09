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
        reconstructed_q = nfw_pos.apply_inverse_cdf(p, c)
        np.testing.assert_allclose(q, reconstructed_q, rtol=0.1)

    def test__sample_radial_positions(
        self,
        nfw_pos,
    ):
        c = 1.0
        r = nfw_pos.sample_scaled_radial_positions(
            halo_concentration=c,
            n_tracers=100,
        )
        # Test that we actually sample an NFW profile
        sorted_r = np.sort(r)
        estimated_cdf = 1.0 * np.arange(len(r)) / (len(r) - 1)
        estimated_cdf_at_r = ius(sorted_r, estimated_cdf)(r)
        true_cdf = (np.log(1.0 + r * c) - r * c / (1 + r * c)) / (
            np.log(1 + c) - c / (1 + c)
        )
        mask = (true_cdf > 0.01) & (estimated_cdf_at_r > 0.01)
        np.testing.assert_allclose(true_cdf[mask], estimated_cdf_at_r[mask], atol=0.1)

    def test__r_to_3d(self, nfw_pos):
        c = 1.0
        r = nfw_pos.sample_scaled_radial_positions(
            halo_concentration=c,
            n_tracers=100,
        )
        x, y, z = nfw_pos.convert_scaled_r_to_3d_pos(r, 1.0)
        r_recovered = np.sqrt(x**2 + y**2 + z**2)
        np.testing.assert_allclose(r, r_recovered, rtol=0.001)

    def test__get_pos(self, halo_cat, nfw_pos):
        # Test that each halo has sampled an NFW profile of the right
        # concentration
        n_tracers_per_halo = np.random.randint(low=100, high=300, size=(len(halo_cat),))
        halo_cat.concentration = np.random.random(size=(len(halo_cat),)) + 1.0
        halo_cat.radius = np.random.random(size=(len(halo_cat),)) + 10.0
        tracer_pos = nfw_pos.get_pos(
            halo_cat=halo_cat, n_tracers_per_halo=n_tracers_per_halo
        )
        n_t = 0
        for i in range(len(halo_cat) - 1):
            start_idx = n_t
            end_idx = n_t + n_tracers_per_halo[i + 1]
            relative_pos = tracer_pos[start_idx:end_idx] - halo_cat.pos[i]
            r = np.linalg.norm(relative_pos, axis=-1) / halo_cat.radius[i]
            sorted_r = np.sort(r)
            estimated_cdf = 1.0 * np.arange(len(r)) / (len(r) - 1)
            estimated_cdf_at_r = ius(sorted_r, estimated_cdf)(r)
            c = halo_cat.concentration[i]
            true_cdf = (np.log(1.0 + r * c) - r * c / (1 + r * c)) / (
                np.log(1 + c) - c / (1 + c)
            )
            mask = (true_cdf > 0.1) & (true_cdf < 0.9)
            n_t += n_tracers_per_halo[i]
        np.testing.assert_allclose(true_cdf[mask], estimated_cdf_at_r[mask], atol=0.1)
