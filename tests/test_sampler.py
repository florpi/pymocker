from pymocker.sampler import Sampler
from pymocker.occupation import Occupation
from pymocker.positioners import IdentityPositioner


class MockOccupation(Occupation):
    def get_mean_occ(
        self,
        halo_mass,
        **kwargs,
    ):
        return 2.0 * halo_mass


def test__unity_sampler(halo_cat):
    sampler = Sampler(
        occupation=MockOccupation(),
        positioner=IdentityPositioner(),
    )
    galaxy_cat = sampler(halo_cat=halo_cat)
    # check that mean occupation is right

    # check that positions are those of halos
