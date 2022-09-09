from pymocker.catalogues import HaloCatalogue
from pymocker.populator import Populator
from pymocker.galaxy import VanillaGalaxy
from pymocker.sampler import Sampler
from pymocker.occupation import Zheng07Centrals, Zheng07Sats
from pymocker.positioners import IdentityPositioner, NFWPositioner

halo_cat = HaloCatalogue.from_forge()
central_sampler = Sampler(
    occupation=Zheng07Centrals(),
    positioner=IdentityPositioner(),
)
sat_sampler = Sampler(
    occupation=Zheng07Sats(),
    positioner=NFWPositioner(),
)
populator = Populator(
    central_sampler=central_sampler,
    satellite_sampler=sat_sampler,
)
galaxy = VanillaGalaxy()
gal_cat = populator(halo_cat=halo_cat, galaxy=galaxy)
