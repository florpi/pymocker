import numpy as np
from pycorr import TwoPointCorrelationFunction, setup_logging
from pymocker.catalogues import HaloCatalogue
from pymocker.populator import Populator
from pymocker.galaxy import VanillaGalaxy, AemulusGalaxy
from pymocker.sampler import Sampler
from pymocker.occupation import Zheng07Centrals, Zheng07Sats
from pymocker.positioners import IdentityPositioner, NFWPositioner, ParticlePositioner
from hmd.concentration import diemer15
from astropy.cosmology import Planck18
import matplotlib.pyplot as plt


setup_logging()

halo_cat = HaloCatalogue.from_abacus(
    node=0, phase=3000, redshift=0.2, boxsize=500,
    include_particles=True)

# halo_cat.concentration = diemer15(
#     prim_haloprop=halo_cat.mass,
#     redshift=0.575,
#     cosmology=Planck18,
#     sigma8=0.8,
#     allow_tabulation=True
# )

central_sampler = Sampler(
    occupation=Zheng07Centrals(),
    positioner=IdentityPositioner(),
)
sat_sampler = Sampler(
    occupation=Zheng07Sats(),
    positioner=ParticlePositioner(),
)
populator = Populator(
    central_sampler=central_sampler,
    satellite_sampler=sat_sampler,
)
galaxy = VanillaGalaxy(
    log_M_min = 13.,
    sigma_log_M = 0.5,
    kappa = 0.5,
    log_M1=13.,
    alpha=1.,
)
gal_cat = populator(halo_cat=halo_cat, galaxy=galaxy)

# edges = (np.logspace(-1, np.log10(50), 51),
#     np.linspace(-1., 1., 201))

# gals_result = TwoPointCorrelationFunction(
#     'smu', edges, data_positions1=gal_cat.pos.T,
#      engine='corrfunc', boxsize=gal_cat.boxsize,
#      los='z', nthreads=256
# )
# halos_result = TwoPointCorrelationFunction(
#     'smu', edges, data_positions1=halo_cat.pos.T,
#      engine='corrfunc', boxsize=gal_cat.boxsize,
#      los='z', nthreads=256
# )


# fig, ax = plt.subplots()
# s, multipoles_gals = gals_result(ells=(0, 2), return_sep=True)
# s, multipoles_halos = halos_result(ells=(0, 2), return_sep=True)

# ax.loglog(s, multipoles_gals[0], label='galaxies')
# ax.loglog(s, multipoles_halos[0], label='halos')

# ax.set_xlabel(r'$s\,[h^{-1}{\rm Mpc}]$')
# ax.set_ylabel(r'$\xi_0(s)$')
# ax.legend()

# plt.savefig('test.png', dpi=300)
