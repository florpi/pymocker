from tkinter import N
from pymocker.catalogues import HaloCatalogue, GalaxyCatalogue
import jax.numpy as jnp


class Sampler:
    def __init__(
        self,
        occupation: "Occupation",
        positioner: "Positioner",
    ):
        self.occupation = occupation
        self.positioner = positioner

    def __call__(
        self, halo_cat: HaloCatalogue, galaxy: "Galaxy", gal_type: str
    ) -> GalaxyCatalogue:
        n_tracers_per_halo = self.occupation(
            halo_cat=halo_cat,
            galaxy=galaxy,
        )
        pos = self.positioner.get_pos(
            halo_cat=halo_cat,
            n_tracers_per_halo=n_tracers_per_halo,
        )
        host_id = jnp.repeat(halo_cat.hid, n_tracers_per_halo)
        # TODO: add vel
        return GalaxyCatalogue(
            pos=pos,
            vel=None,
            host_id=host_id,
            gal_type=[gal_type] * len(halo_cat),
            boxsize=halo_cat.boxsize,
            redshift=halo_cat.redshift,
        )
