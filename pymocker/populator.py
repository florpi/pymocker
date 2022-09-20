from pymocker.catalogues import HaloCatalogue, GalaxyCatalogue

class Populator:
    def __init__(
        self,
        central_sampler: "Sampler",
        satellite_sampler: "Sampler",
    ):
        self.central_sampler = central_sampler
        self.satellite_sampler = satellite_sampler

    def __call__(
        self,
        halo_cat: HaloCatalogue,
        galaxy: "Galaxy",
    ) -> GalaxyCatalogue:
        central_cat = self.central_sampler(
            halo_cat=halo_cat, galaxy=galaxy, gal_type='central',
        )
        if self.satellite_sampler is not None:
            satellite_cat = self.satellite_sampler(
                halo_cat=halo_cat, galaxy=galaxy, gal_type='satellite',
            )
            return central_cat + satellite_cat
        return central_cat
