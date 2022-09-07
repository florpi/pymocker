from abc import ABC, abstractmethod
from pymocker.galaxy import Galaxy, VanillaGalaxy
import numpy as np
from jax.scipy import special
import jax.numpy as jnp


class Occupation(ABC):
    @abstractmethod
    def get_mean_occ(self, halo_property, galaxy: Galaxy) -> np.array:
        pass


class Zheng07Centrals(Occupation):
    def get_mean_occ(
        self,
        halo_mass: np.array,
        galaxy: VanillaGalaxy,
    ) -> np.array:
        """Compute the number of central galaxies as a function of host halo mass.

        Args:
            halo_mass (np.array): mass of each halo to populate
            galaxy (VanillaGalaxy): instance of the HOD parameters to use

        Equation (G2) at arXiv:1811.09504

        Returns:
            np.array:
        """
        return 0.5 * special.erfc(
            jnp.log10(galaxy.M_min / halo_mass) / galaxy.sigma_log_M
        )


class Zheng07Sats(Occupation):
    def lambda_sat(self, halo_mass: np.array, galaxy: VanillaGalaxy) -> np.array:
        """Compute satellite lambda for (G4) in arXiv:1811.09504

        Args:
            halo_mass (np.array): mass of each halo to populate
            galaxy (VanillaGalaxy): instance of the HOD parameters to use

        Returns:
            np.array: lambda value
        """
        return jnp.where(
            halo_mass > galaxy.kappa * galaxy.M_min,
            ((halo_mass - galaxy.kappa * galaxy.M_min) / galaxy.M1) ** galaxy.alpha,
            0.0,
        )

    def get_mean_occ(
        self,
        halo_mass: np.array,
        n_centrals: np.array,
        galaxy: VanillaGalaxy,
    ) -> np.array:
        """Compute the mean number of satellite galaxies as a function of host
        halo mass.

        Equation (G4) at arXiv:1811.09504

        Args:
            halo_mass (np.array): mass of each halo to populate
            n_centrals (np.array): number of central galaxies in that
            host halo mass
            galaxy: object contaiing galaxy parameters

        Returns:
            np.array: mean number of galaxies
        """
        return jnp.where(
            halo_mass > galaxy.kappa * galaxy.M_min,
            n_centrals * self.lambda_sat(halo_mass=halo_mass, galaxy=galaxy),
            0.0,
        )
