from abc import ABC, abstractmethod
from functools import partial
from pymocker.galaxy import Galaxy, VanillaGalaxy
import numpy as np
from jax.scipy import special
from jax import random, jit
import jax.numpy as jnp


class Occupation(ABC):
    @abstractmethod
    def get_mean_occ(self, halo_property: np.array, galaxy: Galaxy) -> np.array:
        pass

    @abstractmethod
    def __call__(self, halo_cat: "HaloCatalogue") -> np.array:
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

    # @partial(jit, static_argnums=(0,))
    def __call__(
        self, halo_cat: "HaloCatalogue", galaxy: VanillaGalaxy, seed: int = 42
    ) -> np.array:
        """Sample number of galaxies per halo

        Args:
            halo_cat (HaloCatalogue): halo catalogue to sample from
            galaxy (VanillaGalaxy): galaxy parameters
            seed (int, optional): random seed. Defaults to 42.

        Returns:
            np.array: number of galaxies per halo
        """
        key = random.PRNGKey(seed)
        mean_n = self.get_mean_occ(
            halo_mass=halo_cat.mass,
            galaxy=galaxy,
        )
        randoms = random.uniform(key, shape=(len(mean_n),))
        return (mean_n > randoms).astype(int)


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
        return n_centrals * self.lambda_sat(halo_mass=halo_mass, galaxy=galaxy)

    def __call__(
        self,
        halo_cat: "HaloCatalogue",
        galaxy: VanillaGalaxy,
        seed: int = 42,
    ) -> np.array:
        """Sample number of satellite galaxies per halo

        Args:
            halo_cat (HaloCatalogue): halo catalogue to sample from.
            galaxy (VanillaGalaxy): galaxy parameters.
            seed (int, optional): random seed. Defaults to 42.

        Returns:
            np.array: array of number of satellites per halo
        """
        key = random.PRNGKey(seed)
        return random.poisson(
            key, self.lambda_sat(halo_mass=halo_cat.mass, galaxy=galaxy)
        )

class AemulusSatellites(Zheng07Sats):
    def lambda_sat(self, halo_mass: np.array, galaxy: VanillaGalaxy) -> np.array:
        """
        https://iopscience.iop.org/article/10.3847/1538-4357/ab0d7b/pdf Eq7

        Args:
            halo_mass (np.array): mass of each halo to populate
            galaxy (VanillaGalaxy): instance of the HOD parameters to use

        Returns:
            np.array: lambda value
        """
        return (halo_mass / galaxy.M_sat) ** galaxy.alpha * np.exp(
            -galaxy.M_cut / halo_mass
        )