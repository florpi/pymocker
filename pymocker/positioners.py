from abc import ABC, abstractmethod
import numpy as np
import jax.numpy as jnp
from jax import random
from scipy import special as scipy_special
from pymocker.catalogues.halo import HaloCatalogue


class Positioner(ABC):
    def get_pos(
        self, halo_cat: HaloCatalogue, n_tracers: np.array, **kwargs
    ) -> np.array:
        """Abstract class describing a positioner

        Args:
            halo_cat (HaloCatalogue): halo catalogue to be populated
            n_tracers (np.array[int]): number of tracers to draw for each halo

        Returns:
            np.array[float]: positions of the desired tracers
        """
        pass


class IdentityPositioner(Positioner):
    def get_pos(self, halo_cat: HaloCatalogue, **kwargs) -> np.array:
        """Sample tracers only where halos are

        Args:
            halo_cat (HaloCatalogue): halo catalogue to be populated

        Returns:
            np.array[float]: positions of the desired tracers
        """
        return halo_cat.pos


class NFWPositioner(Positioner):
    def get_f(self, c: np.array) -> np.array:
        """Function to transform concentration into a useful variable
        for NFW profiles

        Args:
            c (np.array): array of halo concentrations

        Returns:
            np.array: transformed concentrations
        """
        return jnp.log(1 + c) - c / (1 + c)

    def apply_inverse_cdf(self, p: np.array, halo_cat: HaloCatalogue) -> np.array:
        """Compute the inverse CDF for an NFW profile. It is used to transform
        an array of uniformly sampled random variables into an NFW profile
               https://arxiv.org/pdf/1805.09550.pdf (Eq 6)

        Args:
            p (np.array): array of uniformly sampled random variables
            halo_cat (HaloCatalogue): halo catalogue to be populated

        Returns:
            np.array: random variables following an NFW profile
        """
        if halo_cat.concentration is None:
            raise ValueError(
                "The halo catalogue ```halo_cat``` must have concentration values for the NFW to work"
            )
        p *= self.get_f(c=halo_cat.concentration)
        return (
            -(1.0 / jnp.real(scipy_special.lambertw(-np.exp(-p - 1)))) - 1
        ) / halo_cat.concentration

    def sample_radial_positions(
        self,
        halo_cat: HaloCatalogue,
        n_tracers: int,
        seed: int = 42,
    ) -> np.array:
        """Sample radial positions following an NFW profile

        Args:
            halo_cat (HaloCatalogue): halo catalogue to sample from
            n_tracers (int): number of satellites to sample
            seed (int, optional): random seed (for reproducibility). Defaults to 42.

        Raises:
            ValueError: if radius not given

        Returns:
            np.array: radial distances from halo center for tracers, sampled
            from an NFW profile
        """
        if halo_cat.radius is None:
            raise ValueError(
                "The halo catalogue ```halo_cat``` must have radius values for the NFW to work"
            )
        key = random.PRNGKey(seed)
        uniforms = random.uniform(key, shape=(n_tracers,))
        scaled_radial_positions = self.apply_inverse_cdf(
            p=uniforms,
            halo_cat=halo_cat,
        )
        return scaled_radial_positions * halo_cat.radius

    def get_pos(self, halo_cat: HaloCatalogue, n_tracers: np.array) -> np.array:
        """Sample positions of tracers according to an NFW profile

        Args:
            halo_cat (HaloCatalogue): halo catalogue to be populated
            n_tracers (np.array[int]): number of tracers to draw for each halo

        Returns:
            np.array[float]: positions of the desired tracers
        """
        pass
