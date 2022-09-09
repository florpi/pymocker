from abc import ABC, abstractmethod
import numpy as np
import jax.numpy as jnp
from jax import random
from scipy import special as scipy_special
from typing import Optional, Tuple
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

    def apply_inverse_cdf(self, p: np.array, halo_concentration: np.array) -> np.array:
        """Compute the inverse CDF for an NFW profile. It is used to transform
        an array of uniformly sampled random variables into an NFW profile
               https://arxiv.org/pdf/1805.09550.pdf (Eq 6)

        Args:
            p (np.array): array of uniformly sampled random variables
            halo_cat (HaloCatalogue): halo catalogue to be populated

        Returns:
            np.array: random variables following an NFW profile
        """
        p *= self.get_f(c=halo_concentration)
        return (
            -(1.0 / jnp.real(scipy_special.lambertw(-np.exp(-p - 1)))) - 1
        ) / halo_concentration

    def sample_scaled_radial_positions(
        self,
        halo_concentration: np.array,
        n_tracers: int,
        seed: int = 42,
    ) -> np.array:
        """Sample radial positions following an NFW profile

        Args:
            halo_cat (HaloCatalogue): halo catalogue to sample from
            n_tracers (int): number of satellites to sample
            seed (int, optional): random seed (for reproducibility). Defaults to 42.

        Returns:
            np.array: radial distances from halo center for tracers, sampled
            from an NFW profile, in units of the halo radius
        """
        key = random.PRNGKey(seed)
        uniforms = random.uniform(key, shape=(n_tracers,))
        scaled_radial_positions = self.apply_inverse_cdf(
            p=uniforms,
            halo_concentration=halo_concentration,
        )
        return scaled_radial_positions

    def convert_scaled_r_to_3d_pos(
        self,
        r: np.array,
        halo_radius: np.array,
        seed: Optional[int] = 42,
    ) -> Tuple[np.array, np.array, np.array]:
        """Convert scaled r for sampled galaxies into 3D coordinates

        Args:
            r (np.array): distance from halo centre
            seed (Optional[int], optional): random seed. Defaults to 42.

        Returns:
            np.array: x,y,z coordinates
        """
        r *= halo_radius
        key = random.PRNGKey(seed)
        cos_t = 2.0 * random.uniform(key, shape=(len(r),)) - 1.0
        key, subkey = random.split(key)
        phi = 2 * jnp.pi * random.uniform(subkey, shape=(len(r),))
        sin_t = jnp.sqrt((1.0 - cos_t * cos_t))
        x = r * sin_t * jnp.cos(phi)
        y = r * sin_t * jnp.sin(phi)
        z = r * cos_t
        return x, y, z

    def get_pos(
        self, halo_cat: HaloCatalogue, n_tracers_per_halo: np.array, seed: int = 42
    ) -> np.array:
        """Sample positions of tracers according to an NFW profile

        Args:
            halo_cat (HaloCatalogue): halo catalogue to be populated
            n_tracers_per_halo (np.array[int]): number of tracers to draw for each halo

        Returns:
            np.array[float]: positions of the desired tracers
        """
        halo_concentration = jnp.repeat(
            halo_cat.concentration,
            n_tracers_per_halo,
        )
        r = self.sample_scaled_radial_positions(
            halo_concentration=halo_concentration,
            n_tracers=jnp.sum(n_tracers_per_halo),
            seed=seed,
        )
        halo_radius = jnp.repeat(
            halo_cat.radius,
            n_tracers_per_halo,
        )
        (delta_x, delta_y, delta_z,) = self.convert_scaled_r_to_3d_pos(
            r=r,
            halo_radius=halo_radius,
            seed=seed,
        )
        halo_pos = jnp.repeat(
            halo_cat.pos,
            n_tracers_per_halo,
            axis=0,
        )
        return halo_pos + np.vstack((delta_x, delta_y, delta_z)).T
