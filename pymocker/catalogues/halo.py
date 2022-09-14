from pymocker.catalogues.base import Catalogue
import numpy as np
from pathlib import Path
from typing import Optional


class HaloCatalogue(Catalogue):
    def __init__(
        self,
        pos: np.array,
        vel: np.array,
        mass: np.array,
        radius: Optional[np.array] = None,
        hid: Optional[np.array] = None,
        concentration: Optional[np.array] = None,
        boxsize: Optional[float] = None,
        redshift: Optional[float] = None,
        param_dict: Optional = None,
    ):
        """Catalogue of dark matter halos

        Args:
            pos (np.array): array of positions of size (N, 3)
            vel (np.array): array of velocities of size (N, 3)
            mass (np.array): array of halo masses
            hid (Optional[np.array], optional): halo id
            concentration (Optional[np.array], optional): halo concentration, used for NFW profiles. Defaults to None.
            boxsize (Optional[float], optional): size of the simulated box. Defaults to None.
            redshift (Optional[float], optional): redshift of simulated snapshot. Defaults to None.
            param_dict: Dictionary with parameters used to run the simulation
        """
        self.pos = pos
        self.vel = vel
        self.mass = mass
        self.radius = radius
        if hid is None:
            self.hid = np.arange(len(pos))
        else:
            self.hid = hid
        # TODO: Add option to go from halo id to halo idx through dictionary
        self.concentration = concentration
        self.boxsize = boxsize
        self.redshift = redshift
        self.param_dict = param_dict
        self.attrs_to_frame = [
            "mass",
        ]

    @classmethod
    def from_forge(
        cls,
        node: int = 0,
        snapshot: int = 20,
        phase: int = 0,
        boxsize: float = 500.0,
        min_n_particles: int = 100,
    ) -> "HaloCatalogue":
        import pymocker.catalogues.read_utils as ru

        seed = ru.seeds[phase]
        if boxsize == 1500.0:
            data_path = ru.NODES_GR_LARGE_DATA
        elif boxsize == 500.0:
            if node == 0:
                data_path = ru.NODE0_GR_DATA
            else:
                data_path = ru.NODES_GR_DATA
        else:
            raise ValueError(f"Boxsize {boxsize} does not exist")
        param_dict = ru.get_forge_params(node=node)
        for path_to_node in data_path.glob(f"L{int(boxsize)}*"):
            if path_to_node.name.startswith(
                f"L{int(boxsize)}_N{ru.n_particles[int(boxsize)]}_Seed_{seed}_Node_{str(node).zfill(3)}"
            ):
                data, redshift = ru.read_forge_groups(
                    path_to_node, snapshot=snapshot, min_n_particles=min_n_particles
                )
                pos = data[:, :3]
                vel = data[:, 3:6]
                mass = data[:, 6]
                radius = data[:, 7]
                original_idx = data[:, 8]
                return cls(
                    pos=pos,
                    vel=vel,
                    mass=mass,
                    radius=radius,
                    boxsize=boxsize,
                    redshift=redshift,
                    hid=original_idx,
                    param_dict=param_dict
                )
        raise ValueError("Data not found!")


    @classmethod
    def from_abacus(
        cls,
        node: int = 0,
        phase: int = 0,
        redshift: float = 0.575,
        boxsize: float = 2000.0,
        min_n_particles: int = 100,
    ) -> "HaloCatalogue":
        import pymocker.catalogues.read_utils as ru

        data = ru.read_abacus_groups(
            boxsize=boxsize, node=node, phase=phase,
            redshift=redshift, min_n_particles=min_n_particles
        )
        pos = data[:, :3]
        vel = data[:, 3:6]
        mass = data[:, 6]
        radius = data[:, 7]
        hid = data[:, 8]
        param_dict = ru.get_abacus_params(node=node)
        return cls(
            pos=pos,
            vel=vel,
            mass=mass,
            radius=radius,
            boxsize=boxsize,
            redshift=redshift,
            hid=hid,
            param_dict=param_dict
        )
        raise ValueError("Data not found!")
