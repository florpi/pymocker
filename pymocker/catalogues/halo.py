from pymocker.catalogues.base import Catalogue
import numpy as np
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
        self.attrs_to_frame = [
            "mass",
        ]

    def read_node(node: int = 0, snapshot=26, seed=2080, boxsize: int = 500):
        # 2080 or 4257
        boxsize = int(boxsize)
        if boxsize == 1500:
            fr_path = ru.NODES_FR_LARGE_DATA
        else:
            fr_path = ru.NODES_FR_DATA
        for path_to_node in fr_path.glob(f"L{boxsize}*"):
            if path_to_node.name.startswith(
                f"L{boxsize}_N{ru.n_particles[boxsize]}_Seed_{seed}_Node_{str(node).zfill(3)}"
            ):
                return read_groups(path=path_to_node, snapshot=snapshot)
        raise ValueError(f"{node} node not found")

    @classmethod
    def from_forge(
        cls,
        node: int = 0,
        snapshot: int = 20,
        box: int = 0,
        boxsize: float = 500.0,
        min_n_particles: int = 100,
    ) -> "HaloCatalogue":
        import pymocker.catalogues.read_utils as ru

        seed = ru.seeds[box]
        if boxsize == 1500.0:
            data_path = ru.NODES_GR_LARGE_DATA
        elif boxsize == 500.0:
            data_path = ru.NODES_GR_DATA
        else:
            raise ValueError(f"Boxsize {boxsize} does not exist")
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
                )
        raise ValueError("Data not found!")
