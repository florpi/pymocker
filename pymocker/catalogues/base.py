from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path
import pandas as pd
import numpy as np

class Catalogue(ABC):
    def __len__(self,) -> int:
        """Get number of objects in catalogue

        Returns:
            int: number of objects
        """
        return len(self.pos)

    def __add__(self, catalogue: "Catalogue",) -> "Catalogue":
        """Add the content of another catalogue

        Args:
            catalogue (Catalogue): catalogue to be added

        Returns:
            Catalogue: combined catalogue 
        """
        pos = np.vstack((self.pos, catalogue.pos))
        vel = np.vstack((self.vel, catalogue.vel))
        dict_attrs = {
            attr: np.concatenate((getattr(self, attr), getattr(catalogue, attr)))
            for attr in self.attrs_to_frame
            if getattr(self, attr) is not None
        }
        dict_attrs["pos"] = pos
        dict_attrs["vel"] = vel
        return self.__class__(
            **dict_attrs,
            redshift=self.redshift,
            boxsize=self.boxsize,
        )

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        boxsize: Optional[float] = None,
        redshift: Optional[float] = None,
    )->"Catalogue":
        """Generate catalogue from pandas dataframe

        Args:
            df (pd.DataFrame): pandas dataframe 
            boxsize (Optional[float]): box size of the simulation (use same units as positions!)
            redshift (Optional[float], optional): redshift of the simulation snapshot. Defaults to None.

        Returns:
            Catalogue: catalogue
        """
        pos = df[["x", "y", "z"]].values
        vel = df[["v_x", "v_y", "v_z"]].values
        cols_1d = [
            c for c in df.columns if c not in ("x", "y", "z", "v_x", "v_y", "v_z")
        ]
        df_dict = {col: df[col].values for col in cols_1d}
        return cls(
            pos=pos,
            vel=vel,
            boxsize=boxsize,
            redshift=redshift,
            **df_dict,
        )

    @classmethod
    def from_csv(
        cls,
        path: Path,
        boxsize: Optional[float] = None,
        redshift: Optional[float] = None,
    )->"Catalogue":
        """Generate catalogue from csv file

        Args:
            path (Path): path to csv
            boxsize (Optional[float]): size of simulation box
            redshift (Optional[float], optional): redshift of the simulation snapshot. Defaults to None.

        Returns:
            Catalogue: _description_
        """
        df = pd.read_csv(path)
        return cls.from_frame(
            df=df, boxsize=boxsize, redshift=redshift,
        )

    def to_frame(self,)->pd.DataFrame:
        """Convert catalogue into a dataframe

        Returns:
            pd.DataFrame: dataframe with the catalogue's data
        """
        x, y, z = self.pos.T
        v_x, v_y, v_z = self.vel.T
        dict_attrs = {
            attr: getattr(self, attr)
            for attr in self.attrs_to_frame
            if getattr(self, attr) is not None
        }
        dict_attrs["x"] = x
        dict_attrs["y"] = y
        dict_attrs["z"] = z
        dict_attrs["v_x"] = v_x
        dict_attrs["v_y"] = v_y
        dict_attrs["v_z"] = v_z
        return pd.DataFrame(dict_attrs)