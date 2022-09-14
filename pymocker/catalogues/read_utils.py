import numpy as np
import h5py
from typing import List, Dict, Tuple
from pathlib import Path
import os.path
import h5py
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

seeds = ["4257", "2080", "1391", "0333", "6241", "0720", "9156"]
sets = [1, 1, 2, 3, 3, 4, 4]
random_numbers = [54, 74, 40, 15, 22, 10, 82]
n_file_splits = 8
H_0 = 100.0

GRFR_DATA = Path("/cosma6/data/dp004/dc-arno1/CosmicEmulator/")
NDGP_DATA = Path("/cosma8/data/dp004/dc-hern1/CosmicEmulatorPlanck2018/")
NODES_FR_DATA = Path("/cosma6/data/dp004/dc-arno1/CosmicEmulatorNodes/")
NODES_GR_DATA = Path("/cosma8/data/dp203/bl267/FORGE_LCDM/small_boxes/")
NODE0_GR_DATA = Path("/madfs/data/dc-arno1/CosmicEmulatorNodes/")
NODES_FR_LARGE_DATA = Path("/cosma7/data/dp004/bl267/Runs/CosmicEmulator_Large/")
NODES_GR_LARGE_DATA = Path("/cosma8/data/dp203/bl267/FORGE_LCDM/large_boxes/")

ABACUS_BASE = Path("/global/cfs/cdirs/desi/cosmosim/Abacus/")
ABACUS_SMALL = Path("/global/cfs/cdirs/desi/cosmosim/Abacus/small")

n_particles = {
        500: 1024,
        1500: 512,
}

def read_abacus_groups(boxsize: float, node: int, phase: float, redshift: float, min_n_particles: int=100) -> np.array:
    """Read abacus data

    Args:
        boxsize (float): size of box to read, either 500. or 2000. 
        node (int): node in latin hypercube to read
        phase (float): phases of the initial conditions.
        redshift (float): redshift to read.
        min_n_particles (int, optional): Minimum number of particles per halo. Defaults to 100.

    Raises:
        ValueError: if boxsize is not 500 or 2000 
        ValueError: if data path does not exist 

    Returns:
        np.array: halo data 
    """
    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    if boxsize == 2000.0:
        data_path = Path(
            ABACUS_BASE,
            f"AbacusSummit_base_c{node:03}_ph{phase:03}/halos/z{redshift}/"
        )
    elif boxsize == 500.0:
        data_path = Path(
            ABACUS_SMALL,
            f"AbacusSummit_small_c{node:03}_ph{phase:03}/halos/z{redshift}/"
        )
    else:
        raise ValueError(f"Boxsize {boxsize} does not exist")

    if os.path.isdir(data_path):
        logger.info(f"Reading groups in {data_path}")
    else:
        raise ValueError(f"{data_path} not found")

    cat = CompaSOHaloCatalog(str(data_path),
        fields=['id', 'N', 'x_com', 'v_com', 'SO_radius'])

    particle_mass = 2109081520.453063  # in Msun / h
    halo_id = cat.halos['id']    
    n_particles = cat.halos['N']
    radius = cat.halos['SO_radius']
    mass = n_particles * particle_mass
    pos = cat.halos['x_com'].data + boxsize / 2
    vel = cat.halos['v_com'].data
    data = np.c_[pos, vel, mass, radius, halo_id]
    mask = n_particles >= min_n_particles
    data = data[mask]
    return data


def get_attrs_header(path: Path, attrs: List[str])->Dict:
    """Get attributes from header

    Args:
        path (Path): path to hdf5 file where the simulation is stored 
        attrs (List[str]): list of attributes to retrieve

    Returns:
        attrs_in_header (Dict): dictionary with the attributes
    """
    attrs_in_header = {}
    with h5py.File(path, "r") as fin:
        for attr in attrs:
            attrs_in_header[attr] = fin["Header"].attrs[attr]
    return attrs_in_header


def read_forge_groups(path: Path, snapshot: int, min_n_particles: int=100) -> Tuple[np.array, float]:
    """ Function to read data from forge group catalogues.
    For info about units see: https://arepo-code.org/wp-content/userguide/snapshotformat.html

    Args:
        path (Path): path to where group catalogues are stored .
        snapshot (int): snapshot to read.
        min_n_particles (int, optional): Minimum number of particles per halo. Defaults to 100.

    Returns:
        Tuple[np.array, float]: halo data and redshift 
    """
    logger.info(f"Reading groups in {path}")
    group_dir = f"groups_{str(snapshot).zfill(3)}"
    data = np.empty((0, 10)) 
    attrs_header = get_attrs_header(
        path / group_dir / f"fof_subhalo_tab_{str(snapshot).zfill(3)}.0.hdf5",
        ["NumFiles", "Redshift", "Omega0"],
    )
    redshift = attrs_header["Redshift"]
    a = 1.0 / (attrs_header["Redshift"] + 1.0)
    n_file_splits = attrs_header["NumFiles"]
    subhalo_ids, subhalo_vel_disps = [], []
    last_id = 0
    for i in range(n_file_splits):
        group_file = (
            path / group_dir / f"fof_subhalo_tab_{str(snapshot).zfill(3)}.{i}.hdf5"
        )
        with h5py.File(group_file, "r") as fin:
            n_particles = fin["Group"]["GroupLen"][:]
            pos = fin["Group"]["GroupPos"][:]  # comoving Mpc/h
            mass = fin["Group"]["Group_M_Crit200"][:] * 1.0e10  # M_\sun/h
            radius = fin["Group"]["Group_R_Crit200"][
                :
            ]  # comoving Mpc/h, are we sure it is comoving??
            vel = fin["Group"]["GroupVel"][:] / a  # km/s
            subhalo_id = fin["Group"]["GroupFirstSub"][:]
            veldisp = fin["Subhalo"]["SubhaloVelDisp"][:]  # km/s
            original_ids = np.array(range(last_id, last_id + len(n_particles)))
            last_id += len(n_particles)
            data_i = np.c_[pos, vel, mass, radius, original_ids, n_particles]
            data = np.append(data, data_i, axis=0)
            subhalo_ids += list(subhalo_id)
            subhalo_vel_disps += list(veldisp)
    vel_disp = np.array(subhalo_vel_disps)[subhalo_ids]
    data = np.c_[data, vel_disp]
    data = data[data[:, -2] >= min_n_particles]
    return data, redshift

def get_forge_params(node: int)->Dict[str, float]:
    """Get the parameters used to run the simulation

    Args:
        node (int): node to read 

    Returns:
        Dict[str, float]: dictionary of parameters and their values 
    """
    import pandas as pd
    data_dir = Path('/cosma6/data/dp004/dc-arno1/CosmicEmulatorNodes/')
    param_df = pd.read_csv(data_dir / 'Nodes_Omm-S8-h-fR0-sigma8-As-B0_LHCrandommaximin_Seed1_Nodes50_Dim4_AddFidTrue_extended.dat',
                delimiter=r"\s+", skipfooter=2)
    param_df = param_df.drop(columns=['A_s'])
    param_df = param_df.apply(pd.to_numeric)
    return param_df.iloc[node].to_dict()


def get_abacus_params(node: int)->Dict[str, float]:
    """Get the parameters used to run the simulation

    Args:
        node (int): node to read 

    Returns:
        Dict[str, float]: dictionary of parameters and their values 
    """
    import pandas as pd
    data_dir = Path('/global/homes/e/epaillas/data/ds_desi/')
    param_df = pd.read_csv(data_dir / 'AbacusSummit_cosmologies.csv',
                delimiter=",", skipinitialspace=True)
    param_df.columns = param_df.columns.str.strip()
    param_df = param_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    idx = param_df.index[param_df['root'] == f'abacus_cosm{node:03}'][0]
    param_df = param_df.drop(columns=['A_s', 'N_ur', 'N_ncdm',
        'omega_ncdm', 'sigma8_cb', 'notes', 'root'])
    return param_df.to_dict('records')[idx]
