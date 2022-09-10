import numpy as np
import h5py

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
NODES_FR_LARGE_DATA = Path("/cosma7/data/dp004/bl267/Runs/CosmicEmulator_Large/")
NODES_GR_LARGE_DATA = Path("/cosma8/data/dp203/bl267/FORGE_LCDM/large_boxes/")

ABACUS_BASE = Path("/global/cfs/cdirs/desi/cosmosim/Abacus/")
ABACUS_SMALL = Path("/global/cfs/cdirs/desi/cosmosim/Abacus/small")

n_particles = {
        500: 1024,
        1500: 512,
}

def get_attrs_header(path, attrs):
    attrs_in_header = {}
    with h5py.File(path, "r") as fin:
        for attr in attrs:
            attrs_in_header[attr] = fin["Header"].attrs[attr]
    return attrs_in_header


def read_abacus_groups(boxsize, node, phase, redshift, min_n_particles=100):
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
    print(np.shape(data))
    return data


def read_forge_groups(path, snapshot, min_n_particles=100):
    # https://arepo-code.org/wp-content/userguide/snapshotformat.html
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
