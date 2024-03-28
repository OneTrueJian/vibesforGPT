"""Trajectory File I/O"""

import json
import multiprocessing
import os
import shutil
from pathlib import Path

import numpy as np
import xarray as xr
from ase import Atoms, units
from ase.calculators.calculator import PropertyNotImplementedError

from vibes import __version__ as version
from vibes import io, keys, son
from vibes.filenames import filenames
from vibes.helpers import warn
from vibes.helpers.converters import (
    dict2atoms,
    dict2json,
    results2dict,
    results2singlepoint,
)
from vibes.helpers.hash import hash_file
from vibes.helpers.utils import progressbar

from .utils import Timer, talk


def step2file(atoms, calculator=None, file=filenames.trajectory, metadata={}):
    """Save the current step

    Args:
        atoms: The structure at the current step
        calculator: The ASE Calculator for the current run
        file: Path to file to append the current step to
        metadata: the metadata for the calculation, store to atoms.info if possible
    """

    dct = {}
    if metadata:
        for key, val in metadata.items():
            if key in atoms.info and atoms.info[key] == val:
                continue
            if key not in atoms.info:
                atoms.info[key] = val
            else:
                atoms.info.update({"metadata": metadata})
                break

    dct.update(results2dict(atoms, calculator))

    son.dump(dct, file, dumper=dict2json)


def metadata2file(metadata, file="metadata.son"):
    """save metadata to file

    Args:
        metadata: the metadata to save
        file: filepath to the output file
    """

    if metadata is None:
        metadata = {}

    son.dump({**metadata, "vibes": {"version": version}}, file, is_metadata=True)


def write(trajectory, file=filenames.trajectory):
    """Write to son file

    Args:
        file: path to trajecotry son or netcdf file
    """
    from .dataset import get_trajectory_dataset

    timer = Timer(f"Write trajectory to {file}")

    if Path(file).suffix == ".nc":
        dataset = get_trajectory_dataset(trajectory, metadata=True)
        dataset.to_netcdf(file)
        timer()
        return True

    temp_file = "temp.son"

    # check for file and make backup
    if os.path.exists(file):
        outfile = f"{file}.bak"
        shutil.copy(file, outfile)
        talk(f".. {file} copied to {outfile}")

    metadata2file(trajectory.metadata, temp_file)

    talk(f"Write to {temp_file}")
    for elem in progressbar(trajectory):
        son.dump(results2dict(elem), temp_file)

    shutil.move(temp_file, file)

    timer()


def _create_atoms(
    obj: dict,
    pre_atoms_dict: dict,
    pre_calc_dict: dict,
    metadata: dict,
    single_point_calculator: bool,
):
    """create Atoms object from dictionaries"""
    atoms_dict = {**pre_atoms_dict, **obj["atoms"]}

    # remember that the results need to go to a dedicated results dict in calc
    calc_dict = {**pre_calc_dict, "results": obj["calculator"]}

    atoms = dict2atoms(
        atoms_dict,
        calculator_dict=calc_dict,
        single_point_calculator=single_point_calculator,
    )

    # info
    if "MD" in metadata:
        if "dt" in atoms.info:
            atoms.info["dt_fs"] = atoms.info["dt"] / metadata["MD"]["fs"]
    elif "info" in obj:
        info = obj["info"]
        atoms.info.update(info)

    # compatibility with older trajectories
    if "MD" in obj:
        atoms.info.update(obj["MD"])

    # preserve metadata
    if "metadata" in obj:
        atoms.info.update({"metadata": obj["metadata"]})

    return atoms


def _map_create_atoms(pre_trajectory, **kwargs):
    workers = []
    pool = multiprocessing.Pool()
    for obj in pre_trajectory:
        workers.append(pool.apply_async(_create_atoms, args=(obj,), kwds=kwargs))
    pool.close()
    pool.join()

    result = [worker.get() for worker in workers]
    return result


def reader(
    file=filenames.trajectory,
    get_metadata=False,
    fc_file=None,
    single_point_calculator=True,
    verbose=True,
):
    """Convert information in file to Trajectory

    Args:
        file: Trajectory file to read the structures from
        get_metadata: If True return the metadata
        fc_file: force constants file
        single_point_calculator: return calculators as SinglePointCalculator
        verbose: If True print more information to the screen

    Returns:
        trajectory: The trajectory from the file
        metadata: The metadata for the trajectory
    """
    from vibes.trajectory.trajectory import Trajectory

    timer = Timer(f"Parse `{file}`", verbose=verbose)

    if Path(file).suffix == ".nc":
        trajectory = read_netcdf(file)
        if fc_file:
            fc = io.parse_force_constants(fc_file, two_dim=False)
            trajectory.set_force_constants(fc)

        timer()

        return trajectory

    metadata, pre_trajectory = son.load(file, verbose=verbose)
    timer()

    # legacy of trajectory.yaml
    if metadata is None:
        msg = f"metadata in {file} appears to be empty, assume old convention w/o === "
        msg += f"was used. Let's see"
        warn(msg, level=1)
        metadata = pre_trajectory.pop(0)

    pre_calc_dict = metadata["calculator"]
    pre_atoms_dict = metadata["atoms"]

    if "numbers" in pre_atoms_dict and "symbols" in pre_atoms_dict:
        del pre_atoms_dict["symbols"]

    if not pre_trajectory:
        if get_metadata:
            talk(".. trajectory empty, return (Trajectory([]), metadata)")
            return Trajectory([]), metadata
        talk(".. trajectory empty, return Trajectory([])")
        return Trajectory([])

    timer2 = Timer(".. create atoms", verbose=verbose)
    if "MD" in metadata:
        map_metadata = {"MD": metadata["MD"]}
    else:
        map_metadata = {}

    kw = {
        "pre_atoms_dict": pre_atoms_dict,
        "pre_calc_dict": pre_calc_dict,
        "metadata": map_metadata,
        "single_point_calculator": single_point_calculator,
    }
    list = _map_create_atoms(pre_trajectory, **kw)
    trajectory = Trajectory(list, metadata=metadata)
    timer2()

    timer3 = Timer(".. set raw hash", verbose=verbose)
    trajectory.hash_raw = hash_file(file)
    timer3()

    timer("done")

    if fc_file:
        fc = io.parse_force_constants(fc_file, two_dim=False)
        trajectory.set_force_constants(fc)

    if get_metadata:
        return trajectory, metadata

    return trajectory


def to_tdep(trajectory, folder=".", skip=1):
    """Convert to TDEP infiles for direct processing

    Args:
        folder: Directory to store tdep files
        skip: Number of structures to skip
    """
    from contextlib import ExitStack
    from pathlib import Path

    folder = Path(folder)
    folder.mkdir(exist_ok=True)

    talk(f"Write tdep input files to {folder}:")

    # meta
    n_atoms = len(trajectory[0])
    n_steps = len(trajectory) - skip
    try:
        dt = trajectory.metadata["MD"]["timestep"] / trajectory.metadata["MD"]["fs"]
        T0 = trajectory.metadata["MD"]["temperature"] / units.kB
    except KeyError:
        dt = 1.0
        T0 = 0

    lines = [f"{n_atoms}", f"{n_steps}", f"{dt}", f"{T0}"]

    file = folder / "infile.meta"

    with file.open("w") as fo:
        fo.write("\n".join(lines))
        talk(f".. {file} written.")

    # supercell and fake unit cell
    write_settings = {"format": "vasp", "direct": True, "vasp5": True}
    if trajectory.primitive:
        file = folder / "infile.ucposcar"
        trajectory.primitive.write(str(file), **write_settings)
        talk(f".. {file} written.")
    if trajectory.supercell:
        file = folder / "infile.ssposcar"
        trajectory.supercell.write(str(file), **write_settings)
        talk(f".. {file} written.")

    with ExitStack() as stack:
        pdir = folder / "infile.positions"
        fdir = folder / "infile.forces"
        sdir = folder / "infile.stat"
        fp = stack.enter_context(pdir.open("w"))
        ff = stack.enter_context(fdir.open("w"))
        fs = stack.enter_context(sdir.open("w"))

        for ii, atoms in enumerate(trajectory[skip:]):
            # stress and pressure in GPa
            try:
                stress = atoms.get_stress(voigt=True) / units.GPa
                pressure = -1 / 3 * sum(stress[:3])
            except PropertyNotImplementedError:
                stress = np.zeros(6)
                pressure = 0.0
            e_tot = atoms.get_total_energy()
            e_kin = atoms.get_kinetic_energy()
            e_pot = e_tot - e_kin
            temp = atoms.get_temperature()

            for spos in atoms.get_scaled_positions():
                fp.write("{} {} {}\n".format(*spos))

            for force in atoms.get_forces():
                ff.write("{} {} {}\n".format(*force))

            stat = (
                f"{ii:5d} {ii*dt:10.2f} {e_tot:20.8f} {e_pot:20.8f} "
                f"{e_kin:20.15f} {temp:20.15f} {pressure:20.15f} "
            )
            stat += " ".join([str(s) for s in stress])

            fs.write(f"{stat}\n")

    talk(f".. {sdir} written.")
    talk(f".. {pdir} written.")
    talk(f".. {fdir} written.")


def to_db(trajectory, database):
    """Write vibes trajectory as ase database

    Always creates a new database. Database type
    is always inferred from the file. Metadata
    is carried over to the ase database.

    Please be aware that ase.db indices start from
    1, not from 0 as usual.

    Args:
        trajectory: Trajectory instance
        database: Filename or address of database

    """
    from ase.db import connect

    timer = Timer(f"Save as ase database {database}")

    with connect(database, append=False) as db:
        talk("write db")
        for atoms in progressbar(trajectory):
            db.write(atoms, data={"info": atoms.info})

    # metadata can only be written *after* the database exists
    with connect(database) as db:
        db.metadata = trajectory.metadata

    timer("done")


def to_ase_trajectory(trajectory, outfile):
    """Write vibes trajectory as ase trajectory

    Will overwrite outfile if it exists.

    Args:
        trajectory: Trajectory instance
        outfile: Filename of trajectory

    """
    from ase.io import Trajectory

    timer = Timer(f"Save as ase trajectory {outfile}")

    traj = Trajectory(outfile, mode="w")

    traj.description = trajectory.metadata

    talk("write ase trajectory")
    for atoms in progressbar(trajectory):
        traj.write(atoms)

    timer("done")


def read_netcdf(file=filenames.trajectory_dataset):
    """read `trajectory.nc` and return Trajectory"""
    DS = xr.open_dataset(file)

    return parse_dataset(DS)


def parse_dataset(dataset: xr.Dataset) -> list:
    """turn xarray.Datset into Trajectory"""
    from vibes.trajectory.trajectory import Trajectory

    DS = dataset
    attrs = DS.attrs

    # check mandatory keys
    assert keys.reference_atoms in attrs
    assert "positions" in DS
    assert "velocities" in DS
    assert "forces" in DS
    assert keys.energy_potential in DS

    atoms_dict = json.loads(attrs[keys.reference_atoms])

    # metadata
    metadata = None
    if keys.metadata in attrs:
        metadata = json.loads(attrs[keys.metadata])

    # popping `velocities` is obsolete if
    # https://gitlab.com/ase/ase/merge_requests/1563
    # is accepted
    velocities = atoms_dict.pop(keys.velocities, None)

    ref_atoms = Atoms(**atoms_dict)

    if velocities is not None:
        ref_atoms.set_velocities(velocities)

    positions = DS.positions.data
    velocities = DS.velocities.data
    forces = DS.forces.data
    potential_energy = DS[keys.energy_potential].data

    if keys.cell in DS:
        cells = DS.cell.data
    else:
        cells = [None for _ in positions]

    if keys.stress_potential in DS:
        stress = DS[keys.stress_potential].data
    else:
        stress = len(positions) * [None]

    if keys.stresses_potential in DS:
        stresses = DS[keys.stresses_potential].data
    else:
        stresses = None

    if keys.virials in DS:
        virials = DS[keys.virials].data
    else:
        virials = None

    trajectory_list = []
    properties = (positions, cells, velocities, forces, potential_energy, stress)
    for ii, (p, c, v, f, e, s) in enumerate(zip(*properties)):
        atoms = ref_atoms.copy()
        if c is not None:
            atoms.set_cell(c)
        atoms.set_positions(p)
        atoms.set_velocities(v)

        results = {keys.energy: e, keys.forces: f}
        results.update({keys.stress: np.nan_to_num(s)})

        if stresses is not None:
            results.update({keys.stresses: stresses[ii]})

        if virials is not None:
            results.update({keys.virials: virials[ii]})

        calculator = results2singlepoint(atoms=atoms, results=results)
        atoms.calc = calculator

        trajectory_list.append(atoms)

    trajectory = Trajectory(trajectory_list, metadata=metadata)
    trajectory.displacements = DS.displacements.data
    trajectory.times = DS.time.data

    if keys.fc in DS:
        trajectory.set_force_constants(np.asarray(DS[keys.fc]))

    if keys.fc_remapped in DS:
        trajectory.set_force_constants_remapped(np.asarray(DS[keys.fc_remapped]))

    if keys.hash_raw in attrs:
        trajectory.hash_raw = attrs[keys.hash_raw]

    return trajectory
