"""Post processing for FHI-aims calculations"""

from shutil import copyfile

import numpy as np
from ase.io.aims import read_aims
from vibes.fireworks.tasks.calculate_wrapper import check_if_failure_ok
from vibes.helpers.converters import atoms2dict, calc2dict, dict2atoms, key_constraints


def check_aims(
    atoms_dict, calc_dict, outputs, calc_number=0, workdir="./", walltime=0, **kwargs
):
    """
    A function that checks if a relaxation is converged (if outputs is True) and either
    stores the relaxed structure in the MongoDB or appends another Firework as its child
    to restart the relaxation

    Parameters
    ----------
    atoms_dict: dict
        The dictionary representation of the original atoms at the start of this job
    calc_dict: dictr
        The dictionary representation of the original calculator
    outputs: ASE Atoms Object
        The geometry of the final relaxation step
    calc_number: int
        Number of calculations done so far in a single FHI-aims calculation step
    workdir: str
        Working directory for the calculation
    walltime: int
        Length of the calculation in seconds

    Returns
    -------
    completed: bool
        True if the calculation finished
    calc_number: int
        The number of FHI-Aims jobs that have been completed
    new_atoms_dict: dict
        Dictionary of the updated atoms object
    walltime: int
        Length of the calculation in seconds

    Raises
    ------
    ValueError
        If calculation failed
    """
    aims_out = np.array(open(workdir + "/aims.out").readlines())
    completed = "Have a nice day" in aims_out[-2] or "Have a nice day" in aims_out[-3]
    calc = calc2dict(outputs.get_calculator())
    calc_number += 1

    try:
        if "relax_geometry" in calc["calculator_parameters"]:
            new_atoms = read_aims(workdir + "/geometry.in.next_step")
            new_atoms.set_calculator(outputs.get_calculator())
            new_atoms.info = atoms_dict["info"].copy()
        else:
            new_atoms = dict2atoms(atoms_dict)
    except FileNotFoundError:
        if not completed:
            failure_ok = check_if_failure_ok(aims_out, walltime)
            if failure_ok == "WALLTIME":
                walltime *= 2
                calc.parameters["walltime"] = walltime
            elif calc_number > 10:
                raise ValueError(
                    "Number of failed calculations exceeds 10, stopping here"
                )
            elif failure_ok != "E_F_INCONSISTENCY":
                raise ValueError(
                    "There was a problem with the FHI Aims calculation stopping here"
                )
        new_atoms = outputs

    for line in aims_out[::-1]:
        if "ESTIMATED overall HOMO-LUMO gap:" in line:
            new_atoms.info["bandgap"] = float(line.split(":")[1].split("eV")[0])
            break

    new_atoms_dict = atoms2dict(new_atoms)
    if key_constraints in atoms_dict:
        new_atoms_dict[key_constraints] = atoms_dict[key_constraints]

    for key, val in atoms_dict["info"].items():
        if key not in new_atoms_dict["info"]:
            new_atoms_dict["info"][key] = val

    copyfile(f"{workdir}/aims.out", f"{workdir}/aims.out.{calc_number}")

    return completed, calc_number, new_atoms_dict, walltime
