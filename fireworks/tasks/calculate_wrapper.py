"""Wrappers to the vibes calculate functions"""
from pathlib import Path

import numpy as np

from vibes.calculate import calculate, calculate_socket
from vibes.filenames import filenames
from vibes.helpers.converters import dict2atoms
from vibes.helpers.hash import hash_dict
from vibes.settings import Settings


T_S_LINE = (
    "          Detailed time accounting                     : "
    " max(cpu_time)    wall_clock(cpu1)\n"
)
E_F_INCONSISTENCY = "  ** Inconsistency of forces<->energy above specified tolerance.\n"


def check_if_failure_ok(lines, walltime):
    """Checks if the FHI-aims calculation finished

    Parameters
    ----------
    lines: list of str
        List of lines in the aims.out file
    walltime: float
        walltime used in in seconds

    Returns
    -------
    Reason: str
        a string key describing the reason of the failure

    """

    if E_F_INCONSISTENCY in lines:
        return "E_F_INCONSISTENCY"

    line_sum = np.where(lines == T_S_LINE)[0]
    time_used = float(lines[line_sum[0] + 1].split(":")[1].split("s")[1])
    sum_present = line_sum.size > 0

    if walltime and sum_present and time_used / walltime > 0.95:
        return "WALLTIME"

    return "UNKOWN"


def wrap_calc_socket(
    atoms_dict_to_calculate,
    calculator_dict,
    metadata,
    phonon_times=None,
    mem_use=None,
    trajectory_file=filenames.trajectory,
    workdir=".",
    backup_folder="backups",
    walltime=None,
    fw_settings=None,
    **kwargs,
):
    """Wrapper for the clalculate_socket function

    Parameters
    ----------
    atoms_dict_to_calculate : list of dicts
        A list of dicts representing the cellsto calculate the forces on
    calculator_dict : dict
        A dictionary representation of the ASE Calculator used to calculate the Forces
    metadata : dict
        metadata for the force trajectory file
    phonon_times : list
        List of all the phonon calculation times (Default value = None)
    mem_use : list of floats
        List of amount of memory used for all calculations (Default value = None)
    trajectory : str
        file name for the trajectory file (Default value = "trajectory.son")
    workdir : str
        work directory for the force calculations (Default value = ".")
    backup_folder : str
        Directory to store backups (Default value = "backups")
    walltime : int
        number of seconds to run the calculation for (Default value = None)


    Returns
    -------
    bool
        True if all the calculations completed

    Raises
    ------
    RuntimeError
        If the calculation fails

    """
    atoms_to_calculate = []

    if calculator_dict["calculator"].lower() == "aims":
        settings = Settings(settings_file=None)
        if "species_dir" in calculator_dict["calculator_parameters"]:
            from os import path

            species_type = calculator_dict["calculator_parameters"][
                "species_dir"
            ].split("/")[-1]
            calculator_dict["calculator_parameters"]["species_dir"] = path.join(
                settings.machine.basissetloc, species_type
            )

        calculator_dict["command"] = settings.machine.aims_command

        if walltime:
            calculator_dict["calculator_parameters"]["walltime"] = walltime - 180

    for at_dict in atoms_dict_to_calculate:
        atoms_to_calculate.append(dict2atoms(at_dict, calculator_dict, False))

    calculator = dict2atoms(atoms_dict_to_calculate[0], calculator_dict, False).calc
    if "use_pimd_wrapper" in calculator.parameters:
        if calculator.parameters["use_pimd_wrapper"][0][:5] == "UNIX:":
            atoms_hash = hash_dict({"to_calc": atoms_dict_to_calculate})
            name = atoms_to_calculate[0].get_chemical_formula()
            calculator.parameters["use_pimd_wrapper"][
                0
            ] += f"cm_{name}_{atoms_hash[:15]}"

    try:
        return calculate_socket(
            atoms_to_calculate,
            calculator,
            metadata=metadata,
            trajectory_file=trajectory_file,
            workdir=workdir,
            backup_folder=backup_folder,
            **kwargs,
        )
    except RuntimeError:
        if calculator_dict["calculator"].lower() == "aims":
            path = Path(workdir) / "calculations"
            lines = np.array(open(path / filenames.output.aims).readlines())
            failure_okay = check_if_failure_ok(lines, walltime)
            if not failure_okay:
                raise RuntimeError(
                    "FHI-aims failed to converge, and it is not a walltime issue"
                )
            return True

        raise RuntimeError("The calculation failed")


def wrap_calculate(atoms, calculator, workdir=".", walltime=1800, fw_settings=None):
    """Wrapper for the clalculate_socket function

    Parameters
    ----------
    atoms : ase.atoms.Atoms
        structure to be simulated
    calc : ase.calculator.calculator
        Calculator for the simulation
    workdir : folder
        Folder to perform calculation in. (Default value = ".")
    walltime : int
        number of seconds to run the calculation for (Default value = 1800)

    Returns
    -------
    bool
        True if all the calculations completed

    Raises
    ------
    RuntimeError
        If the calculation fails

    """
    calculator.parameters["walltime"] = walltime
    calculator.parameters.pop("use_pimd_wrapper", None)

    try:
        return calculate(atoms, calculator, workdir)
    except RuntimeError:
        if calculator.name.lower() == "aims":
            path = Path(workdir)
            lines = np.array(open(path / filenames.output.aims).readlines())
            failure_okay = check_if_failure_ok(lines, walltime)

            if failure_okay:
                return atoms

            raise RuntimeError(
                "FHI-aims failed to converge, and it is not a walltime issue"
            )
        raise RuntimeError("The calculation failed")
