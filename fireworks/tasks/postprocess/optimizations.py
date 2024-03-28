"""Postprocess steps for k-grid optimizations"""
from pathlib import Path

from vibes.son import last_from


def load_last_step(atoms_dict, calculator_dict, workdir, trajectory_file):
    """Loads the last step from a trajectory and update returns calculator objects

    Parameters
    ----------
    atoms_dict : dict
        The dictionary representation of the original atoms at the start of this job
    calculator_dict : dict
        The dictionary representation of the original calculator
    workdir : str
        Path to the working directory
    trajectory_file : str
        Path to the trajectory file

    Returns
    -------
    trajectory : vibes.trajectory.Trajectory
        The trajectory of the calculation
    atoms_dict : dict
        The dictionary representation of the last step of the trajectory
    calculator_dict : dict
        The dictionary representation of the calculator with the results from
        last step of the calculation

    """
    trajectory_file = Path(workdir) / trajectory_file
    last_step_dict = last_from(trajectory_file)

    for key, val in last_step_dict["atoms"].items():
        atoms_dict[key] = val

    calculator_dict["results"] = last_step_dict["calculator"]

    return trajectory_file, atoms_dict, calculator_dict


def move_trajectory_file(trajectory_file):
    """Move a trajectory to a new file name

    Parameters
    ----------
    trajectory : str
        The path to the trajectory to move

    Returns
    -------

    """
    split_trajectory_file = trajectory_file.split(".")

    try:
        temp_list = split_trajectory_file[-2].split("_")
        temp_list[-1] = str(int(temp_list[-1]) + 1)
        split_trajectory_file[-2] = "_".join(temp_list)
        trajectory_file = ".".join(split_trajectory_file)
    except ValueError:
        split_trajectory_file[-2] += "_restart_1"
        trajectory_file = ".".join(split_trajectory_file)
