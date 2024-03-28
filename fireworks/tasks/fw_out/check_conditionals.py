"""Define a set of functions for conditional stopping

Defines a set of functions used for checking if the
properties of an output means that the calculation should stop
"""

from ase.atoms import Atoms
from fireworks import FWAction
from importlib import import_module
from phonopy import Phonopy

import numpy as np

from vibes.phonopy.wrapper import get_debye_temperature, get_thermal_properties
from vibes.trajectory import Trajectory

allowed_comparisions = ["ge", "gt", "le", "lt", "eq", "ne"]

supported_atoms_attributes = {
    "volume": lambda atoms: atoms.get_volume(),
    "energy": lambda atoms: atoms.get_properties(["energy"])["energy"],
    "free_energy": lambda atoms: atoms.get_properties(["free_energy"])["free_energy"],
    "lattice_parameters": lambda atoms: atoms.cell.cellpar()[:3],
    "lattice_angles": lambda atoms: atoms.cell.cellpar()[3:],
    "bandgap": lambda atoms: atoms.info.get("bandgap", np.nan),
}


supported_phonon_attributes = {
    "heat_capacity": lambda ph, T=300: get_thermal_properties(ph, temperatures=[T])[
        "heat_capacity"
    ][0],
    "free_energy": lambda ph, T=300: get_thermal_properties(ph, temperatures=[T])[
        "free_energy"
    ][0],
    "entropy": lambda ph, T=300: get_thermal_properties(ph, temperatures=[T])[
        "entropy"
    ][0],
    "frequencies": lambda ph, q=[0, 0, 0]: ph.get_frequencies(q),
    "theta_D": lambda ph: get_debye_temperature(ph)[2],
    "theta_D_infty": lambda ph: get_debye_temperature(ph)[1],
    "theta_P": lambda ph: get_debye_temperature(ph)[0],
    "n_atoms": lambda ph: len(ph.get_supercell()),
}

supported_traj_attributes = {
    "sigma": (lambda traj: traj.sigma),
}


def check_stop_conditional(function_path, obj):
    """Checks if an object has a property that should stop the workflow

    Parameters
    ----------
    function_path : str
        Path to the function used to check the object
    obj : Object
        The object that the function should act upon

    Returns
    -------
    FWAction
        The desired action to take for the workflow
    """
    module = import_module(".".join(function_path.split(".")[:-1]))
    check_func = getattr(module, function_path.split(".")[-1])

    return check_func(obj)


def check_condition(value_obj, condition):
    """Check if the condition is met

    Parameters
    ----------
    value_obj : float or array like
        The object to check the condition of
    condition : list
        The condition to check [data_op, comparison op, comparison value]
        data_op must describe a function inside numpy that returns a float

    Returns
    -------
    bool
        True if condition is met
    """
    if condition[1] not in allowed_comparisions:
        raise ValueError(
            f"The comparison operator {condition[1]} is not allowed to be used."
        )

    if condition[0] and len(condition[0]) > 0:
        value = float(getattr(np, condition[0])(value_obj))
    else:
        value = float(value_obj)
    return (
        getattr(value, f"__{condition[1]}__")(condition[2])
        if value is not np.nan
        else False
    )


def check_ojbect(obj, condition_list, **kwargs):
    """Checks all conditions of the pre-made functions

    Parameters
    ----------
    obj : (Object)
        The object to be checked
    condition_list : list
        The list of conditions to check
    kwargs : dict
        The keyword arguments for the conditions

    Returns
    -------
    bool
        True if the condition is True
    """
    if isinstance(obj, Atoms):
        supported_attributes = supported_atoms_attributes
    elif isinstance(obj, Phonopy):
        supported_attributes = supported_phonon_attributes
    elif isinstance(obj, Trajectory):
        supported_attributes = supported_traj_attributes
    else:
        raise ValueError("obj is of an unsupported type.")

    # condition_list = {cond[0]: cond[1:] for cond in condition_list}
    for cond in condition_list:
        if cond[0] not in supported_attributes:
            raise ValueError(
                f"The requested attribute {cond[0]} is not supported by this function."
            )

    for cond in condition_list:
        if len(cond) > 4:
            check_value = supported_attributes[cond[0]](obj, *cond[4:])
        else:
            check_value = supported_attributes[cond[0]](obj)

        if check_condition(check_value, cond[1:]):
            return True

    return False


def run_all_checks(obj, stop_if, update_spec=None):
    """Check all predefined and user-defined conditions

    Parameters
    ----------
    obj : Ojbect
        The object to be checked
    stop_if : dict
        Dictionary defining all stopping conditions
    update_spec : dict
        Dictionary of the updated spec
    Returns
    -------
    FWAction
        defuse_workflow if a condition is met
    """

    if check_ojbect(obj, stop_if.get("condition_list", [])):
        return FWAction(defuse_workflow=True, update_spec=update_spec)

    for func_path in stop_if.get("external_functions", []):
        if check_stop_conditional(func_path, obj):
            return FWAction(defuse_workflow=True, update_spec=update_spec)

    if update_spec is not None:
        return FWAction(update_spec=update_spec)

    return None
