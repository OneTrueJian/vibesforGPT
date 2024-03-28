"""Standardize python function for FW PyTasks"""
import os

from vibes import DEFAULT_CONFIG_FILE
from vibes.helpers import Timer
from vibes.helpers.converters import dict2atoms
from vibes.settings import Settings


def get_func(func_path):
    """A function that takes in a path to a python function and returns that function

    Parameters
    ----------
    func_path : str
        The path to the python function

    Returns
    -------
    Function
        function to use for the task

    """
    toks = func_path.rsplit(".", 1)
    if len(toks) == 2:
        modname, funcname = toks
        mod = __import__(modname, globals(), locals(), [str(funcname)], 0)
        return getattr(mod, funcname)
    # Handle built in functions.
    return getattr("builtins", toks[0])


def atoms_calculate_task(
    func_path,
    func_fw_out_path,
    func_kwargs,
    func_fw_out_kwargs,
    atoms_dict,
    calculator_dict,
    *args,
    fw_settings=None,
    walltime=None,
):
    """A wrapper function for

    Converts a general function that performs some operation on ASE Atoms/Calculators
    into a FireWorks style operation

    Parameters
    ----------
    func_path : str
        Path to the function describing the desired set operations to be performed
    func_fw_out_path : str
        Path to the function that describes how the results should alter the workflow
    func_kwargs : dict
        A dictionary describing the key word arguments to func
    func_fw_out_kwargs : dict
        Keyword arguments for fw_out function
    atoms_dict : dict
        A dictionary describing the ASE Atoms object
    calculator_dict : dict
        A dictionary describing the ASE Calculator
    args : list
        a list of function arguments passed to func
    fw_settings : dict
        A dictionary describing the FireWorks specific settings used in func_fw_out
        (Default value = None)
    walltime : float
        time of job


    Returns
    -------
    fireworks.FWAction
        The FWAction func_fw_out outputs

    Raises
    ------
    RuntimeError
        If the Task fails

    """
    if walltime:
        func_kwargs["walltime"] = walltime
        func_fw_out_kwargs["walltime"] = walltime

    func_kwargs["fw_settings"] = fw_settings

    start_dir = os.getcwd()
    if fw_settings is None:
        fw_settings = {}

    func = get_func(func_path)
    func_fw_out = get_func(func_fw_out_path)

    default_settings = Settings(DEFAULT_CONFIG_FILE)

    if calculator_dict["calculator"].lower() == "aims":
        calculator_dict["command"] = default_settings.machine.aims_command
        if "species_dir" in calculator_dict["calculator_parameters"]:
            calculator_dict["calculator_parameters"]["species_dir"] = (
                str(default_settings.machine.basissetloc)
                + "/"
                + calculator_dict["calculator_parameters"]["species_dir"].split("/")[-1]
            )

    if "results" in calculator_dict:
        del calculator_dict["results"]

    atoms = dict2atoms(atoms_dict.copy(), calculator_dict, False)

    try:
        func_timer = Timer()
        if args:
            outputs = func(atoms, atoms.calc, *args, **func_kwargs)
        else:
            outputs = func(atoms, atoms.calc, **func_kwargs)
        func_fw_out_kwargs["run_time"] = func_timer()
    except Exception:
        os.chdir(start_dir)
        raise RuntimeError(
            f"Function calculation failed, moving to {start_dir} to finish Firework."
        )

    os.chdir(start_dir)
    fw_acts = func_fw_out(
        atoms_dict,
        calculator_dict,
        outputs,
        func_path,
        func_fw_out_path,
        func_kwargs,
        func_fw_out_kwargs,
        fw_settings,
    )
    return fw_acts


def general_function_task(
    func_path, func_fw_out_path, *args, fw_settings=None, **kwargs
):
    """A wrapper function that converts a python function into a FireWork

    Parameters
    ----------
    func_path : str
        Path to the function describing the desired set operations to be performed
    func_fw_out_path : str
        Path to the function that describes how the results should alter the Workflow
    args : list
        A list of arguments to pass to func and func_fw_out
    fw_settings : dict
        A dictionary describing the FireWorks specific settings used in func_fw_out
        (Default value = None)
    kwargs : dict
        A dict of key word arguments to pass to the func and func_fw_out

    Returns
    -------
    fireworks.FWAction
        The FWAction func_fw_out outputs

    """
    if fw_settings is None:
        fw_settings = {}
    func = get_func(func_path)
    func_fw_out = get_func(func_fw_out_path)

    kwargs["outputs"] = func(*args, **kwargs)

    return func_fw_out(
        func_path, func_fw_out_path, *args, fw_settings=fw_settings, **kwargs
    )
