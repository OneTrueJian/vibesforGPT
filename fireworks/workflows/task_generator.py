"""Creates FireWorks Tasks"""
from fireworks import PyTask


def setup_atoms_task(task_spec, atoms, calculator, fw_settings):
    """Setups an ASE Atoms task

    Parameters
    ----------
    task_spec : TaskSpec
        Specification of the Firetask
    atoms : dict
        Dictionary representation of the ase.atoms.Atoms or key string to access
        atoms from mongodb
    calculator : dict
        Dictionary representation of the ASE Calculator or key string to access
        calc from mongodb
    fw_settings : dict
        FireWorks specific parameters

    Returns
    -------
    str
        PyTask function name
    list
        PyTask args
    list of str
        PyTask inputs
    dict
        PyTask kwargs

    """
    pt_func = "vibes.fireworks.tasks.general_py_task.atoms_calculate_task"
    pt_args = list(task_spec.pt_args[:4])
    args = list(task_spec.pt_args[4:])
    pt_inputs = list(task_spec.pt_inputs)
    task_spec.fw_settings = fw_settings
    pt_kwargs = task_spec.pt_kwargs
    if isinstance(atoms, str):
        pt_inputs = [atoms, calculator] + pt_inputs
    elif isinstance(calculator, str):
        pt_inputs = [calculator] + pt_inputs
        pt_args += [atoms]
    else:
        pt_args += [atoms, calculator, *args]
    return (pt_func, pt_args, pt_inputs, pt_kwargs)


def setup_general_task(task_spec, fw_settings):
    """Setups a general task

    Parameters
    ----------
    task_spec : TaskSpec
        Specification of the Firetask
    fw_settings : dict
        FireWorks specific parameters

    Returns
    -------
    str
        PyTask function name
    list
        PyTask args
    list of str
        PyTask inputs
    dict
        PyTask kwargs

    """
    pt_args = task_spec.pt_args
    pt_func = "vibes.fireworks.tasks.general_py_task.general_function_task"
    pt_inputs = task_spec.pt_inputs
    task_spec.fw_settings = fw_settings
    pt_kwargs = task_spec.pt_kwargs
    return (pt_func, pt_args, pt_inputs, pt_kwargs)


def generate_task(task_spec, fw_settings, atoms, calculator):
    """Generates a PyTask for a Firework

    Parameters
    ----------
    task_spec : TaskSpec
        Specification of the Firetask
    fw_settings : dict
        FireWorks specific parameters
    atoms : dict
        Dictionary representation of the ase.atoms.Atoms or key string to access
        atoms from mongodb
    calculator : dict
        Dictionary representation of the ASE Calculator or key string to access
        calc from mongodb

    Returns
    -------
    PyTask
        Task for the given TaskSpec

    """
    if task_spec.task_with_atoms_obj:
        pt_params = setup_atoms_task(task_spec, atoms, calculator, fw_settings)
    else:
        pt_params = setup_general_task(task_spec, fw_settings)

    return PyTask(
        {
            "func": pt_params[0],
            "args": pt_params[1],
            "inputs": pt_params[2],
            "kwargs": pt_params[3],
        }
    )


def generate_update_calc_task(calc_spec, updated_settings):
    """Generate a calculator update task

    Parameters
    ----------
    calc_spec : str
        Spec for the calculator in the Fireworks database
    updated_settings : dict
        What parameters to update

    Returns
    -------
    PyTask
        Task to update the calculator in the Fireworks database

    """
    return PyTask(
        {
            "func": "vibes.fireworks.tasks.utility_tasks.update_calc_in_db",
            "args": [calc_spec, updated_settings],
            "inputs": [calc_spec],
        }
    )


def generate_mod_calc_task(atoms, calculator, calc_spec, kpt_spec):
    """Generate a calculator modifier task

    Parameters
    ----------
    atoms: dict or str
        Either an Atoms dict or a spec key to get the it for the modified system
    calculator: dict or str
        Either a Calculator dict or a spec key to get it for the modified system
    calc_spec : str
        Spec for the calculator in the Fireworks database
    kpt_spec : str
        Spec to update the k-point density of the system

    Returns
    -------
    PyTask
        Task to update the calculator in the Fireworks database

    """
    args = ["k_grid_density", calc_spec]
    kwargs = {"spec_key": kpt_spec}
    if isinstance(calculator, str):
        inputs = [calculator, kpt_spec]
    else:
        args.append(calculator)
        inputs = [kpt_spec]
    if isinstance(atoms, dict):
        kwargs["atoms"] = atoms
    else:
        inputs.append(atoms)
    return PyTask(
        {
            "func": "vibes.fireworks.tasks.utility_tasks.mod_calc",
            "args": args,
            "inputs": inputs,
            "kwargs": kwargs,
        }
    )
