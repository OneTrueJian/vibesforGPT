"""FWAction generators for relaxations"""

from fireworks import FWAction
from vibes.fireworks.tasks.postprocess.aims import check_aims
from vibes.fireworks.tasks.postprocess.phonons import time2str
from vibes.fireworks.tasks.fw_out.check_conditionals import run_all_checks
from vibes.fireworks.workflows.firework_generator import generate_firework
from vibes.helpers.converters import dict2atoms
from vibes.helpers.k_grid import k2d


def check_aims_complete(
    atoms_dict,
    calculator_dict,
    outputs,
    func,
    func_fw_out,
    func_kwargs,
    func_fw_kwargs,
    fw_settings,
):
    """A function that checks if a relaxation is converged (if outputs is True)

    either stores the relaxed structure in the MongoDB or
    appends another Firework as its child to restart the relaxation

    Parameters
    ----------
    atoms_dict : dict
        The dictionary representation of the original atoms at the start of this job
    calculator_dict : dict
        The dictionary representation of the original calculator
    outputs : ase.atoms.Atoms
        The geometry of the final relaxation step
    func : str
        Path to function that performs the MD like operation
    func_fw_out : str
        Path to this function
    func_kwargs : dict
        keyword arguments for func
    func_fw_kwargs : dict
        Keyword arguments for fw_out function
    fw_settings : dict
        FireWorks specific settings

    Returns
    -------
    fireworks.FWAction
        The correct action (restart or updated spec) if convergence is reached

    Raises
    ------
    RuntimeError
        If the FHI-Aims calculation fails

    """
    check_aims_fwargs = func_fw_kwargs.copy()
    check_aims_fwargs.update(func_kwargs)
    completed, calc_number, new_atoms_dict, walltime = check_aims(
        atoms_dict, calculator_dict, outputs, **check_aims_fwargs
    )

    update_spec = {}
    if completed:
        if "stop_if" in func_fw_kwargs:
            new_atoms = dict2atoms(new_atoms_dict)
            action = run_all_checks(new_atoms, func_fw_kwargs["stop_if"])
            if action is not None:
                return action

        update_spec = {}
        if "out_spec_atoms" in fw_settings:
            update_spec[fw_settings["out_spec_atoms"]] = new_atoms_dict
        if "out_spec_calc" in fw_settings:
            update_spec[fw_settings["out_spec_calc"]] = calculator_dict

        return FWAction(update_spec=update_spec)

    update_spec = {}

    if "in_spec_atoms" in fw_settings:
        update_spec[fw_settings["in_spec_atoms"]] = new_atoms_dict
    if "in_spec_calc" in fw_settings:
        update_spec[fw_settings["in_spec_calc"]] = calculator_dict

    update_spec["kgrid"] = k2d(
        dict2atoms(new_atoms_dict), calculator_dict["calculator_parameters"]["k_grid"]
    )
    if "walltime" in calculator_dict["calculator_parameters"]:
        calculator_dict["calculator_parameters"]["walltime"] = walltime
        func_kwargs["walltime"] = walltime

    if fw_settings and "spec" in fw_settings and "_queueadapter" in fw_settings["spec"]:
        fw_settings["spec"]["_queueadapter"]["walltime"] = time2str(walltime)

    func_fw_kwargs["calc_number"] = calc_number

    calculator_dict.pop("results", None)
    new_atoms = dict2atoms(new_atoms_dict, calculator_dict, False)

    fw_settings["fw_name"] = fw_settings["fw_base_name"] + str(calc_number)
    fw_settings["spec"].update(update_spec)
    fw_settings["from_db"] = False

    if "to_launchpad" in fw_settings and fw_settings["to_launchpad"]:
        fw_settings["to_launchpad"] = False

    fw = generate_firework(
        func=func,
        func_fw_out=func_fw_out,
        func_kwargs=func_kwargs,
        atoms=new_atoms,
        calc=new_atoms.calc,
        func_fw_out_kwargs=func_fw_kwargs,
        fw_settings=fw_settings,
    )

    return FWAction(detours=[fw], update_spec=update_spec)
