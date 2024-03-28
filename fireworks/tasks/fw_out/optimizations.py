"""FWAction generators for optimizations"""

from fireworks import FWAction
from vibes.fireworks.tasks.postprocess.optimizations import (
    load_last_step,
    move_trajectory_file,
)
from vibes.fireworks.workflows.firework_generator import generate_firework
from vibes.helpers.converters import calc2dict


def check_kgrid_opt_completion(
    atoms_dict,
    calculator_dict,
    outputs,
    func,
    func_fw_out,
    func_kwargs,
    func_fw_kwargs,
    fw_settings,
):
    """A function that checks if an MD like calculation is converged (if outputs is True)

    either stores the relaxed structure in the MongoDB or appends another Firework as
    its child to restart the MD

    Parameters
    ----------
    atoms_dict : dict
        The dictionary representation of the original atoms at the start of this job
    calculator_dict : dict
        The dictionary representation of the original calculator
    outputs : list (bool
        (Converged?, current k-point density,current ASE Calculator)
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
        Either another k-grid optimization step, or an updated spec

    """
    trajectory, atoms_dict, calculator_dict = load_last_step(
        atoms_dict,
        calculator_dict,
        func_kwargs["workdir"],
        func_kwargs["trajectory_file"],
    )

    if outputs[0]:
        up_spec = {
            fw_settings["out_spec_k_den"]: outputs[1] - 1.0,
            fw_settings["out_spec_atoms"]: atoms_dict,
            fw_settings["out_spec_calc"]: calc2dict(outputs[2]),
        }
        return FWAction(update_spec=up_spec)

    fw_settings["fw_name"] = fw_settings["fw_base_name"]
    if fw_settings["to_launchpad"]:
        fw_settings["to_launchpad"] = False

    move_trajectory_file(trajectory)

    fw = generate_firework(
        func=func,
        func_fw_out=func_fw_out,
        func_kwargs=func_kwargs,
        atoms=atoms_dict,
        calculator=outputs[2],
        func_fw_out_kwargs=func_fw_kwargs,
        fw_settings=fw_settings,
    )
    return FWAction(detours=[fw])
