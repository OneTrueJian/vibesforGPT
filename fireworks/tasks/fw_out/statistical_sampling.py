"""Generate FWActions after post-processing statistical sampling calculations"""
from fireworks import FWAction
from vibes.fireworks.tasks.fw_out.check_conditionals import run_all_checks
from vibes.fireworks.tasks.postprocess.statistical_sampling import get_sigma
from vibes.helpers.converters import dict2atoms
from vibes.helpers.k_grid import k2d


def add_stat_samp_to_spec(func, func_fw_out, *args, fw_settings=None, **kwargs):
    """Add the phonon_dict to the spec

    Parameters
    ----------
    func : str
        Path to the phonon analysis function
    func_fw_out : str
        Path to this function
    fw_settings : dict
        Dictionary for the FireWorks specific systems (Default value = None)
    kwargs : dict
        Dictionary of keyword arguments that must have the following objects
        workdir: str
            Working directory for the calculation
        trajectory_file: str
            filename for the trajectory_file

    Returns
    -------
    fireworks.FWAction
        FWAction that adds the phonon_dict to the spec

    """
    trajectory_file = f"{kwargs['workdir']}/{kwargs['trajectory_file']}"

    sigma, traj, metadata = get_sigma(trajectory_file, True)

    calculator_dict = metadata["calculator"]
    calculator_dict["calculator"] = calculator_dict["calculator"].lower()
    if calculator_dict["calculator"] == "aims":
        k_pt_density = k2d(
            dict2atoms(metadata["supercell"]),
            calculator_dict["calculator_parameters"]["k_grid"],
        )
    else:
        k_pt_density = None
    qadapter = {}
    if fw_settings and "spec" in fw_settings:
        qadapter = fw_settings["spec"].get("_queueadapter", None)
    update_spec = {
        "stat_samp_calculator": calculator_dict,
        "stat_samp_supercell": metadata["supercell"],
        "_queueadapter": qadapter,
        "sigma": sigma,
    }
    update_spec["kgrid"] = k_pt_density

    if "stop_if" in kwargs:
        return run_all_checks(traj, kwargs["stop_if"], update_spec)

    return FWAction(update_spec=update_spec)
