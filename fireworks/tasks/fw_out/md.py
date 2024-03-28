"""FireWorks output for MD runs"""

from fireworks import FWAction

from vibes.fireworks.tasks.fw_out.check_conditionals import run_all_checks
from vibes.fireworks.tasks.postprocess.md import check_completion
from vibes.fireworks.workflows.firework_generator import generate_md_fw
from vibes.helpers.converters import dict2atoms
from vibes.settings import Settings
from vibes.trajectory.io import reader


def check_md_finish(atoms_dict, calculator_dict, *args, **kwargs):
    """Check phonon convergence and set up future calculations after a phonon calculation

    Parameters
    ----------
    func: str
        Path to the phonon analysis function
    func_fw_out: str
        Path to this function
    args: list
        list arguments passed to the phonon analysis
    fw_settings: dict
        Dictionary for the FireWorks specific systems
    kwargs: dict
        Dictionary of keyword arguments with the following keys

        workdir: str
            Path to the base phonon force calculations
        trajectory: str
            trajectory file name
        max_steps: int
            Maximum number of steps for the MD

    Returns
    -------
    FWAction
        Increases the supercell size or adds the phonon_dict to the spec
    """
    fw_settings = args[-1]
    workdir = args[3]["md_settings"].pop("workdir")
    settings = Settings(f"{workdir}/md.in", config_files=None)
    settings.md["workdir"] = workdir

    if check_completion(workdir, settings["md"]["maxsteps"]):
        if "stop_if" in kwargs:
            traj = reader(kwargs["trajectory"])
            return run_all_checks(traj, kwargs["stop_if"])
        return

    detours = generate_md_fw(
        settings,
        dict2atoms(atoms_dict, calculator_dict, False),
        fw_settings,
        fw_settings["spec"].get("_queueadapter", None),
        workdir,
    )

    return FWAction(detours=detours)
