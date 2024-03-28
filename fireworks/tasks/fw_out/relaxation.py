"""FireWorks output for MD runs"""

from ase.io.aims import read_aims
from fireworks import FWAction
from jconfigparser.dict import DotDict

from vibes.fireworks.tasks.postprocess.relaxation import check_completion
from vibes.fireworks.tasks.fw_out.check_conditionals import run_all_checks
from vibes.fireworks.workflows.firework_generator import generate_relax_fw
from vibes.helpers.converters import atoms2dict, dict2atoms
from vibes.settings import Settings


def check_relax_finish(atoms_dict, calc_dict, *args, **kwargs):
    """Check phonon convergence and set up future calculations after a phonon calculation

    Parameters
    ----------
    atoms_dict: dict
        Dictionary describing the Atoms object
    calc_dict: dict
        Dictionary describing the Calculator object
    args: list
        list arguments passed to the phonon analysis
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
    relax_settings = args[3]["relax_settings"]
    workdir = args[3]["relax_settings"].pop("workdir", None)

    if workdir is None:
        workdir = "."

    settings = Settings(f"{workdir}/relaxation.in", config_files=None)

    settings["relaxation"] = DotDict(
        {relax_settings["step"]: settings.pop("relaxation")}
    )

    settings.relaxation[relax_settings["step"]]["qadapter"] = fw_settings["spec"].get(
        "_qadapter"
    )

    settings["fireworks"] = DotDict()
    settings.fireworks["workdir"] = DotDict({"remote": workdir})

    new_atoms = read_aims(f"{workdir}/geometry.in.next_step")
    if calc_dict.get("calculator") == "Aims":
        for line in open(f"{workdir}/calculation/aims.out").readlines()[::-1]:
            if "ESTIMATED overall HOMO-LUMO gap:" in line:
                new_atoms.info["bandgap"] = float(line.split(":")[1].split("eV")[0])
                break

    new_atoms_dict = atoms2dict(new_atoms)

    if check_completion(workdir, relax_settings["fmax"]):
        if "stop_if" in relax_settings:
            action = run_all_checks(new_atoms, relax_settings["stop_if"])
            if action is not None:
                return action

        update_spec = {}
        if "out_spec_atoms" in fw_settings:
            update_spec[fw_settings["out_spec_atoms"]] = new_atoms_dict
        if "out_spec_calc" in fw_settings:
            update_spec[fw_settings["out_spec_calc"]] = calc_dict
        return FWAction(update_spec=update_spec)

    detours = generate_relax_fw(
        settings,
        dict2atoms(new_atoms_dict, calc_dict, False),
        fw_settings,
        relax_settings["step"],
    )
    update_spec = {}
    if "out_spec_atoms" in fw_settings:
        update_spec[fw_settings["out_spec_atoms"]] = new_atoms_dict
    if "out_spec_calc" in fw_settings:
        update_spec[fw_settings["out_spec_calc"]] = calc_dict

    detours.spec.update(update_spec)

    return FWAction(detours=detours, update_spec=update_spec)
