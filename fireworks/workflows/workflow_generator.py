"""Functions used to generate a FireWorks Workflow"""
from pathlib import Path

from fireworks import Workflow
from jconfigparser.dict import DotDict

from vibes.fireworks.launchpad import LaunchPad
from vibes.fireworks.workflows.firework_generator import (
    generate_aims_fw,
    generate_aims_relax_fw,
    generate_gruniesen_fd_fw,
    generate_kgrid_fw,
    generate_md_fw,
    generate_phonon_fw,
    generate_phonon_postprocess_fw,
    generate_relax_fw,
    generate_stat_samp_fw,
    generate_stat_samp_postprocess_fw,
    get_phonon_file_location,
)
from vibes.helpers.hash import hash_atoms_and_calc
from vibes.phonopy._defaults import kwargs as ph_defaults


def process_relaxation(workflow, atoms, fw_settings, basis):
    """Processes the workflow settings to get all relaxation steps

    Parameters
    ----------
    workflow : settings.Settings
        Settings for the workflow
    atoms : ase.atoms.Atoms
        The Atoms object for the structure
    fw_settings : dict
        FireWorks specific settings
    basis : str
        The default basis set used for this calculation:

    Returns
    -------
    list of FireWorks
        Relaxation FireWorks to add to the workflow

    """
    fw_steps = []
    relaxation_steps = []
    settings = workflow.settings.relaxation
    for key, val in settings.items():
        if isinstance(val, DotDict):
            try:
                relaxation_steps.append(int(key))
            except ValueError:
                raise ValueError("relaxation step keys must be whole numbers")

    for step in sorted(relaxation_steps):
        if not settings.get("use_aims_relax", False):
            fw_steps.append(
                generate_relax_fw(workflow.settings, atoms, fw_settings, str(step))
            )
        else:
            fw_steps.append(
                generate_aims_relax_fw(workflow.settings, atoms, fw_settings, str(step))
            )

    return fw_steps


def process_phonons(
    workflow, atoms, fw_settings, basis, update_in_spec=True, prev_dos_fp=None
):
    """Processes the workflow settings to get all phonopy steps

    Parameters
    ----------
    workflow : settings.Settings
        Settings for the workflow
    atoms : ase.atoms.Atoms
        The Atoms object for the structure
    fw_settings : dict
        FireWorks specific settings
    basis : str
        The default basis set used for this calculation
    update_in_spec : bool
        If True then update the input spec
    prev_dos_fp: PhononDOSFingerprint
        The fingerprint of the previous phonopy calculation

    Returns
    -------
    list of FireWorks
        Phonopy related FireWorks to add to the workflow

    Raises
    ------
    ValueError
        If supercell_matrix is not provided for phonopy, phono3py, or statistical_sampling

    """
    phonon_fws = []
    ignore_keys = ["trigonal", "q_mesh"]
    settings = workflow.settings.phonopy

    for key, val in ph_defaults.items():
        if key not in settings and key not in ignore_keys:
            settings[key] = val

    if "serial" not in settings:
        workflow.settings.phonopy["serial"] = True

    if basis:
        workflow.settings.phonopy["basisset_type"] = basis
    else:
        workflow.settings.phonopy.pop("basisset_type", None)

    if "supercell_matrix" not in settings:
        raise ValueError("Initial supercell_matrix must be provided")

    phonon_fws.append(
        generate_phonon_fw(
            workflow.settings, atoms, fw_settings, "phonopy", update_in_spec
        )
    )
    phonon_fws.append(
        generate_phonon_postprocess_fw(
            workflow.settings, atoms, fw_settings, "phonopy", prev_dos_fp
        )
    )
    if "gruneisen" in workflow.settings:
        phonon_fws += process_grun(workflow, atoms, fw_settings)
    return phonon_fws


def process_stat_samp(workflow, atoms, fw_settings):
    """Processes the workflow settings to get all Statistical Sampling steps

    Parameters
    ----------
    workflow : settings.Settings
        Settings for the workflow
    atoms : ase.atoms.Atoms
        The Atoms object for the structure
    fw_settings : dict
        FireWorks specific settings

    Returns
    -------
    list of FireWorks
        Statistical Sampling related FireWorks to add to the workflow

    """
    stat_samp_fws = []
    stat_samp_fws.append(generate_stat_samp_fw(workflow.settings, atoms, fw_settings))

    stat_samp_fws.append(
        generate_stat_samp_postprocess_fw(workflow.settings, atoms, fw_settings)
    )
    return stat_samp_fws


def process_grun(workflow, atoms, fw_settings):
    """Processes the workflow settings to get all Gruneisen steps

    Parameters
    ----------
    workflow : settings.Settings
        Settings for the workflow
    atoms : ase.atoms.Atoms
        The Atoms object for the structure
    fw_settings : dict
        FireWorks specific settings

    Returns
    -------
    list of FireWorks
        Gruneisen parameter related FireWorks to add to the workflow

    """
    grun_fws = []
    settings = workflow.settings
    trajectory_file = get_phonon_file_location(settings, atoms)

    constraints = []
    for cosntr in atoms.constraints:
        constraints.append(cosntr.todict())
    grun_fws.append(
        generate_gruniesen_fd_fw(
            settings, atoms, trajectory_file, constraints, fw_settings
        )
    )
    return grun_fws


def process_workdir(workdir, atoms, make_absolute):
    """process the working directory

    Parameters
    ----------
    workdir: str
        input working directory
    atoms: ase.atoms.Atoms
        ASE Atoms object for the working directory
    make_absolute: bool
        If True make the working directory absolute

    Returns
    -------
    workdir: str
        output working directory

    """
    if make_absolute:
        workdir = Path(workdir).absolute()

    workdir = (
        workdir
        / atoms.symbols.get_chemical_formula(mode="metal", empirical=True)
        / hash_atoms_and_calc(atoms)[0]
    )

    return str(workdir) + "/"


def generate_workflow(workflow, atoms, launchpad_yaml=None, make_absolute=True):
    """Generates a workflow from given set of steps

    Parameters
    ----------
    workflow : Settings
        The settings object for the desired workflow
    atoms : ase.atoms.Atoms
        ASE Atoms object to preform the calculation on, with an attached calculator
    launchpad_yaml : str
        filename for the launchpad definition file (Default value = None)
    make_absolute: bool
        If True make the working directory absolute

    Returns
    -------
    fireworks.Workflow
        The FireWorks Workflow for the given workflow

    """
    fw_steps = []
    fw_dep = {}

    name = atoms.symbols.get_chemical_formula(mode="metal", empirical=True)
    name += f"_{hash_atoms_and_calc(atoms)[0]}"
    fw_settings = {"name": name}

    wd_remote = workflow.settings.fireworks.workdir.get(
        "remote", workflow.settings.fireworks.workdir.local
    )
    workflow.settings.fireworks.workdir["local"] = process_workdir(
        workflow.settings.fireworks.workdir.local, atoms, make_absolute
    )
    workflow.settings.fireworks.workdir["remote"] = process_workdir(
        wd_remote, atoms, make_absolute
    )

    if "basissets" in workflow.settings.calculator:
        basis = workflow.settings.calculator.basissets.get("default", "light")
    else:
        basis = None

    # K-grid optimization
    if (
        workflow.settings.calculator.name == "aims"
        and "optimize_kgrid" in workflow.settings
    ):
        fw_steps.append(generate_kgrid_fw(workflow.settings, atoms, fw_settings))

    # Relaxation
    if "relaxation" in workflow.settings:
        fw_steps += process_relaxation(workflow, atoms, fw_settings, basis)

    # Setup workflow branching point
    for ii in range(len(fw_steps) - 1):
        fw_dep[fw_steps[ii]] = fw_steps[ii + 1]

    if fw_steps:
        final_initialize_fw = fw_steps[-1]
        fw_dep[final_initialize_fw] = []
    else:
        final_initialize_fw = None

    # Phonon Calculations
    if "phonopy" in workflow.settings:
        phonon_fws = process_phonons(workflow, atoms, fw_settings, basis)
        fw_dep[phonon_fws[0]] = phonon_fws[1]
        if final_initialize_fw:
            fw_dep[final_initialize_fw].append(phonon_fws[0])

        if len(phonon_fws) > 2:
            fw_dep[phonon_fws[1]] = [phonon_fws[2]]
        else:
            fw_dep[phonon_fws[1]] = []
        fw_steps += phonon_fws

    # Statistical Sampling
    if "statistical_sampling" in workflow.settings:
        stat_samp_fws = process_stat_samp(workflow, atoms, fw_settings)

        if "phonopy" in workflow.settings:
            fw_dep[phonon_fws[1]].append(stat_samp_fws[0])
        elif final_initialize_fw:
            fw_dep[final_initialize_fw].append(stat_samp_fws[0])

        fw_dep[stat_samp_fws[0]] = stat_samp_fws[1]
        fw_steps += stat_samp_fws

    # Molecular dynamics
    if "md" in workflow.settings:
        md_fws = generate_md_fw(workflow.settings, atoms, fw_settings)
        if "stop_if" in workflow.settings.get("statistical_sampling", {}):
            fw_dep[stat_samp_fws[1]] = md_fws
        elif "phonopy" in workflow.settings:
            fw_dep[phonon_fws[1]] += md_fws
        elif final_initialize_fw:
            fw_dep[final_initialize_fw] += md_fws
        fw_steps += md_fws

    # Aims Calculations if no other term is present
    if not fw_steps:
        fw_steps.append(generate_aims_fw(workflow.settings, atoms, fw_settings))

    name = f"{workflow.settings.fireworks.name}_{fw_settings['name']}"
    if launchpad_yaml:
        launchpad = LaunchPad.from_file(launchpad_yaml)
        launchpad.add_wf(Workflow(fw_steps, fw_dep, name=name))
        return None

    return Workflow(fw_steps, fw_dep, name=name)
