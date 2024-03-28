"""Functions used to wrap around HiLDe Phonopy/Phono3py functions"""
from pathlib import Path

import numpy as np
from ase.constraints import (
    FixCartesianParametricRelations,
    FixScaledParametricRelations,
    dict2constraint,
)
from jconfigparser.dict import DotDict

from vibes.context import TaskContext
from vibes.fireworks.tasks.general_py_task import get_func
from vibes.fireworks.workflows.workflow_generator import generate_workflow
from vibes.helpers.converters import dict2atoms, input2dict
from vibes.phonopy import displacement_id_str
from vibes.phonopy.context import PhonopyContext
from vibes.phonopy.postprocess import postprocess
from vibes.settings import Settings
from vibes.structure.convert import to_Atoms
from vibes.trajectory import metadata2file, reader, step2file


def setup_calc(settings):
    """Sets up a calculation

    Parameters
    ----------
    settings : Settings
        The settings object for the calculation

    Returns
    -------
    settings : Settings
        The updated settings object

    """
    if settings.calculator.name.lower() != "aims":
        return settings

    if "species_dir" in settings.calculator.parameters:
        sd = settings["calculator"]["parameters"].pop("species_dir")
        settings["calculator"]["basissets"] = DotDict({"default": sd.split("/")[-1]})

    if "use_pimd_wrapper" in settings.calculator.parameters:
        pimd = settings["calculator"]["parameters"].pop("use_pimd_wrapper")
        host = pimd[0]
        port = pimd[1]
        if "UNIX:" in host:
            unixsocket = host
            host = None
        else:
            unixsocket = None
        settings["calculator"]["socketio"] = DotDict(
            {"port": port, "host": host, "unixsocket": unixsocket}
        )
    else:
        settings["calculator"].pop("socketio", None)

    settings["calculator"]["parameters"].pop("aims_command", None)

    return settings


def setup_phonon_outputs(ph_settings, settings, prefix, atoms):
    """Sets up the phonon outputs

    Parameters
    ----------
    ph_settings : dict
        Settings object for the phonopy, phono3py object
    settings : Settings
        General settings for the step
    prefix : str
        key prefix for the task
    atoms : ase.atoms.Atoms
        ASE Atoms object for the material

    Returns
    -------
    outputs : dict
        All the necessary output/metadata for the task

    """
    settings = setup_calc(settings)
    settings[f"{prefix}onopy"] = ph_settings.copy()

    if "serial" in settings[f"{prefix}onopy"]:
        del settings[f"{prefix}onopy"]["serial"]

    ctx = PhonopyContext(settings=settings)
    ctx.primitive = atoms.copy()
    ctx.atoms = atoms.copy()

    outputs = ctx.bootstrap()

    outputs["metadata"]["supercell"] = {"atoms": outputs["metadata"]["atoms"]}
    outputs["metadata"]["primitive"] = input2dict(atoms)
    outputs["metadata"]["settings"] = ctx.settings.to_dict()
    outputs["prefix"] = prefix
    outputs["settings"] = ph_settings.copy()
    return outputs


def bootstrap_phonon(
    atoms,
    calculator,
    kpt_density=None,
    ph_settings=None,
    ph3_settings=None,
    fw_settings=None,
):
    """Creates a Settings object and passes it to the bootstrap function

    Parameters
    ----------
    atoms : ase.atoms.Atoms
        Atoms object of the primitive cell
    calculator: ase.calculators.calulator.Calculator
        Calculator for the force calculations
    kpt_density : float
        k-point density for the MP-Grid (Default value = None)
    ph_settings : dict
        kwargs for phonopy setup (Default value = None)
    ph3_settings : dict
        kwargs for phono3py setup (Default value = None)
    fw_settings : dict
        FireWork specific settings (Default value = None)

    Returns
    -------
    outputs : dict
        The output of vibes.phonopy.workflow.bootstrap for phonopy and phono3py

    """
    settings = Settings(settings_file=None)
    settings["calculator"] = DotDict({"name": calculator.name})
    settings["calculator"]["parameters"] = DotDict(calculator.parameters.copy())
    settings["calculator"]["make_species_dir"] = False
    if kpt_density:
        settings["calculator"]["kpoints"] = DotDict({"density": kpt_density})
        settings["calculator"]["parameters"].pop("k_grid")

    outputs = []
    at = atoms.copy()
    at.set_calculator(None)
    if ph_settings:
        outputs.append(setup_phonon_outputs(ph_settings, settings, "ph", at))

    if ph3_settings:
        outputs.append(setup_phonon_outputs(ph3_settings, settings, "ph3", at))

    if kpt_density:
        for out in outputs:
            out["kpt_density"] = kpt_density
    return outputs


def collect_to_trajectory(workdir, trajectory_file, calculated_atoms, metadata):
    """Collects forces to a single trajectory file

    Parameters
    ----------
    workdir : str
        path to working directory
    trajectory : str
        trajectory file name
    calculated_atoms : list of atoms
        Distorted supercells with forces calculated
    metadata : dict
        Metadata for the phonopy/phono3py calculation

    """
    trajectory_outfile = Path(workdir) / trajectory_file
    trajectory_outfile.parent.mkdir(exist_ok=True, parents=True)
    if "Phonopy" in metadata:
        for el in metadata["Phonopy"]["displacement_dataset"]["first_atoms"]:
            el["number"] = int(el["number"])

    if "Phono3py" in metadata:
        for el1 in metadata["Phono3py"]["displacement_dataset"]["first_atoms"]:
            el1["number"] = int(el1["number"])
            for el2 in el1["second_atoms"]:
                el2["number"] = int(el2["number"])

    metadata2file(metadata, str(trajectory_outfile))
    if isinstance(calculated_atoms[0], dict):
        temp_atoms = []
        for atoms_dict in calculated_atoms:
            calc_dict = atoms_dict.pop("calculator_dict")
            temp_atoms.append(dict2atoms(atoms_dict, calc_dict))
    else:
        temp_atoms = calculated_atoms.copy()

    try:
        calculated_atoms = sorted(
            temp_atoms,
            key=lambda x: x.info[displacement_id_str]
            if x
            else len(calculated_atoms) + 1,
        )
    except KeyError:
        calculated_atoms = sorted(
            temp_atoms,
            key=lambda x: float(x.info["info_str"][1].split("T = ")[1].split(" K")[0])
            if x
            else len(calculated_atoms) + 1,
        )
    for atoms in calculated_atoms:
        if atoms:
            step2file(atoms, atoms.calc, str(trajectory_outfile))


def phonon_postprocess(func_path, phonon_times, kpt_density, **kwargs):
    """Performs phonon postprocessing steps

    Parameters
    ----------
    func_path : str
        The path to the postprocessing function
    phonon_times : list of ints
        The time it took to calculate the phonon forces in seconds
    kwargs : dict
        The keyword arguments for the phonon calculations
    kpt_density : float
        k-point density used for the FHI-aims calculator

    Returns
    -------
    phonopy.Phonopy or phono3py.phonon3.Phono3py
        The Phonopy or Phono3py object generated by the post processing

    """
    func = get_func(func_path)
    return func(**kwargs)


def prepare_gruneisen(settings, primitive, vol_factor):
    """Prepare a Gruneisen calculation

    Parameters
    ----------
    settings : Settings
        The settings object for the calculation
    primitive : ase.atoms.Atoms
        The primitive cell for the phonon calculation
    vol_factor : float
        The volume rescaling factor

    Returns
    -------
    fireworks.Workflow
        A Fireworks workflow for the gruneisen calculation

    """
    if "stop_if" in settings.get("phonopy", {}):
        settings["phonopy"].pop("stop_if")

    dist_primitive = primitive.copy()
    scaled_pos = dist_primitive.get_scaled_positions()
    dist_primitive.cell *= vol_factor ** (1.0 / 3.0)
    dist_primitive.set_scaled_positions(scaled_pos)

    for constraint in dist_primitive.constraints:
        if isinstance(constraint, FixCartesianParametricRelations):
            constraint.params = []
            constraint.Jacobian = np.zeros((constraint.Jacobian.shape[0], 0))
            constraint.Jacobian_inv = np.zeros((0, constraint.Jacobian.shape[0]))
            constraint.const_shift = dist_primitive.cell.flatten()

    dist_settings = Settings()
    for sec_key, sec_val in settings.items():
        if isinstance(sec_val, dict):
            dist_settings[sec_key] = DotDict()
            for key, val in sec_val.items():
                try:
                    dist_settings[sec_key][key] = val.copy()
                except AttributeError:
                    dist_settings[sec_key][key] = val
        else:
            dist_settings[sec_key] = val

    if "geometry" in dist_settings:
        file_original = dist_settings.geometry.pop("file", None)

    if "geometry" in dist_settings:
        dist_settings.geometry["file"] = file_original

    dist_settings.fireworks.workdir["remote"] = str(
        Path(dist_settings.fireworks.workdir["remote"]).parents[1]
    )
    dist_settings.fireworks.workdir["local"] = str(
        Path(dist_settings.fireworks.workdir["local"]).parents[1]
    )

    dist_ctx = TaskContext(name=None, settings=dist_settings)
    dist_ctx.atoms = dist_primitive
    dist_primitive.set_calculator(dist_ctx.calculator)

    return generate_workflow(dist_ctx, dist_primitive, launchpad_yaml=None)


def setup_gruneisen(settings, trajectory_file, constraints, _queueadapter, kpt_density):
    """Set up the finite difference gruniesen parameter calculations

    Parameters
    ----------
    settings : dict
        The workflow settings
    trajectory_file: str
        The trajectory file for the equilibrium phonon calculation
    constraints : list of dict
        list of relevant constraint dictionaries for relaxations
    _queueadapter : dict
        The queueadapter for the equilibrium phonon calculations
    kpt_density :
        The k-point density to use for the calculations

    Returns
    -------
    fireworks.Workflow
        Workflow to run the positive volume change Gruniesen parameter calcs
    fireworks.Workflow
        Workflow to run the positive volume change Gruniesen parameter calcs

    """
    # Prepare settings by reset general work_dir and do not reoptimize k_grid
    settings.pop("optimize_kgrid", None)
    gruneisen = settings.pop("gruneisen", None)
    settings["phonopy"].pop("convergence", None)

    settings.pop("statistical_sampling", None)
    settings.pop("md", None)

    if _queueadapter:
        settings["phonopy.qadapter"] = _queueadapter

    # Get equilibrium phonon
    eq_phonon = postprocess(trajectory_file)
    _, metadata = reader(trajectory_file, get_metadata=True)

    settings["phonopy"]["supercell_matrix"] = eq_phonon.get_supercell_matrix()
    settings["phonopy"]["symprec"] = metadata["Phonopy"].get("symprec", 1e-5)
    settings["phonopy"]["displacement"] = metadata["Phonopy"]["displacements"][0][1]

    settings["calculator"] = DotDict({"name": metadata["calculator"]["calculator"]})
    settings["calculator"]["parameters"] = DotDict(
        metadata["calculator"]["calculator_parameters"]
    )
    if settings["calculator"]["name"].lower() == "aims":
        settings["calculator"]["parameters"].pop("kgrid", None)
        settings["calculator"]["kpoints"] = DotDict({"density": kpt_density})
        settings["calculator"]["basissets"] = DotDict(
            {
                "default": settings["calculator"]["parameters"]
                .get("species_dir")
                .split("/")[-1]
            }
        )
        host, port = settings["calculator"]["parameters"].pop(
            "use_pimd_wrapper", [None, None]
        )
        if "UNIX" in host:
            unixsocket, host = host, None
        else:
            unixsocket = None
        if port:
            settings["calculator"]["socketio"] = DotDict(
                {"port": port, "host": host, "unixsocket": unixsocket}
            )
        else:
            settings["calculator"].pop("socketio", None)
    elif settings["calculator"]["name"].lower() == "emt":
        settings["calculator"].pop("socketio", None)

    if "relaxation" not in settings:
        settings["relaxation"] = DotDict(
            {
                "1": {"driver": "BFGS", "unit_cell": False, "fmax": 0.001},
                "use_ase_relax": True,
            }
        )
    else:
        use_aims_relax = settings["relaxation"].get("use_aims_relax")
        for key, val in settings["relaxation"].items():
            if issubclass(type(val), dict):
                if use_aims_relax:
                    settings["relaxation"][key]["relax_unit_cell"] = "none"
                else:
                    settings["relaxation"][key]["unit_cell"] = False

    primitive = to_Atoms(eq_phonon.get_primitive())
    add_constraints = []
    for constr in constraints:
        constraint = dict2constraint(constr)
        if isinstance(constraint, FixScaledParametricRelations):
            if constraint.params:
                add_constraints.append(constraint)

    if len(constraints) > 0 and len(add_constraints) == 0:
        settings.pop("relaxation", None)

    primitive.constraints = add_constraints
    gruneisen_list = []
    for fact in gruneisen["volume_factors"]:
        gruneisen_list.append(prepare_gruneisen(settings, primitive, fact))

    return gruneisen_list
