"""Generates Task Specific FireWorks"""
import numpy as np
from fireworks import Firework
from jconfigparser.dict import DotDict
from pathlib import Path

from vibes.filenames import filenames
from vibes.fireworks.tasks.fw_out.check_conditionals import (
    supported_atoms_attributes,
    supported_phonon_attributes,
    supported_traj_attributes,
)
from vibes.fireworks.tasks.task_spec import TaskSpec
from vibes.fireworks.tasks.utility_tasks import update_calc
from vibes.fireworks.workflows.task_generator import (
    generate_mod_calc_task,
    generate_task,
    generate_update_calc_task,
)
from vibes.fireworks.workflows.task_spec_generator import (
    gen_aims_task_spec,
    gen_gruniesen_task_spec,
    gen_kgrid_task_spec,
    gen_md_task_spec,
    gen_phonon_analysis_task_spec,
    gen_phonon_task_spec,
    gen_relax_task_spec,
    gen_stat_samp_analysis_task_spec,
    gen_stat_samp_task_spec,
)
from vibes.helpers.converters import atoms2dict, calc2dict
from vibes.helpers.hash import hash_atoms_and_calc
from vibes.helpers.k_grid import k2d, update_k_grid, update_k_grid_calc_dict
from vibes.helpers.numerics import get_3x3_matrix
from vibes.helpers.watchdogs import str2time


def ceheck_condition_list(condition_list):
    """Checks the condition list to ensure that the conditions are valid

    Parameters
    ----------
    condition_list : list
        The condition to check

    Raises
    ------
    ValueError
        If one of the terms in the condition list is wrong
    """
    if len(condition_list) not in [4, 5]:
        raise ValueError("Invalid size for the condition: {condition_list}.")
    elif (len(condition_list) == 5) and (
        condition_list[0] not in ["cv", "free_energy", "entropy", "frequencies"]
    ):
        raise ValueError(
            "An invalid extra parameter passed to the condition: {condition_list}."
        )

    if condition_list[2] not in ["eq", "ne", "le", "ge", "lt", "gt"]:
        raise ValueError(
            f"Invalid comparison operation for the condition: {condition_list}."
        )

    if condition_list[1] not in ["", "min", "max", "mean", "median", "std"]:
        raise ValueError(
            f"Invalid vector operation for the condition: {condition_list}."
        )

    if condition_list[0] not in (
        list(supported_atoms_attributes.keys())
        + list(supported_phonon_attributes.keys())
        + list(supported_traj_attributes.keys())
    ):
        raise ValueError(f"Invalid property for the condition: {condition_list}.")


def parse_conditional(condition):
    """Parse the stopping condition and replace it with the condition list.

    Parameters
    ----------
    condition : str
        The condition to be parsed

    Returns
    -------
    list
        ["key", "vector_op", "comparision_op", value]
    """
    comp_ops = {"==": "eq", "!=": "ne", "<=": "le", ">=": "ge", "<": "lt", ">": "gt"}
    if isinstance(condition, list):
        ceheck_condition_list(condition)
        return condition

    for key, val in comp_ops.items():
        if key in condition:
            condition_split = condition.split(key)
            condition_list = ["", "", val, float(condition_split[-1])]
            chars, counts = np.unique(list(condition_split[0]), return_counts=True)
            op_check = [
                (char, count)
                for char, count in zip(chars, counts)
                if char in ["(", ")"]
            ]
            if (len(op_check) not in [0, 2]) or (
                not all(oc[1] == 1 for oc in op_check)
            ):
                raise ValueError(
                    f'Vector operation is not properly formatted in "{condition}".'
                )
            elif len(op_check) == 0:
                condition_list[:2] = [condition_split[0].strip(), ""]
            else:
                vec_split = condition_split[0].split("(")

                condition_list[:2] = [vec_split[1].split(")")[0].strip(), vec_split[0]]

            ceheck_condition_list(condition_list)
            return condition_list

    raise ValueError(f'No comparison operator found in the condition "{condition}"')


def get_phonon_file_location(settings, atoms, remote=False):
    """Get the phonon file location for a task

    Parameters
    ----------
    settings: Settings
        The settings for the overall workflow
    atoms: ase.atoms.Atoms
        The Atoms object for the workflow
    remote: bool
        True if use remote directory instead of the local one

    Returns
    str
        The phonon_file location
    """
    if "phonopy" not in settings:
        raise IOError("phonon file must be given")

    conv = False
    if "convergence" in settings.phonopy:
        if isinstance(settings.phonopy.convergence, bool):
            conv = settings.phonopy.convergence
        else:
            conv = True

    if remote:
        base_direc = settings.fireworks.workdir.remote
    else:
        base_direc = settings.fireworks.workdir.local

    if conv:
        phonon_file = f"{base_direc}/converged/trajectory.son"
    else:
        sc_mat = get_3x3_matrix(settings.phonopy.supercell_matrix)
        sc_natoms = int(round(np.linalg.det(sc_mat) * len(atoms)))

        phonopy_dir = "phonopy" if remote else "phonopy_analysis"
        rel_dir = f"/sc_natoms_{sc_natoms}/{phonopy_dir}/trajectory.son"

        phonon_file = base_direc + rel_dir

    return phonon_file


def update_fw_settings(fw_settings, fw_name, queueadapter=None, update_in_spec=True):
    """update the fw_settings for the next step

    Parameters
    ----------
    fw_settings : dict
        Current fw_settings
    fw_name : str
        name of the current step
    queueadapter : dict
        dict describing the queueadapter changes for this firework (Default value = None)
    update_in_spec : bool
        If true move current out_spec to be in_spec (Default value = True)

    Returns
    -------
    dict
        The updated fw_settings

    """
    if "out_spec_atoms" in fw_settings and update_in_spec:
        fw_settings["in_spec_atoms"] = fw_settings["out_spec_atoms"]
        fw_settings["in_spec_calc"] = fw_settings["out_spec_calc"]
        fw_settings["from_db"] = True

    fw_settings["out_spec_atoms"] = fw_name + "_atoms"
    fw_settings["out_spec_calc"] = fw_name + "_calc"
    fw_settings["fw_name"] = fw_name
    if "spec" not in fw_settings:
        fw_settings["spec"] = {}

    if queueadapter:
        fw_settings["spec"]["_queueadapter"] = queueadapter
    elif "_queueadapter" in fw_settings["spec"]:
        del fw_settings["spec"]["_queueadapter"]

    return fw_settings


def generate_firework(
    task_spec_list=None,
    atoms=None,
    calculator=None,
    fw_settings=None,
    atoms_calc_from_spec=False,
    update_calc_settings=None,
    func=None,
    func_fw_out=None,
    func_kwargs=None,
    func_fw_out_kwargs=None,
    args=None,
    inputs=None,
):
    """A function that takes in a set of inputs and returns a Firework for them

    Parameters
    ----------
    task_spec_list : list of TaskSpecs
        list of task specifications to perform (Default value = None)
    atoms : ase.atoms.Atoms
        If not atoms_calc_from_spec then this must be an ASE Atoms object or a dict
        If atoms_calc_from_spec then this must be a key str
    calculator: ase.calculators.calulator.Calculator, dictionary or str
        If not atoms_calc_from_spec then this must be an ASE Calculator or a dict
        If atoms_calc_from_spec then this must be a key str (Default value = None)
    fw_settings : dict
        Settings used by fireworks to place objects in the right part of the MongoDB
        (Default value = None)
    atoms_calc_from_spec : bool
        If True retrieve the atoms/Calculator objects from the MongoDB launchpad
        (Default value = False)
    update_calc_settings : dict
        Used to update the Calculator parameters (Default value = None)
    func : str
        Function path for the firework (Default value = None)
    func_fw_out : str
        Function path for the fireworks FWAction generator (Default value = None)
    func_kwargs : dict
        Keyword arguments for the main function (Default value = None)
    func_fw_out_kwargs : dict
        Keyword arguments for the fw_out function (Default value = None)
    args : list
        List of arguments to pass to func (Default value = None)
    inputs : (ist
        List of spec to pull in as args from the FireWorks Database (Default value = None)

    Returns
    -------
    fireworks.Firework
        A Firework that will perform the desired operation on a set of atoms

    Raises
    ------
    ValueError
        If conflicting task_spec definitions provided, or none are provided

    """
    fw_settings = fw_settings.copy()

    if "spec" not in fw_settings:
        fw_settings["spec"] = {}

    if update_calc_settings is None:
        update_calc_settings = {}

    if func and not task_spec_list:
        task_with_atoms_obj = atoms is not None
        task_spec_list = [
            TaskSpec(
                func,
                func_fw_out,
                task_with_atoms_obj,
                func_kwargs,
                func_fw_out_kwargs,
                args,
                inputs,
            )
        ]
    elif not task_spec_list:
        raise ValueError(
            "Define either a task_spec_list or arguments to generate one, but not both"
        )
    if isinstance(task_spec_list, TaskSpec):
        task_spec_list = [task_spec_list]

    atoms_calc_from_spec = fw_settings.get("from_db", False)

    if "fw_name" not in fw_settings:
        fw_settings["fw_base_name"] = ""
    elif "fw_base_name" not in fw_settings:
        fw_settings["fw_base_name"] = fw_settings["fw_name"]

    setup_tasks = []
    if atoms:
        if not atoms_calc_from_spec:
            # Preform calculator updates here
            at = atoms2dict(atoms, add_constraints=True)

            if not isinstance(calculator, str):
                if "k_grid_density" in update_calc_settings:
                    if not isinstance(calculator, dict) and calculator.name == "aims":
                        update_k_grid(
                            atoms, calculator, update_calc_settings["k_grid_density"]
                        )
                    elif calculator["claculator"].lower() == "aims":
                        recipcell = np.linalg.pinv(at["cell"]).transpose()
                        calculator = update_k_grid_calc_dict(
                            calculator,
                            recipcell,
                            update_calc_settings["k_grid_density"],
                        )

                cl = calc2dict(calculator)

                for key, val in update_calc_settings.items():
                    if key not in ("k_grid_density", "kgrid"):
                        cl = update_calc(cl, key, val)

                if cl["calculator"].lower() == "aims":
                    fw_settings["spec"]["kgrid"] = k2d(
                        atoms, cl["calculator_parameters"]["k_grid"]
                    )
                else:
                    fw_settings["spec"]["kgrid"] = None
            else:
                cl = calculator
                setup_tasks.append(
                    generate_update_calc_task(calculator, update_calc_settings)
                )
        else:
            # Add tasks to update calculator parameters
            at = atoms
            cl = calculator
            if update_calc_settings.keys():
                setup_tasks.append(
                    generate_update_calc_task(calculator, update_calc_settings)
                )

            setup_tasks.append(generate_mod_calc_task(at, cl, "calculator", "kgrid"))
            cl = "calculator"
    else:
        at = None
        cl = None
    job_tasks = [generate_task(ts, fw_settings, at, cl) for ts in task_spec_list]

    return Firework(
        setup_tasks + job_tasks, name=fw_settings["fw_name"], spec=fw_settings["spec"]
    )


def generate_fw(
    atoms, task_list, fw_settings, qadapter, update_settings=None, update_in_spec=True
):
    """Generates a FireWork

    Parameters
    ----------
    atoms : ase.atoms.Atoms
        ASE Atoms object to preform the calculation on
    task_list : list of TaskSpecs
        Definitions for the tasks to be run
    fw_settings : dict
        FireWork settings for the step
    qadapter : dict
        The queueadapter for the step
    update_settings : dict
        update calculator settings (Default value = None)
    update_in_spec : bool
        If True move the current out_spec to be in_spec (Default value = True)

    Returns
    -------
    fireworks.Firework
        A firework for the task

    """
    fw_settings = update_fw_settings(
        fw_settings, fw_settings["fw_name"], qadapter, update_in_spec=update_in_spec
    )
    fw_settings[
        "fw_name"
    ] += f"_{atoms.symbols.get_chemical_formula()}_{hash_atoms_and_calc(atoms)[0]}"

    if not update_settings:
        update_settings = {}

    if "in_spec_atoms" in fw_settings:
        at = fw_settings["in_spec_atoms"]
    else:
        at = atoms.copy()

    if "in_spec_calc" in fw_settings:
        cl = fw_settings["in_spec_calc"]
    else:
        cl = atoms.calc

    return generate_firework(
        task_list, at, cl, fw_settings, update_calc_settings=update_settings
    )


def generate_kgrid_fw(settings, atoms, fw_settings):
    """Generate a k-grid optimization Firework

    Parameters
    ----------
    settings : Settings
        settings settings where the task is defined
    atoms : ase.atoms.Atoms
        ASE Atoms object to preform the calculation on
    fw_settings : dict
        Firework settings for the step

    Returns
    -------
    fireworks.Firework
        Firework for the k-grid optimization

    """
    # Get queue adapter settings
    fw_settings["fw_name"] = "kgrid_opt"
    fw_settings["out_spec_k_den"] = "kgrid"

    qadapter = settings.optimize_kgrid.get("qadapter", {})

    func_kwargs = {
        "workdir": f"{settings.fireworks.workdir.remote}/{fw_settings['fw_name']}/",
        "trajectory_file": filenames.trajectory,
        "dfunc_min": settings.optimize_kgrid.get("dfunc_min", 1e-6),
    }

    if qadapter and "walltime" in qadapter:
        func_kwargs["walltime"] = str2time(qadapter["walltime"])

    task_spec = gen_kgrid_task_spec(func_kwargs)
    return generate_fw(atoms, task_spec, fw_settings, qadapter)


def generate_aims_relax_fw(settings, atoms, fw_settings, step):
    """Generates a Firework for the relaxation step

    Parameters
    ----------
    settings : Settings
        settings settings where the task is defined
    atoms : ase.atoms.Atoms
        ASE Atoms object to preform the calculation on
    fw_settings : dict
        Firework settings for the step
    step : str
        Basis Set parameters to use for the calculation

    Returns
    -------
    fireworks.Firework
        Firework for the relaxation step

    """
    relax_settings = settings.relaxation.copy()
    relax_settings.pop("use_aims_relaxation", None)

    relax_settings.update(settings.relaxation[step])
    qadapter = relax_settings.get("qadapter", {})

    if "condition_list" in relax_settings.get("stop_if", {}):
        relax_settings["stop_if"]["condition_list"] = [
            parse_conditional(condition)
            for condition in relax_settings["stop_if"]["condition_list"]
        ]

    abreviated_step = [bt[0] for bt in step.split("_")]
    fw_settings["fw_name"] = f"{'_'.join(abreviated_step)}_relax"

    func_kwargs = {
        "workdir": f"{settings.fireworks.workdir.remote}/{fw_settings['fw_name']}/"
    }
    fw_out_kwargs = {"calc_number": 0}

    task_spec = gen_aims_task_spec(func_kwargs, fw_out_kwargs)

    method = relax_settings.get("method", "trm")
    force_crit = relax_settings.get("fmax", 1e-3)
    relax_unit_cell = relax_settings.get("relax_unit_cell", "full")
    basis = relax_settings.get("basis")

    update_settings = {
        "relax_geometry": f"{method} {force_crit}",
        "basisset_type": basis,
        "relax_unit_cell": relax_unit_cell,
    }

    return generate_fw(atoms, task_spec, fw_settings, qadapter, update_settings, True)


def generate_relax_fw(settings, atoms, fw_settings, step):
    """Generates a Firework for the relaxation step

    Parameters
    ----------
    settings: Settings
        settings settings where the task is defined
    atoms: ase.atoms.Atoms, dict
        ASE Atoms object to preform the calculation on
    fw_settings: dict
        Firework settings for the step

    Returns
    -------
    Firework
        Firework for the relaxation step
    """
    relax_settings = settings.relaxation.copy()
    relax_settings.pop("use_aims_relaxation", None)

    relax_settings.update(settings.relaxation[step])
    qadapter = relax_settings.get("qadapter", {})

    if "condition_list" in relax_settings.get("stop_if", {}):
        relax_settings["stop_if"]["condition_list"] = [
            parse_conditional(condition)
            for condition in relax_settings["stop_if"]["condition_list"]
        ]

    fw_settings["fw_name"] = f"{step}_relax"

    update_settings = {}
    if settings.calculator.name.lower() == "aims":
        update_settings["basisset_type"] = relax_settings.get("basis")

    relax_settings[
        "workdir"
    ] = f"{settings.fireworks.workdir.remote}/{fw_settings['fw_name']}/"
    relax_settings["step"] = step
    task_spec = gen_relax_task_spec(relax_settings, fw_settings)

    return generate_fw(atoms, task_spec, fw_settings, qadapter, update_settings, True)


def generate_phonon_fw(settings, atoms, fw_settings, name, update_in_spec=True):
    """Generates a Firework for the phonon initialization

    Parameters
    ----------
    settings : Settings
        settings settings where the task is defined
    atoms : ase.atoms.Atoms or dict
        ASE Atoms object to preform the calculation on
    fw_settings : dict
        Firework settings for the step
    name : str
        either phonopy or phono3py

    Returns
    -------
    fireworks.Firework
        Firework for the relaxation step

    """
    qadapter = settings[name].get("qadapter", {})

    update_settings = {}
    if "basisset_type" in settings[name]:
        update_settings["basisset_type"] = settings[name].pop("basisset_type")

    if "socketio" in settings.calculator and settings.calculator.name.lower() == "aims":
        if isinstance(settings.calculator["socketio"], bool):
            port = 12345
            host = "localhost"
        else:
            port = settings.calculator.socketio["port"]
            host = settings.calculator.socketio.get("host", "localhost")
        update_settings["use_pimd_wrapper"] = [host, port]
    elif "use_pimd_wrapper" in settings.calculator.get("parameters", {}):
        update_settings["use_pimd_wrapper"] = settings.calculator.parameters.pop(
            "use_pimd_wrapper"
        )
        host, port = update_settings["use_pimd_wrapper"]
        if "UNIX" in host:
            host, unixsocket = None, host
        settings.calculator["socketio"] = DotDict(
            {"host": host, "port": port, "unixsocket": unixsocket}
        )

    if "walltime" in qadapter:
        settings[name]["walltime"] = str2time(qadapter["walltime"])

    fw_settings["fw_name"] = name
    natoms = len(atoms) * np.linalg.det(
        get_3x3_matrix(settings[name]["supercell_matrix"])
    )
    natoms = int(round(natoms))
    settings[name][
        "workdir"
    ] = f"{settings.fireworks.workdir.remote}/sc_natoms_{natoms}/{name}/"

    if name == "phonopy":
        func_kwargs = {"ph_settings": settings[name].copy()}
    elif name == "phono3py":
        func_kwargs = {"ph3_settings": settings[name].copy()}

    task_spec = gen_phonon_task_spec(func_kwargs, fw_settings)

    return generate_fw(
        atoms,
        task_spec,
        fw_settings,
        qadapter,
        update_settings,
        update_in_spec=update_in_spec,
    )


def generate_phonon_postprocess_fw(
    settings, atoms, fw_settings, name, prev_dos_fp=None
):
    """Generates a Firework for the phonon analysis

    Parameters
    ----------
    settings : Settings
        settings settings where the task is defined
    atoms : ase.atoms.Atoms
        ASE Atoms object to preform the calculation on
    fw_settings : dict
        Firework settings for the step
    name : str
        either phonopy or phono3py

    Returns
    -------
    fireworks.Firework
        Firework for the phonon analysis

    """
    if name == "phonopy":
        fw_settings["mod_spec_add"] = "ph"
        fw_settings["fw_name"] = "phonopy_analysis"
    else:
        fw_settings["fw_name"] = "phono3py_analysis"
        fw_settings["mod_spec_add"] = "ph3"
    fw_settings["mod_spec_add"] += "_forces"

    func_kwargs = settings[name].copy()

    if "condition_list" in func_kwargs.get("stop_if", {}):
        func_kwargs["stop_if"]["condition_list"] = [
            parse_conditional(condition)
            for condition in func_kwargs["stop_if"]["condition_list"]
        ]

    if "workdir" in func_kwargs:
        func_kwargs.pop("workdir")

    natoms = len(atoms) * np.linalg.det(
        get_3x3_matrix(settings[name]["supercell_matrix"])
    )
    natoms = int(round(natoms))

    func_kwargs[
        "analysis_workdir"
    ] = f"{settings.fireworks.workdir.local}/sc_natoms_{natoms}/{fw_settings['fw_name']}/"
    func_kwargs["init_workdir"] = f"{settings.fireworks.workdir.remote}/{name}/"
    func_kwargs["prev_dos_fp"] = prev_dos_fp

    task_spec = gen_phonon_analysis_task_spec(
        "vibes." + fw_settings["fw_name"][:-9] + ".postprocess.postprocess",
        func_kwargs,
        fw_settings["mod_spec_add"][:-7] + "_metadata",
        fw_settings["mod_spec_add"],
        fw_settings["mod_spec_add"][:-7] + "_times",
        False,
    )
    fw_settings[
        "fw_name"
    ] += f"_{atoms.symbols.get_chemical_formula()}_{hash_atoms_and_calc(atoms)[0]}"
    return generate_firework(task_spec, None, None, fw_settings=fw_settings.copy())


def generate_stat_samp_fw(settings, atoms, fw_settings):
    """Generates a Firework for the statistical sampling initialization

    Parameters
    ----------
    settings : settings.Settings
        settings settings object
    atoms : ase.Atoms or dict
        ASE Atoms object to preform the calculation on
    fw_settings : dict
        Firework settings for the step

    Returns
    -------
    fireworks.Firework
        Firework for the harmonic analysis initialization

    """
    fw_settings["fw_name"] = "stat_samp"
    qadapter = settings.statistical_sampling.get("qadapter", {})
    if not qadapter and "phonopy" in settings:
        qadapter = settings.phonopy.get("qadapter", {})

    if qadapter and "walltime" in qadapter:
        settings.statistical_sampling["walltime"] = str2time(qadapter["walltime"])

    add_qadapter = False
    if "phonopy" in settings and "convergence" in settings.phonopy:
        add_qadapter = True

    settings.statistical_sampling[
        "workdir"
    ] = f"{settings.fireworks.workdir.remote}/statistical_sampling/"

    if "phonon_file" not in settings.statistical_sampling:
        settings.statistical_sampling["phonon_file"] = get_phonon_file_location(
            settings, atoms
        )
    else:
        settings.statistical_sampling["phonon_file"] = str(
            Path(settings.statistical_sampling.phonon_file).absolute()
        )

    fw_settings.pop("in_spec_calc", None)
    fw_settings.pop("in_spec_atoms", None)
    fw_settings["from_db"] = False

    task_spec = gen_stat_samp_task_spec(
        settings.statistical_sampling, fw_settings, add_qadapter
    )
    return generate_fw(atoms, task_spec, fw_settings, qadapter, None, False)


def generate_stat_samp_postprocess_fw(settings, atoms, fw_settings):
    """Generates a Firework for the statistical sampling analysis

    Parameters
    ----------
    settings : settings.Settings
        settings settings object
    atoms : ase.Atoms or dict
        ASE Atoms object to preform the calculation on
    fw_settings : dict
        Firework settings for the step

    Returns
    -------
    fireworks.Firework
        Firework for the anharmonicity analysis

    """
    fw_settings["fw_name"] = "statistical_sampling_analysis"
    fw_settings["mod_spec_add"] = "stat_samp"
    fw_settings["mod_spec_add"] += "_forces"

    func_kwargs = settings.statistical_sampling.copy()
    if "workdir" in func_kwargs:
        func_kwargs.pop("workdir")

    if "condition_list" in func_kwargs.get("stop_if", {}):
        func_kwargs["stop_if"]["condition_list"] = [
            parse_conditional(condition)
            for condition in func_kwargs["stop_if"]["condition_list"]
        ]

    func_kwargs[
        "analysis_workdir"
    ] = f"{settings.fireworks.workdir.local}/{fw_settings['fw_name']}/"

    task_spec = gen_stat_samp_analysis_task_spec(
        func_kwargs,
        fw_settings["mod_spec_add"][:-7] + "_metadata",
        fw_settings["mod_spec_add"],
        False,
    )
    fw_settings[
        "fw_name"
    ] += f"_{atoms.symbols.get_chemical_formula()}_{hash_atoms_and_calc(atoms)[0]}"
    return generate_firework(task_spec, None, None, fw_settings=fw_settings.copy())


def generate_aims_fw(settings, atoms, fw_settings):
    """Generates a Firework for the relaxation step

    Parameters
    ----------
    settings : Settings
        settings settings where the task is defined
    atoms : ase.atoms.Atoms or dict
        ASE Atoms object to preform the calculation on
    fw_settings : dict
        Firework settings for the step

    Returns
    -------
    fireworks.Firework
        Firework for the relaxation step

    """
    qadapter = settings.get("qadapter", {})

    fw_settings["fw_name"] = f"single_point"
    func_kwargs = {
        "workdir": f"{settings.fireworks.workdir.remote}/single_point_calculation/"
    }
    task_spec = gen_aims_task_spec(func_kwargs, {}, relax=False)

    return generate_fw(atoms, task_spec, fw_settings, qadapter, None, True)


def generate_gruniesen_fd_fw(
    settings, atoms, trajectory_file, constraints, fw_settings
):
    """Generate a FireWork to calculate the Gruniesen Parameter with finite differences

    Parameters
    ----------
    settings : Settings
        The Workflow Settings
    atoms : ase.atoms.Atoms
        The initial ASE Atoms object of the primitive cell
    trajecotry_file: str
        Path the the equilibrium phonon trajectory
    constraints : list of dict
        list of relevant constraint dictionaries for relaxations
    fw_settings : dict
        The FireWork Settings for the current job
    trajectory :


    Returns
    -------
    fireworks.Firework
        The Gruniesen setup firework

    """
    chem_form = atoms.symbols.get_chemical_formula(empirical=True, mode="metal")
    atoms_hash = hash_atoms_and_calc(atoms)[0]
    fw_settings["fw_name"] = f"gruniesen_setup_{chem_form}_{atoms_hash}"

    task_spec = gen_gruniesen_task_spec(settings, trajectory_file, constraints)
    return generate_firework(task_spec, None, None, fw_settings.copy())


def generate_md_fw(settings, atoms, fw_settings, qadapter=None, workdir=None):
    """Generate a FireWork to run a Molecular Dynamics calculation

    Parameters
    ----------
    settings: Settings
        The Workflow Settings
    atoms: ase.atoms.Atoms
        The initial ASE Atoms object of the primitive cell
    fw_settings: dict
        The FireWork Settings for the current job
    qadapter: dict
        The queueadapter for the step
    workdir: str
        The working directory for the calculation

    Returns
    -------
    Firework:
        The Molecular Dynamics setup firework
    """
    fw_settings = fw_settings.copy()
    fw_settings.pop("in_spec_atoms", None)
    fw_settings.pop("in_spec_calc", None)
    fw_settings["from_db"] = False

    if qadapter is None:
        qadapter = settings.md.pop("qadapter", None)

    md_settings = settings["md"].copy()

    if "phonon_file" not in settings.md and "phonopy" in settings:
        md_settings["phonon_file"] = get_phonon_file_location(settings, atoms, True)
    else:
        md_settings["phonon_file"] = str(Path(settings.md.phonon_file).absolute())

    temps = md_settings.pop("temperatures", None)
    if temps is None:
        temps = [md_settings.pop("temperature")]

    fireworks = []
    for temp in temps:
        fw_settings["fw_name"] = f"md_{temp}"
        md_set = md_settings.copy()
        if workdir is None:
            md_set[
                "workdir"
            ] = f"{settings.fireworks.workdir.remote}/{fw_settings['fw_name']}/"
        else:
            md_set["workdir"] = workdir
        md_set["temperature"] = temp
        task_spec = gen_md_task_spec(md_set, fw_settings)
        fireworks.append(
            generate_fw(atoms, task_spec, fw_settings, qadapter, None, False)
        )
    return fireworks
