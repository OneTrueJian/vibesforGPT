"""Generate FWActions after setting Phonon Calculations"""

from fireworks import FWAction, Workflow
from vibes.fireworks.tasks.postprocess.phonons import (
    get_converge_phonon_update,
    time2str,
)
from vibes.fireworks.tasks.fw_out.check_conditionals import run_all_checks
from vibes.fireworks.utils.converters import phonon3_to_dict, phonon_to_dict
from vibes.fireworks.workflows.firework_generator import generate_firework
from vibes.fireworks.workflows.workflow_generator import process_phonons

from vibes.context import TaskContext

from vibes.helpers.converters import atoms2dict, calc2dict, dict2atoms
from vibes.helpers.k_grid import k2d
from vibes.helpers.watchdogs import str2time
from vibes.phonopy._defaults import keys, kwargs as ph_kwargs
from vibes.settings import Settings
from vibes.structure.convert import to_Atoms
from vibes.trajectory import reader


def post_init_mult_calcs(
    atoms_dict,
    calculator_dict,
    outputs,
    func,
    func_fw_out,
    func_kwargs,
    func_fw_kwargs,
    fw_settings,
):
    """postprocessing for initializing parallel force calculaitons

    Parameters
    ----------
    atoms_dict : dict
        The dictionary representation of the original atoms at the start of this job
    calculator_dict : dict
        The dictionary representation of the original calculator
    outputs : dict
        The outputs after setting up claculations
    func : str
        Function path to the main function
    func_fw_out : str
        Function path to the fw_out function
    func_kwargs : dict
        The kwargs for the main function
    func_fw_kwargs : dict
        The kwargs for the fw_out function
    fw_settings : dict
        The FireWorks settings

    Returns
    -------
    fireworks.FWAction
        The action that will run all force calculations

    """
    if fw_settings is None:
        fw_settings = {}

    update_spec = {}
    detours = []

    for out in outputs:
        prefix = out["prefix"]
        fw_set = fw_settings.copy()
        func_set = out["settings"]

        if prefix + "_settings" in func_fw_kwargs:
            func_set = dict(func_set, **func_fw_kwargs[prefix + "_settings"])
        if "serial" not in func_set:
            func_set["serial"] = True

        update_spec[prefix + "_metadata"] = out["metadata"]

        if "k_pt_density" in out:
            update_spec["kgrid"] = out["k_pt_density"]

        if "spec" in fw_set:
            fw_set["spec"].update(update_spec)
        else:
            fw_set["spec"] = update_spec.copy()

        fw_set["mod_spec_add"] = prefix + "_forces"
        fw_set["metadata_spec"] = prefix + "_metadata"

        if "calculator" in out:
            calculator_dict = calc2dict(out["calculator"])

        if (
            "prev_dos_fp" in fw_set
            and "walltime" in fw_set
            and "serial" in func_set
            and func_set["serial"]
        ):
            fw_set["walltime"] = time2str(
                str2time(fw_set["walltime"]) * len(out["atoms_to_calculate"])
            )

        detours = get_detours(
            out["atoms_to_calculate"],
            calculator_dict,
            prefix,
            func_set,
            fw_set,
            update_spec,
            atoms_dict,
            detours,
        )
    return FWAction(update_spec=update_spec, detours=detours)


def get_detours(
    atoms_to_calculate,
    calculator_dict,
    prefix,
    calc_kwargs,
    fw_settings,
    update_spec,
    atoms=None,
    detours=None,
):
    """Add a set of detours for force calculations

    Parameters
    ----------
    atoms_to_calculate : list of ase.atoms.Atoms
        List of structures to calculate forces for
    calculator_dict : dict
        Dictionary representation of the ase.calculators.calulator.Calculator
    prefix : str
        prefix to add to force calculations
    calc_kwargs : dict
        A set of kwargs for the Force calculations
    fw_settings : dict
        A dictionary describing all FireWorks settings
    update_spec : dict
        Parmeters to be added to the FireWorks spec
    atoms : ase.atoms.Atoms
        Initial ASE Atoms object representation of the structure (Default value = None)
    detours : list of fireworks.Firework
        Current list of force calculations to perform (Default value = None)

    Returns
    -------
    list of fireworks.Firework
        The updated detours object

    """
    if detours is None:
        detours = []
    fw_settings["time_spec_add"] = prefix + "_times"
    if "walltime" in calc_kwargs:
        if "spec" in fw_settings and "_queueadapter" in fw_settings["spec"]:
            if "walltime" in fw_settings["spec"]["_queueadapter"]:
                calc_kwargs["walltime"] = (
                    str2time(fw_settings["spec"]["_queueadapter"]["walltime"]) - 120
                )
            else:
                fw_settings["spec"]["_queueadapter"]["walltime"] = (
                    time2str(calc_kwargs["walltime"]) + 120
                )
        else:
            if "spec" not in fw_settings:
                fw_settings["spec"] = {}
            fw_settings["spec"]["_queueadapter"] = {
                "walltime": time2str(calc_kwargs["walltime"] + 120)
            }

    if calc_kwargs["serial"]:
        update_spec[prefix + "_calculated_atoms"] = [
            atoms2dict(at) for at in atoms_to_calculate
        ]
        update_spec[prefix + "_calculator"] = calculator_dict
        fw_settings["spec"].update(update_spec)
        fw_settings["calc_atoms_spec"] = prefix + "_calculated_atoms"
        fw_settings["calc_spec"] = prefix + "_calculator"
        return add_socket_calc_to_detours(
            detours, atoms, calc_kwargs, fw_settings, prefix
        )
    return add_single_calc_to_detours(
        detours, calc_kwargs, atoms_to_calculate, calculator_dict, fw_settings, prefix
    )


def add_socket_calc_to_detours(detours, atoms_dict, func_kwargs, fw_settings, prefix):
    """Generates a Firework to run a socket calculator and adds it to the detours

    Parameters
    ----------
    detours : list of fireworks.Firework
        Current list of detours
    atoms_dict : dict
        Dictionary representation of the initial structure
    func_kwargs : dict
        kwargs needed to do the socket I/O calculation
    fw_settings : dict
        FireWorks settings
    prefix : str
        ph for phonopy and ph3 for phono3py calculations

    Returns
    -------
    list of fireworks.Firework
        an updated detours list

    """
    calc_kwargs = {}
    calc_keys = ["trajectory_file", "workdir", "backup_folder", "walltime"]
    for key in calc_keys:
        if key in func_kwargs:
            calc_kwargs[key] = func_kwargs[key]
    fw_set = fw_settings.copy()
    formula = dict2atoms(atoms_dict).get_chemical_formula(mode="metal", empirical=True)
    fw_set["fw_name"] = prefix + f"_ser_F_{formula}"
    fw = generate_firework(
        func="vibes.fireworks.tasks.calculate_wrapper.wrap_calc_socket",
        func_fw_out="vibes.fireworks.tasks.fw_out.calculate.socket_calc_check",
        func_kwargs=calc_kwargs,
        atoms_calc_from_spec=False,
        inputs=[
            prefix + "_calculated_atoms",
            prefix + "_calculator",
            prefix + "_metadata",
        ],
        fw_settings=fw_set.copy(),
    )
    detours.append(fw)
    return detours


def add_single_calc_to_detours(
    detours, func_fw_kwargs, atoms_list, calculator_dict, fw_settings, prefix
):
    """Adds a group of Fireworks to run as single calculations

    Parameters
    ----------
    detours : list of fireworks.Firework
        Current list of detours
    func_fw_kwargs : dict
        kwargs needed for the FireWorks output function stage
    atoms_list : list of ase.atoms.Atoms
        List of supercells to perform force calculations on
    calculator_dict : dict
        Dictionary representing the ASE Calculator for the force calculations
    fw_settings : dict
        FireWorks settings
    prefix : str
        ph for phonopy and ph3 for phono3py calculations

    Returns
    -------
    list of fireworks.Firework
        an updated detours list

    """
    for i, sc in enumerate(atoms_list):
        if not sc:
            continue
        fw_settings = fw_settings.copy()
        fw_settings["from_db"] = False
        fw_settings.pop("kpoint_density_spec", None)
        sc.info["displacement_id"] = i
        sc_dict = atoms2dict(sc)
        for key, val in calculator_dict.items():
            sc_dict[key] = val
        calc_kwargs = {"workdir": func_fw_kwargs["workdir"] + f"/{i:05d}"}
        formula = dict2atoms(sc_dict).get_chemical_formula(mode="metal", empirical=True)

        fw_settings["fw_name"] = prefix + f"forces_{formula}_{i}"
        detours.append(
            generate_firework(
                func="vibes.fireworks.tasks.calculate_wrapper.wrap_calculate",
                func_fw_out="vibes.fireworks.tasks.fw_out.calculate.mod_spec_add",
                func_kwargs=calc_kwargs,
                atoms=sc_dict,
                calculator=calculator_dict,
                atoms_calc_from_spec=False,
                fw_settings=fw_settings,
            )
        )
    return detours


def add_phonon_to_spec(func, func_fw_out, *args, fw_settings=None, **kwargs):
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
        ouputs(phonopy.Phonopy): The Phonopy object from post-processing
    *args :

    **kwargs :


    Returns
    -------
    FWAction
        FWAction that adds the phonon_dict to the spec

    """
    trajectory_file = f"{kwargs['workdir']}/{kwargs['trajectory_file']}"
    _, metadata = reader(trajectory_file, True)
    calculator_dict = metadata["calculator"]
    calculator_dict["calculator"] = calculator_dict["calculator"].lower()
    if calculator_dict["calculator"] == "aims":
        k_pt_density = k2d(
            dict2atoms(metadata["supercell"]["atoms"]),
            calculator_dict["calculator_parameters"]["k_grid"],
        )
    else:
        k_pt_density = None

    phonon = kwargs["outputs"]
    if "stop_if" in kwargs:
        action = run_all_checks(phonon, kwargs["stop_if"])
        if action is not None:
            return action

    qadapter = {}
    if fw_settings and "spec" in fw_settings:
        qadapter = fw_settings["spec"].get("_queueadapter", None)
    if "phono3py" in args[0]:
        update_spec = {
            "ph3_dict": phonon3_to_dict(kwargs["outputs"]),
            "ph3_calculator": calculator_dict,
            "ph3_supercell": atoms2dict(to_Atoms(kwargs["outputs"].get_primitive())),
        }
    else:
        update_spec = {
            "ph_dict": phonon_to_dict(kwargs["outputs"]),
            "ph_calculator": calculator_dict,
            "ph_supercell": atoms2dict(to_Atoms(kwargs["outputs"].get_supercell())),
            "_queueadapter": qadapter,
        }
    update_spec["kgrid"] = k_pt_density
    return FWAction(update_spec=update_spec)


def converge_phonons(func, func_fw_out, *args, fw_settings=None, **kwargs):
    """Check phonon convergence and set up future calculations after a phonon calculation

    Parameters
    ----------
    func : str
        Path to the phonon analysis function
    func_fw_out : str
        Path to this function
    args : list
        list arguments passed to the phonon analysis
    fw_settings : dict
        Dictionary for the FireWorks specific systems (Default value = None)
    kwargs : dict
        Dictionary of keyword arguments with the following keys:
        outputs : phonopy.Phonopy
            The Phonopy object from post-processing
        serial : bool
            If True use a serial calculation
        init_workdir : str
            Path to the base phonon force calculations
        trajectory_file: str
            trajectory_file file name

    Returns
    -------
    fireworks.FWAction
        Increases the supercell size or adds the phonon_dict to the spec

    """
    if not fw_settings:
        fw_settings = {}
        fw_settings["spec"] = {}
    elif "spec" not in fw_settings:
        fw_settings["spec"] = {}

    fw_settings["from_db"] = False
    fw_settings.pop("in_spec_calc", None)
    fw_settings.pop("in_spec_atoms", None)

    fw_settings["spec"]["kgrid"] = args[-1]

    phonon = kwargs.pop("outputs")
    workdir = kwargs.pop("workdir")
    trajectory_file = kwargs.pop("trajectory_file")

    conv, update_job = get_converge_phonon_update(
        workdir, trajectory_file, args[1], phonon, **kwargs
    )

    if not kwargs.get("serial", True):
        update_job["expected_walltime"] /= len(
            phonon.get_supercells_with_displacements()
        )

    if conv:
        if "stop_if" in kwargs:
            action = run_all_checks(phonon, kwargs["stop_if"])
            if action is not None:
                return action

        qadapter = None
        qadapter = fw_settings["spec"].get("_queueadapter", None)
        update_spec = dict(_queueadapter=qadapter, **update_job)
        update_spec["kgrid"] = args[-1]
        return FWAction(update_spec=update_spec)
    elif "n_atoms" in [
        cond[0] for cond in kwargs.get("stop_if", {}).get("condition_list", [])
    ]:
        condition = [
            cond for cond in kwargs["stop_if"]["condition_list"] if cond[0] == "n_atoms"
        ][0]
        if getattr(len(update_job["ph_supercell"]), f"__{condition[1]}__")(
            condition[2]
        ):
            return FWAction(defuse_workflow=True)

    fw_settings["in_spec_calc"] = "ph_calculator"
    update_spec = {
        "ph_calculator": update_job["ph_calculator"],
        "prev_dos_fp": update_job["prev_dos_fp"],
    }

    fw_settings["spec"].update(update_spec)

    update_spec["ph_supercell"] = update_job["ph_supercell"]
    func_kwargs = {
        "type": "phonopy",
        "displacement": update_job["displacement"],
        "supercell_matrix": update_job["supercell_matrix"],
        "serial": kwargs["serial"],
        "converge_phonons": True,
        "prev_wd": update_job["prev_wd"],
    }
    kwargs.update(func_kwargs)

    fw_settings["spec"]["prev_dos_fp"] = update_job["prev_dos_fp"]

    if "_queueadapter" in fw_settings["spec"]:
        func_kwargs.pop("walltime", None)
        fw_settings["spec"]["_queueadapter"].pop("queue", None)
        fw_settings["spec"]["_queueadapter"]["walltime"] = time2str(
            update_job["expected_walltime"] + 120
        )
        fw_settings["spec"]["_queueadapter"]["ntasks"] = update_job["ntasks"]
        qadapter = fw_settings["spec"]["_queueadapter"]
    else:
        qadapter = None

    primitive = to_Atoms(phonon.get_primitive())

    _, metadata = reader(f"{workdir}/{trajectory_file}", True)

    fireworks_dct = {
        "name": "phonon_continuation",
        "workdir": {
            "local": update_job["analysis_wd"],
            "remote": update_job["init_workdir"],
        },
    }

    phonopy_dct = {
        "serial": kwargs["serial"],
        keys.supercell_matrix: update_job["supercell_matrix"],
        keys.symprec: metadata["Phonopy"][keys.symprec],
        keys.displacement: update_job["displacement"],
        keys.is_diagonal: metadata["Phonopy"].get(
            keys.is_diagonal, ph_kwargs[keys.is_diagonal]
        ),
        keys.is_plusminus: metadata["Phonopy"].get(
            keys.is_plusminus, ph_kwargs[keys.is_plusminus]
        ),
        "qadapter": qadapter,
        "convergence": {
            "minimum_similarity_score": kwargs["conv_crit"],
            "sc_matrix_base": update_job["sc_matrix_base"],
        },
    }

    calculator_dct = {
        "name": update_job["ph_calculator"]["calculator"],
        "parameters": update_job["ph_calculator"]["calculator_parameters"],
    }

    if calculator_dct["name"] == "aims":
        socketio = calculator_dct["parameters"].get("use_pimd_wrapper")
        if socketio:
            port = socketio[1]
            host = socketio[0]
            unixsocket = None
            if "UNIX" in host:
                unixsocket = host
                host = None

            calculator_dct["socketio"] = {
                "host": host,
                "port": port,
                "unixsocket": unixsocket,
            }

        k_grid = calculator_dct["parameters"].get("k_grid")
        if k_grid:
            calculator_dct["kpoints"] = {"density": args[-1]}

    basis = calculator_dct["parameters"].get("species_dir")
    if basis:
        basis = basis.split("/")[-1]
        calculator_dct["basissets"] = {"default": basis}

    dct = {
        "fireworks": fireworks_dct,
        "phonopy": phonopy_dct,
        "calculator": calculator_dct,
    }

    settings = Settings(dct=dct)
    workflow = TaskContext(name=None, settings=settings)
    workflow.atoms = primitive

    init_fw, analysis_fw = process_phonons(
        workflow, primitive, fw_settings, basis, False, update_job["prev_dos_fp"]
    )

    detours = [init_fw, analysis_fw]
    wf = Workflow(detours, {init_fw: [analysis_fw]})
    return FWAction(detours=wf, update_spec=update_spec)
