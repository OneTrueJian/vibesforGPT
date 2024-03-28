"""Generates TaskSpec Objects"""
import numpy as np

from vibes.filenames import filenames
from vibes.fireworks.tasks.task_spec import TaskSpec
from vibes.helpers.numerics import get_3x3_matrix


def gen_phonon_task_spec(func_kwargs, fw_settings=None):
    """Generate a parallel Phononpy or Phono3py calculation task

    Parameters
    ----------
    func_kwargs : dict
        The defined kwargs for func
    fw_settings : dict
        Settings used by fireworks to place objects in the right part of the
        MongoDB (Default value = None)

    Returns
    -------
    TaskSpec
        The specification object of the task

    """
    if fw_settings is not None:
        fw_settings = fw_settings.copy()
    kwargs_init = {}
    kwargs_init_fw_out = {}

    preprocess_keys = {
        "ph_settings": ["supercell_matrix", "displacement"],
        "ph3_settings": ["supercell_matrix", "displacement", "cutoff_pair_distance"],
    }
    out_keys = ["walltime", "trajectory_file", "backup_folder", "serial"]

    for set_key in ["ph_settings", "ph3_settings"]:
        if set_key in func_kwargs:
            kwargs_init[set_key] = {}
            if "workdir" in func_kwargs[set_key]:
                wd = func_kwargs[set_key]["workdir"]
            else:
                wd = "."
            kwargs_init_fw_out[set_key] = {"workdir": wd}
            for key, val in func_kwargs[set_key].items():
                if key in preprocess_keys[set_key]:
                    kwargs_init[set_key][key] = val
                if key in out_keys:
                    kwargs_init_fw_out[set_key][key] = val

    inputs = ["kgrid"]
    args = []

    return TaskSpec(
        "vibes.fireworks.tasks.phonopy_phono3py_functions.bootstrap_phonon",
        "vibes.fireworks.tasks.fw_out.phonons.post_init_mult_calcs",
        True,
        kwargs_init,
        inputs=inputs,
        args=args,
        func_fw_out_kwargs=kwargs_init_fw_out,
        make_abs_path=False,
    )


def gen_stat_samp_task_spec(func_kwargs, fw_settings=None, add_qadapter=False):
    """Generate a Harmonic Analysis task

    Parameters
    ----------
    func_kwargs : dict
        The defined kwargs for func
    fw_settings : dict
        Settings used by fireworks to place objects in the right part of the
        MongoDB (Default value = None)

    Returns
    -------
    TaskSpec
        The specification object of the task

    """
    preprocess_keys = [
        "supercell_matrix",
        "phonon_file",
        "temperatures",
        "debye_temp_fact",
        "n_samples",
        "rattle",
        "quantum",
        "deterministic",
        "plus_minus",
        "gauge_eigenvectors",
        "ignore_negative",
        "random_seed",
    ]

    out_keys = ["walltime", "trajectory_file", "backup_folder", "serial"]
    kwargs_init = {}
    kwargs_init_fw_out = {}

    kwargs_init["stat_samp_settings"] = {}
    kwargs_init_fw_out["stat_samp_settings"] = {}

    if "workdir" in func_kwargs:
        wd = func_kwargs["workdir"]
    else:
        wd = "."

    kwargs_init_fw_out["stat_samp_settings"] = {"workdir": wd}
    kwargs_init["stat_samp_settings"] = {"workdir": wd}

    for key, val in func_kwargs.items():
        if key in preprocess_keys:
            kwargs_init["stat_samp_settings"][key] = val
        if key in out_keys:
            kwargs_init_fw_out["stat_samp_settings"][key] = val

    if fw_settings and "kpoint_density_spec" in fw_settings:
        inputs = [fw_settings["kpoint_density_spec"]]
        args = []
        if "kpt_density" in func_kwargs:
            del func_kwargs["kpt_density"]
    elif "kpt_density" in func_kwargs:
        args = [func_kwargs.pop("kpt_density")]
    else:
        inputs = []
        args = [None]

    if add_qadapter:
        inputs.append("_queueadapter")

    return TaskSpec(
        "vibes.fireworks.tasks.statistical_sampling_wrappers.bootstrap_stat_sample",
        "vibes.fireworks.tasks.fw_out.phonons.post_init_mult_calcs",
        True,
        kwargs_init,
        args=args,
        inputs=inputs,
        func_fw_out_kwargs=kwargs_init,
        make_abs_path=False,
    )


def gen_phonon_analysis_task_spec(
    func, func_kwargs, metakey, forcekey, timekey, make_abs_path=False
):
    """Generate a serial Phononpy or Phono3py calculation task

    Parameters
    ----------
    func : str
        The function path to the serial calculator
    func_kwargs : dict
        The defined kwargs for func
    metakey : str
        Key to find the phonon calculation's metadata to recreate the trajectory
    forcekey : str
        Key to find the phonon calculation's force data to recreate the trajectory
    timekey : str
        Key to find the time needed for the phonon supercell calculations
    make_abs_path : bool
        If True make the paths of directories absolute (Default value = False)

    Returns
    -------
    TaskSpec
        The specification object of the task

    """
    if "workdir" in func_kwargs and "init_workdir" not in func_kwargs:
        func_kwargs["init_workdir"] = func_kwargs["workdir"]

    if "analysis_workdir" in func_kwargs:
        func_kwargs["workdir"] = func_kwargs["analysis_workdir"]
    elif "workdir" not in func_kwargs:
        func_kwargs["workdir"] = "."

    conv = False
    if "convergence" in func_kwargs:
        if isinstance(func_kwargs["convergence"], bool):
            conv = func_kwargs["convergence"]
            conv_crit = 0.80
        else:
            conv = True
            conv_crit = func_kwargs["convergence"].get("minimum_similarity_score", 0.80)
            supercell_matrix = get_3x3_matrix(func_kwargs["supercell_matrix"]).flatten()
            sc_matrix_base = get_3x3_matrix(
                func_kwargs["convergence"].get("sc_matrix_base", supercell_matrix)
            ).flatten()

            inds = np.where(np.abs(sc_matrix_base) > 0.99)[0]
            quot = supercell_matrix[inds] / sc_matrix_base[inds]

            try:
                assert np.all(quot == quot[0])
            except AssertionError:
                raise ValueError(
                    "supercell_matrix is not a scalar multiple of sc_matrix_base"
                )

            try:
                assert np.all(quot > 0.99)
            except AssertionError:
                raise ValueError("sc_matrix_base is smaller than supercell_matrix")

            try:
                assert np.all(np.abs(quot - np.round(quot)) < 1e-10)
            except AssertionError:
                raise ValueError(
                    "supercell_matrix is not an integer scalar multiple of sc_matrix_base"
                )

            func_kwargs["sc_matrix_base"] = [
                int(round(sc_el)) for sc_el in sc_matrix_base.flatten()
            ]

    if conv:
        func_out = "vibes.fireworks.tasks.fw_out.phonons.converge_phonons"
        func_kwargs["conv_crit"] = conv_crit
        func_kwargs.pop("convergence")
    else:
        func_out = "vibes.fireworks.tasks.fw_out.phonons.add_phonon_to_spec"

    if "trajectory_file" not in func_kwargs:
        func_kwargs["trajectory_file"] = filenames.trajectory
    task_spec_list = []
    task_spec_list.append(
        TaskSpec(
            "vibes.fireworks.tasks.phonopy_phono3py_functions.collect_to_trajectory",
            "vibes.fireworks.tasks.fw_out.general.fireworks_no_mods_gen_function",
            False,
            args=[func_kwargs["workdir"], func_kwargs["trajectory_file"]],
            inputs=[forcekey, metakey],
            make_abs_path=make_abs_path,
        )
    )
    task_spec_list.append(
        TaskSpec(
            "vibes.fireworks.tasks.phonopy_phono3py_functions.phonon_postprocess",
            func_out,
            False,
            args=[func],
            inputs=[timekey, "kgrid"],
            func_kwargs=func_kwargs,
            make_abs_path=make_abs_path,
        )
    )
    return task_spec_list


def gen_stat_samp_analysis_task_spec(
    func_kwargs, metakey, forcekey, make_abs_path=False
):
    """Generate a serial Phononpy or Phono3py calculation task

    Parameters
    ----------
    func_kwargs : dict
        The defined kwargs for func
    metakey : str
        Key to find the phonon calculation's metadata to recreate the trajectory
    forcekey : str
        Key to find the phonon calculation's force data to recreate the trajectory
    make_abs_path : bool
        If True make the paths of directories absolute (Default value = False)

    Returns
    -------
    TaskSpec
        The specification object of the task

    """
    if "analysis_workdir" in func_kwargs:
        func_kwargs["workdir"] = func_kwargs["analysis_workdir"]
    elif "workdir" not in func_kwargs:
        func_kwargs["workdir"] = "."

    if "trajectory_file" not in func_kwargs:
        func_kwargs["trajectory_file"] = filenames.trajectory

    task_spec_list = []
    task_spec_list.append(
        TaskSpec(
            "vibes.fireworks.tasks.phonopy_phono3py_functions.collect_to_trajectory",
            "vibes.fireworks.tasks.fw_out.general.fireworks_no_mods_gen_function",
            False,
            args=[func_kwargs["workdir"], func_kwargs["trajectory_file"]],
            inputs=[forcekey, metakey],
            make_abs_path=make_abs_path,
        )
    )
    stat_samp_head = "vibes.fireworks.tasks.statistical_sampling_wrappers"
    fout_head = "vibes.fireworks.tasks.fw_out.statistical_sampling"
    task_spec_list.append(
        TaskSpec(
            f"{stat_samp_head}.postprocess_statistical_sampling",
            f"{fout_head}.add_stat_samp_to_spec",
            False,
            args=[],
            inputs=[],
            func_kwargs=func_kwargs,
            make_abs_path=make_abs_path,
        )
    )
    return task_spec_list


def gen_aims_task_spec(
    func_kwargs, func_fw_out_kwargs, make_abs_path=False, relax=True
):
    """Gets the task spec for an FHI-aims calculations

    Parameters
    ----------
    func_kwargs : dict
        The defined kwargs for func
    func_fw_outkwargs : dict
        The defined kwargs for fw_out
    make_abs_path : bool
        If True make the paths of directories absolute (Default value = False)
    relax : bool
        If True it is a relaxation (Default value = True)
    func_fw_out_kwargs :


    Returns
    -------
    TaskSpec
        The task_spec for the calculation

    """
    fw_out = "vibes.fireworks.tasks.fw_out.aims.check_aims_complete"
    if not relax:
        fw_out = "vibes.fireworks.tasks.fw_out.general.fireworks_no_mods"
    return TaskSpec(
        "vibes.fireworks.tasks.calculate_wrapper.wrap_calculate",
        fw_out,
        True,
        func_kwargs,
        func_fw_out_kwargs=func_fw_out_kwargs,
        make_abs_path=make_abs_path,
    )


def gen_kgrid_task_spec(func_kwargs, make_abs_path=False):
    """Gets the task spec for a k-grid optimization

    Parameters
    ----------
    func_kwargs : dict
        The defined kwargs for func
    make_abs_path : bool
        If True make the paths of directories absolute (Default value = False)

    Returns
    -------
    TaskSpec
        The TaskSpec for the kgrid optimization

    """
    return TaskSpec(
        "vibes.k_grid.converge_kgrid.converge_kgrid",
        "vibes.fireworks.tasks.fw_out.optimizations.check_kgrid_opt_completion",
        True,
        func_kwargs,
        make_abs_path=make_abs_path,
    )


def gen_gruniesen_task_spec(settings, trajectory_file, constraints):
    """Generate a TaskSpec for setting up a Gruniesen parameter calculation

    Parameters
    ----------
    settings : Settings
        The workflow settings
    trajectory_file: str
        Path the the equilibrium phonon trajectory
    constraints : list of dict
        list of relevant constraint dictionaries for relaxations

    Returns
    -------
    TaskSpec
        The specification object of the Gruniesen setup task

    """
    task_spec_list = [
        TaskSpec(
            "vibes.fireworks.tasks.phonopy_phono3py_functions.setup_gruneisen",
            "vibes.fireworks.tasks.fw_out.general.add_additions_to_spec",
            False,
            args=[settings, trajectory_file, constraints],
            inputs=["_queueadapter", "kgrid"],
            make_abs_path=False,
        )
    ]
    return task_spec_list


def gen_md_task_spec(md_settings, fw_settings=None):
    """Generate a TaskSpec for setting up a Gruniesen parameter calculation

    Returns
    -------
    TaskSpec
        The specification object of the MD task
    """

    if fw_settings and "kpoint_density_spec" in fw_settings:
        inputs = [fw_settings["kpoint_density_spec"]]
        args = []
        if "kpt_density" in md_settings:
            del md_settings["kpt_density"]
    elif "kpt_density" in md_settings:
        inputs = []
        args = [md_settings.pop("kpt_density")]
    else:
        inputs = []
        args = [None]

    task_spec_list = [
        TaskSpec(
            "vibes.fireworks.tasks.md.run",
            "vibes.fireworks.tasks.fw_out.md.check_md_finish",
            True,
            {"md_settings": md_settings},
            args=args,
            inputs=inputs,
            make_abs_path=False,
        )
    ]
    return task_spec_list


def gen_relax_task_spec(relax_settings, fw_settings=None):
    """Generate a TaskSpec for setting up a Gruniesen parameter calculation

    Returns
    -------
    TaskSpec
        The specification object of the MD task
    """

    inputs = []
    args = [None]

    task_spec_list = [
        TaskSpec(
            "vibes.fireworks.tasks.relaxation.run",
            "vibes.fireworks.tasks.fw_out.relaxation.check_relax_finish",
            True,
            {"relax_settings": relax_settings},
            args=args,
            inputs=inputs,
            make_abs_path=False,
        )
    ]
    return task_spec_list
