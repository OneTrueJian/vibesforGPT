"""Postprocessing for phonon clalculations"""
from pathlib import Path
from shutil import copyfile, rmtree

import numpy as np

from vibes.filenames import filenames
from vibes.fireworks.utils.converters import phonon_to_dict
from vibes.helpers.converters import atoms2dict
from vibes.helpers.k_grid import update_k_grid
from vibes.helpers.paths import cwd
from vibes.materials_fp.material_fingerprint import (
    fp_tup,
    get_phonon_dos_fp,
    scalar_product,
)
from vibes.phonopy.utils import remap_force_constants
from vibes.phonopy.wrapper import preprocess as ph_preprocess
from vibes.settings import Settings
from vibes.helpers.supercell import get_commensurate_q_points
from vibes.structure.convert import to_Atoms
from vibes.trajectory import reader


def time2str(n_sec):
    """Converts a number of seconds into a time string

    Parameters
    ----------
    n_sec : int
        A time presented as a number of seconds

    Returns
    -------
    str
        A string representing a specified time

    """
    secs = int(n_sec % 60)
    mins = int(n_sec / 60) % 60
    hrs = int(n_sec / 3600) % 24
    days = int(n_sec / 86400)
    return f"{days}-{hrs}:{mins}:{secs}"


def get_base_work_dir(wd):
    """Converts wd to be it's base (no task specific directories)

    Parameters
    ----------
    wd : str
        Current working directory

    Returns
    -------
    str
        The base working directory for the workflow

    """
    wd_list = wd.split("/")
    # remove analysis directories from path
    while "phonopy_analysis" in wd_list:
        wd_list.remove("phonopy_analysis")

    while "phono3py_analysis" in wd_list:
        wd_list.remove("phono3py_analysis")

    while "phonopy" in wd_list:
        wd_list.remove("phonopy")

    while "phono3py" in wd_list:
        wd_list.remove("phono3py")

    # Remove all "//" from the path
    while "" in wd_list:
        wd_list.remove("")

    # If starting from root add / to beginning of the path
    if wd[0] == "/":
        wd_list = [""] + wd_list

    # Remove "sc_natoms_???" to get back to the base directory
    if len(wd_list[-1]) > 10 and wd_list[-1][:10] == "sc_natoms_":
        return "/".join(wd_list[:-1])
    return "/".join(wd_list)


def get_memory_expectation(new_supercell, calculator, k_pt_density, workdir):
    """Runs a dry_run of the new calculation and gets the estimated memory usage

    Parameters
    ----------
    new_supercell : ase.atoms.Atoms
        The structure to get the memory estimation for
    calculator: ase.atoms.Calculator
        The ASE Calculator to be used
    k_pt_density : list of floats
        The k-point density in all directions
    workdir : str
        Path to working directory

    Returns
    -------
    total_mem : float
        The expected memory of the calculation, scaling based on empirical tests

    """
    settings = Settings()
    calculator.parameters["dry_run"] = True
    calculator.parameters.pop("use_local_index", None)
    calculator.parameters.pop("load_balancing", None)
    calculator.command = settings.machine.aims_command
    bs_base = settings.machine.basissetloc
    calculator.parameters["species_dir"] = (
        bs_base + "/" + calculator.parameters["species_dir"].split("/")[-1]
    )
    calculator.parameters["compute_forces"] = False
    update_k_grid(new_supercell, calculator, k_pt_density, even=True)
    new_supercell.set_calculator(calculator)
    mem_expect_dir = workdir + "/.memory_expectation"
    with cwd(mem_expect_dir, mkdir=True):
        try:
            new_supercell.calculator.calculate()
        except RuntimeError:
            calculator.parameters["dry_run"] = False
        lines = open(filenames.output.aims).readlines()
    rmtree(mem_expect_dir)

    for line in lines:
        if "Maximum number of basis functions" in line:
            n_basis = int(line.split(":")[1])
            continue

        if "Number of Kohn-Sham states (occupied + empty)" in line:
            n_states = int(line.split(":")[1])
            continue

        if "Size of matrix packed + index [n_hamiltonian_matrix_size]" in line:
            n_hamiltonian_matrix_size = int(line.split(":")[1])
            break

        if "Number of k-point" in line:
            n_kpt = int(line.split(":")[1])
            continue
    n_spin = 1
    total_mem = n_basis * n_basis
    total_mem += n_hamiltonian_matrix_size
    total_mem += n_basis * n_states * n_spin * n_kpt
    total_mem *= 16 * 10
    return total_mem


def check_phonon_conv(dos_fp, prev_dos_fp, conv_crit):
    """Checks if the density of state finger prints are converged

    Parameters
    ----------
    dos_fp : vibes.materials_fp.material_fingerprint.fp_tup
        Current fingerprint
    prev_dos_fp : vibes.materials_fp.material_fingerprint.fp_tup
        Fingerprint of the previous step
    conv_crit : float
        convergence criteria

    Returns
    -------
    bool
        True if conv_criteria is met

    """
    if not isinstance(prev_dos_fp, fp_tup):
        for ll in range(4):
            prev_dos_fp[ll] = np.array(prev_dos_fp[ll])
        prev_dos_fp = fp_tup(
            prev_dos_fp[0], prev_dos_fp[1], prev_dos_fp[2], prev_dos_fp[3]
        )
    return (
        scalar_product(dos_fp, prev_dos_fp, col=1, pt=0, normalize=False, tanimoto=True)
        >= conv_crit
    )


def get_converge_phonon_update(
    workdir,
    trajectory_file,
    calc_times,
    phonon,
    conv_crit=0.95,
    prev_dos_fp=None,
    init_workdir="./",
    **kwargs,
):
    """Check phonon convergence and set up future calculations after a phonon calculation

    Parameters
    ----------
    workdir : str
        path to the working directory
    trajectory : str
        name of hte trajectory.son file
    calc_times : list of floats
        Total calculation times for all structures
    ph : phonopy.Phonopy
        phonopy object for the current calculation
    conv_crit : float
        Convergence criteria for Tanimoto Similarity of phonon density of states
        (Default value = 0.95)
    prev_dos_fp : vibes.materials_fp.material_fingerprint.fp_tup
        Fingerprint of the previous step (Default value = None)
    init_workdir : str
        Path to the base phonon force calculations (Default value = "./")
    kwargs : dict
        Dictionary of keyword arguments
    **kwargs :


    Returns
    -------
    ph_conv : bool
        True if phonon model is converged with respect to the supercell size
    update_job : dict
        Dictionary describing all necessary job updates

    """
    calc_time = np.sum(calc_times)

    _, metadata = reader(f"{workdir}/{trajectory_file}", True)
    calculator_dict = metadata["calculator"]
    calculator_dict["calculator"] = calculator_dict["calculator"].lower()

    # Calculate the phonon DOS
    primitive = to_Atoms(phonon.get_primitive())
    supercell = to_Atoms(phonon.get_supercell())
    brav_pts = get_commensurate_q_points(primitive.cell, supercell.cell)
    for k in brav_pts:
        if np.any(phonon.get_frequencies(k) < -1.0e-1):
            raise ValueError("Negative frequencies at an included lattice point.")

    phonon.run_mesh([45, 45, 45])

    if "sc_matrix_base" not in kwargs:
        kwargs["sc_matrix_base"] = phonon.get_supercell_matrix()

    ind = np.where(np.array(kwargs["sc_matrix_base"]).flatten() != 0)[0][0]
    n_cur = int(
        round(
            phonon.get_supercell_matrix().flatten()[ind]
            / np.array(kwargs["sc_matrix_base"]).flatten()[ind]
        )
    )
    displacement = phonon._displacement_dataset["first_atoms"][0]["displacement"]
    disp_mag = np.linalg.norm(displacement)

    if n_cur > 1 and prev_dos_fp is None:
        sc_mat = (n_cur - 1) * np.array(kwargs["sc_matrix_base"]).reshape((3, 3))
        phonon_small, supercell, _ = ph_preprocess(
            to_Atoms(phonon.get_unitcell(), db=True), sc_mat, displacement=disp_mag
        )
        phonon_small.set_force_constants(
            remap_force_constants(
                phonon.get_force_constants(),
                to_Atoms(phonon.get_unitcell()),
                to_Atoms(phonon.get_supercell()),
                supercell,
                reduce_fc=True,
            )
        )
        brav_pts = get_commensurate_q_points(primitive.cell, supercell.cell)
        for k in brav_pts:
            if np.any(phonon_small.get_frequencies(k) < -1.0e-1):
                raise ValueError("Negative frequencies at an included lattice point.")

        phonon_small.run_mesh([45, 45, 45])
        phonon_small.run_total_dos(freq_pitch=0.01)
        n_bins = len(phonon_small.get_total_dos_dict()["frequency_points"])
        prev_dos_fp = get_phonon_dos_fp(phonon_small, nbins=n_bins)

    if prev_dos_fp:
        de = prev_dos_fp[0][1] - prev_dos_fp[0][0]
        min_f = prev_dos_fp[0][0] - 0.5 * de
        max_f = prev_dos_fp[0][-1] + 0.5 * de
        phonon.run_total_dos(freq_min=min_f, freq_max=max_f, freq_pitch=0.01)
    else:
        # If Not Converged update phonons
        phonon.run_total_dos(freq_pitch=0.01)

    # Get a phonon DOS Finger print to compare against the previous one
    n_bins = len(phonon.get_total_dos_dict()["frequency_points"])
    dos_fp = get_phonon_dos_fp(phonon, nbins=n_bins)

    # Get the base working directory
    init_workdir = get_base_work_dir(init_workdir)
    analysis_wd = get_base_work_dir(workdir)

    # Check phonon Convergence
    if prev_dos_fp:
        ph_conv = check_phonon_conv(dos_fp, prev_dos_fp, conv_crit)
    else:
        ph_conv = False

    # Check to see if phonons are converged
    if prev_dos_fp is not None and ph_conv:
        Path(f"{analysis_wd}/converged/").mkdir(exist_ok=True, parents=True)
        copyfile(
            f"{workdir}/{trajectory_file}", f"{analysis_wd}/converged/trajectory.son"
        )
        update_job = {
            "ph_dict": phonon_to_dict(phonon),
            "ph_calculator": calculator_dict,
            "ph_primitive": atoms2dict(to_Atoms(phonon.get_unitcell(), db=True)),
            "ph_time": calc_time / len(phonon.get_supercells_with_displacements()),
        }
        return True, update_job

    # Reset dos_fp to include full Energy Range for the material
    if prev_dos_fp:
        phonon.set_total_DOS(tetrahedron_method=True, freq_pitch=0.01)
        n_bins = len(phonon.get_total_dos_dict()["frequency_points"])
        dos_fp = get_phonon_dos_fp(phonon, nbins=n_bins)

    # If Not Converged update phonons
    sc_mat = (n_cur + 1) * np.array(kwargs["sc_matrix_base"]).reshape((3, 3))

    ratio = np.linalg.det(sc_mat) / np.linalg.det(phonon.get_supercell_matrix())
    phonon, _, _ = ph_preprocess(
        to_Atoms(phonon.get_unitcell(), db=True), sc_mat, displacement=displacement
    )

    if phonon.get_supercell().get_number_of_atoms() > 500:
        time_scaling = 3.0 * ratio ** 3.0
    else:
        time_scaling = 3.0 * ratio

    expected_walltime = max(calc_time * time_scaling, 1680)

    ntasks = int(np.ceil(phonon.supercell.get_number_of_atoms() * 0.75))

    # init_workdir += f"/sc_natoms_{phonon.get_supercell().get_number_of_atoms()}"
    # analysis_wd += f"/sc_natoms_{phonon.get_supercell().get_number_of_atoms()}"

    displacement = phonon._displacement_dataset["first_atoms"][0]["displacement"]
    disp_mag = np.linalg.norm(displacement)

    update_job = {
        "sc_matrix_base": kwargs["sc_matrix_base"],
        "supercell_matrix": sc_mat,
        "init_workdir": init_workdir,
        "analysis_wd": analysis_wd,
        "ntasks": ntasks,
        "expected_walltime": expected_walltime,
        "ph_calculator": calculator_dict,
        "ph_primitive": atoms2dict(to_Atoms(phonon.get_unitcell(), db=True)),
        "ph_supercell": atoms2dict(to_Atoms(phonon.get_supercell(), db=True)),
        "prev_dos_fp": dos_fp,
        "prev_wd": workdir,
        "displacement": disp_mag,
    }
    return False, update_job
