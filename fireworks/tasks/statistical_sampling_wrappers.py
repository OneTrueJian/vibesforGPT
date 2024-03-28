"""Wrappers to prepare statistical sampling"""
import ase
import numpy as np

from vibes.cli.scripts.create_samples import generate_samples
from vibes.helpers.converters import atoms2dict, calc2dict, dict2atoms
from vibes.helpers.dict import AttributeDict
from vibes.helpers.k_grid import k2d, update_k_grid
from vibes.helpers.numerics import get_3x3_matrix
from vibes.helpers.supercell import make_supercell
from vibes.helpers.warnings import warn
from vibes.phonopy.postprocess import postprocess as postprocess_ph
from vibes.phonopy.utils import get_force_constants_from_trajectory
from vibes.phonopy.wrapper import get_debye_temperature
from vibes.settings import Settings
from vibes.structure.convert import to_Atoms
from vibes.trajectory import reader


def bootstrap(name="statistical_sampling", settings=None, **kwargs):
    """load settings, prepare atoms, calculator, and phonopy

    Parameters
    ----------
    name : str
        name of the statistical sampling task (Default value = "statistical_sampling")
    settings : Settings
        settings used for the workflow (Default value = None)
    **kwargs :


    Returns
    -------
    dict
        distorted supercells to calculate and metadata for the calculation

    """

    if settings is None:
        settings = Settings()

    stat_sample_settings = {}

    if name not in settings:
        warn(f"Settings do not contain {name} instructions.", level=1)
    else:
        stat_sample_settings.update(settings[name])

    _, ph_metadata = reader(stat_sample_settings["phonon_file"], get_metadata=True)
    ph_atoms = dict2atoms(ph_metadata["atoms"], ph_metadata["calculator"], False)

    # Get sampling metadata
    stat_sample_settings.update(kwargs)
    metadata = get_metadata(**stat_sample_settings)

    supercell = dict2atoms(metadata["supercell"])
    # Generate Samples
    fc = metadata.pop("force_constants_2D")

    td_cells = []
    for temp in metadata["temperatures"]:
        td_cells += generate_samples(
            supercell, temp, force_constants=fc, **metadata["generate_sample_args"]
        )

    calculator = ph_atoms.calc
    if calculator.name.lower() == "aims":
        kpt_density = k2d(ph_atoms, calculator.parameters["k_grid"])
        calculator = update_k_grid(td_cells[0], calculator, kpt_density)
        calculator.parameters.pop("scaled", False)

    # save metadata
    metadata["calculator"] = calc2dict(calculator)

    return {
        "atoms_to_calculate": td_cells,
        "calculator": calculator,
        "metadata": metadata,
        "workdir": name,
        "settings": stat_sample_settings,
    }


def get_metadata(phonon_file, temperatures=None, debye_temp_fact=None, **kwargs):
    """Sets ups the statistical sampling

    Parameters
    ----------
    phonon_file : str
        String to the phonopy trajectory
    temperatures : list (floats)
        List of temperatures to excite the phonons to (Default value = None)
    debye_temp_fact : list(floats)
        List of factors to multiply the debye temperature by to populate temperatures
        (Default value = None)
    **kwargs : dict
        list of terms used to generate distorted supercells
        Possible items:
        supercell_matrix : np.ndarray(int)
            Supercell matrix used to create the supercell from atoms
        n_samples : int
            number of samples to create (default: 1)
        mc_rattle : bool
            hiphive mc rattle
        quantum : bool
            use Bose-Einstein distribution instead of Maxwell-Boltzmann
        deterministic : bool
            create sample deterministically
        rng_seed : (int
            seed for random number generator


    Returns
    -------
    metadata : dict
        The metadata used for the statistical sampling steps

    """
    if temperatures is None:
        temperatures = ()

    # Set up supercell and Force Constants
    phonon = postprocess_ph(
        phonon_file, write_files=False, calculate_full_force_constants=False
    )

    atoms = to_Atoms(phonon.get_unitcell())

    if "supercell_matrix" in kwargs:
        supercell = make_supercell(atoms, get_3x3_matrix(kwargs["supercell_matrix"]))
    else:
        supercell = to_Atoms(phonon.get_supercell())

    force_constants = get_force_constants_from_trajectory(
        phonon_file, supercell, reduce_fc=True
    )
    fc_two_dim = get_force_constants_from_trajectory(
        phonon_file, supercell, two_dim=True
    )

    # If using Debye temperature calculate it
    if debye_temp_fact is not None:
        if phonon is None:
            raise IOError(
                "Debye Temp must be calculated with phonopy, please add phonon_file"
            )
        phonon.run_mesh([45, 45, 45])
        debye_temp = get_debye_temperature(phonon, 5e-3)[-1]
        temperatures += [tt * debye_temp for tt in debye_temp_fact]
    elif temperatures is None:
        raise IOError("temperatures must be given to do harmonic analysis")

    # Generate metadata
    metadata = {
        "ase_vesrsion": ase.__version__,
        "n_samples_per_temperature": kwargs.get("n_samples", 1),
        "temperatures": temperatures,
        "force_constants": force_constants,
        "force_constants_2D": fc_two_dim,
        "supercell": atoms2dict(supercell, reduce=True),
        "primitive": atoms2dict(atoms, reduce=True),
        "ph_supercell": atoms2dict(to_Atoms(phonon.get_supercell()), reduce=True),
        "atoms": atoms2dict(supercell, reduce=True),
    }

    if not kwargs.get("deterministic", True):
        if kwargs.get("rng_seed", None) is None:
            rng_seed = np.random.randint(2 ** 32 - 1)
        else:
            rng_seed = int(kwargs.get("rng_seed"))
    else:
        rng_seed = None

    metadata["generate_sample_args"] = {
        "n_samples": kwargs.get("n_samples", 1),
        "rattle": kwargs.get("mc_rattle", False),
        "quantum": kwargs.get("quantum", False),
        "deterministic": kwargs.get("deterministic", True),
        "zacharias": kwargs.get("plus_minus", True),
        "gauge_eigenvectors": kwargs.get("gauge_eigenvectors", True),
        "ignore_negative": kwargs.get("ignore_negative", False),
        "failfast": kwargs.get("failfast", True),
        "random_seed": rng_seed,
        "propagate": False,
    }

    return metadata


def bootstrap_stat_sample(
    atoms,
    calculator,
    kpt_density=None,
    qadapter=None,
    stat_samp_settings=None,
    fw_settings=None,
):
    """Initializes the statistical sampling task

    Parameters
    ----------
    atoms : ase.atoms.Atoms
        Atoms object of the primitive cell
    calculator : ase.calculators.Calculator
        Calculator for the force calculations
    kpt_density : float
        k-point density for the MP-Grid (Default value = None)
    qadapter : dict
        properties used for running things on queues
    stat_samp_settings : dict
        kwargs for statistical sampling setup (Default value = None)
    fw_settings : dict
        FireWork specific settings (Default value = None)

    Returns
    -------
    dict
        The output of vibes.statistical_sampling.workflow.bootstrap

    """
    if qadapter:
        fw_settings["spec"]["_queueadapter"] = qadapter

    settings = Settings(settings_file=None)
    settings.atoms = atoms
    if kpt_density:
        settings["control_kpt"] = AttributeDict({"density": kpt_density})

    settings["statistical_sampling"] = stat_samp_settings.copy()
    stat_samp_out = bootstrap(
        atoms=atoms, name="statistical_sampling", settings=settings
    )
    stat_samp_out["prefix"] = "stat_samp"
    stat_samp_out["settings"] = stat_samp_settings.copy()

    outputs = [stat_samp_out]
    return outputs


def postprocess_statistical_sampling(**kwargs):
    """Dummy function for post processing of statistical sampling

    Parameters
    ----------
    kwargs : dict
        keyword arguments for the task

    """
    return None
