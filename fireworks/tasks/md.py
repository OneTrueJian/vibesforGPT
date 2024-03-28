"""Functions used to wrap around HiLDe Phonopy/Phono3py functions"""
from pathlib import Path

import numpy as np
from jconfigparser.dict import DotDict

from vibes.cli.scripts.create_samples import generate_samples
from vibes.filenames import filenames
from vibes.helpers.converters import dict2atoms
from vibes.helpers.k_grid import k2d, update_k_grid
from vibes.helpers.numerics import get_3x3_matrix
from vibes.helpers.supercell import make_supercell
from vibes.molecular_dynamics.context import MDContext
from vibes.phonopy.postprocess import extract_results, postprocess
from vibes.phonopy.utils import remap_force_constants
from vibes.settings import Settings
from vibes.trajectory import reader


def run(atoms, calculator, kpt_density=None, md_settings=None, fw_settings=None):
    """Creates a Settings object and passes it to the bootstrap function

    Parameters
    ----------
    atoms: ase.atoms.Atoms
        Atoms object of the primitive cell
    calculator: ase.calculators.calulator.Calculator
        Calculator for the force calculations
    kpt_density: float
        k-point density for the MP-Grid
    md_settings: dict
        kwargs for md setup
    fw_settings: dict
        FireWork specific settings

    Returns
    -------
    completed: bool
        True if the workflow completed
    """
    workdir = md_settings.get("workdir", None)
    if workdir:
        workdir = Path(workdir)
        trajectory_file = workdir / filenames.trajectory
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        trajectory_file = None
        workdir = Path(".")

    if (
        "phonon_file" not in md_settings
        and not Path(workdir / filenames.atoms).exists()
    ):
        sc_matrix = md_settings.pop("supercell_matrix", np.eye(3))
        supercell = make_supercell(atoms, sc_matrix)
        supercell.write(str(workdir / "supercell.in"), format="aims", scaled=True)
        atoms.write(str(workdir / "unitcell.in"), format="aims", scaled=True)
        supercell.write(str(workdir / filenames.atoms), format="aims", scaled=True)
    else:
        if Path(md_settings["phonon_file"]).parent.name == "converged":
            sc_list = Path(md_settings["phonon_file"]).parents[1].glob("sc_n*")
            sc_list = sorted(sc_list, key=lambda s: int(str(s).split("_")[-1]))
            md_settings["phonon_file"] = str(sc_list[-1] / "phonopy/trajectory.son")

        _, ph_metadata = reader(md_settings["phonon_file"], get_metadata=True)
        ph_workdir = str(Path(md_settings["phonon_file"]).parent)
        ph_trajectory_file = str(Path(md_settings["phonon_file"]).name)
        phonon = postprocess(ph_trajectory_file, ph_workdir, verbose=False)
        extract_results(phonon, output_dir=workdir / "phonopy_output")

        atoms_dict = ph_metadata["primitive"]["atoms"]
        ph_atoms = dict2atoms(atoms_dict, ph_metadata["calculator"], False)
        calculator = ph_atoms.calc
        if calculator.name.lower() == "aims":
            kpt_density = k2d(ph_atoms, calculator.parameters["k_grid"])
        if not Path(workdir / "geometry.in").exists():
            sc = dict2atoms(ph_metadata["supercell"]["atoms"])
            sc_matrix = md_settings.pop(
                "supercell_matrix",
                get_3x3_matrix(ph_metadata["Phonopy"]["supercell_matrix"]),
            )
            supercell = make_supercell(ph_atoms, sc_matrix)

            fc = remap_force_constants(
                phonon.get_force_constants(), ph_atoms, sc, supercell, two_dim=True
            )

            atoms_md = generate_samples(
                supercell,
                md_settings["temperature"] * 1.05,
                n_samples=1,
                force_constants=fc,
                rattle=False,
                quantum=False,
                deterministic=False,
                zacharias=False,
                gauge_eigenvectors=False,
                ignore_negative=True,
                random_seed=np.random.randint(2 ** 32),
                propagate=0,
            )[0]

            supercell.write(str(workdir / "supercell.in"), format="aims", scaled=True)
            ph_atoms.write(str(workdir / "unitcell.in"), format="aims", scaled=True)
            info_str = atoms_md.info.pop("info_str")
            atoms_md.write(
                str(workdir / filenames.atoms),
                info_str=info_str,
                format="aims",
                scaled=True,
                velocities=True,
            )

    if kpt_density is not None:
        update_k_grid(atoms, calculator, kpt_density, even=True)

    calculator.parameters.pop("sc_init_iter", None)

    settings = Settings(
        settings_file=None, config_files=None, dct={"md": DotDict(md_settings)}
    )
    settings_file = str(workdir / "md.in")
    settings._settings_file = settings_file
    settings["md"] = DotDict(md_settings)
    settings["calculator"] = DotDict({"name": calculator.name})
    if calculator.name.lower() == "aims":
        settings["calculator"]["basissets"] = DotDict(
            {"default": calculator.parameters.pop("species_dir").split("/")[-1]}
        )

        host, port = calculator.parameters.pop("use_pimd_wrapper", [None, None])
        if "UNIX" in host:
            unixsocket = host
            host = None
        else:
            unixsocket = None
        if port:
            settings["calculator"]["socketio"] = DotDict(
                {"port": port, "host": host, "unixsocket": unixsocket}
            )
        else:
            settings["calculator"].pop("socketio", None)
    elif calculator.name.lower() == "emt":
        settings["calculator"].pop("socketio", None)

    settings["calculator"]["parameters"] = DotDict(calculator.parameters)
    settings["files"] = DotDict(
        {
            "geometry": str(workdir.absolute() / filenames.atoms),
            "primitive": str(workdir.absolute() / "unitcell.in"),
            "supercell": str(workdir.absolute() / "supercell.in"),
        }
    )
    settings.write(settings_file)
    ctx = MDContext(Settings(settings_file=settings_file), workdir, trajectory_file)

    return ctx.run()
