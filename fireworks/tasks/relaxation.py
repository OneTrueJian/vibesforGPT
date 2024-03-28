"""Functions used to wrap around HiLDe Phonopy/Phono3py functions"""
from pathlib import Path

from jconfigparser.dict import DotDict

from vibes.helpers.k_grid import update_k_grid
from vibes.relaxation.context import RelaxationContext
from vibes.settings import Settings


def run(atoms, calculator, kpt_density=None, relax_settings=None, fw_settings=None):
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
    workdir = relax_settings.get("workdir", None)
    if workdir:
        workdir = Path(workdir)
        trajectory = workdir / "trajectory.son"
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        trajectory = "trajectory.son"
        workdir = Path(".")

    atoms.write(
        str(workdir / "geometry.in"), format="aims", scaled=True, geo_constrain=True
    )

    if calculator.name.lower() == "aims":
        if kpt_density is not None:
            update_k_grid(atoms, calculator, kpt_density, even=True)
        calculator.parameters.pop("sc_init_iter", None)

    settings = Settings(settings_file=None)
    settings_file = f"{workdir}/relaxation.in"
    settings._settings_file = settings_file
    settings["relaxation"] = DotDict(relax_settings)
    settings["calculator"] = DotDict({"name": calculator.name})
    settings["calculator"]["parameters"] = DotDict(calculator.parameters.copy())

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

    settings["files"] = DotDict({"geometry": str(workdir.absolute() / "geometry.in")})
    settings.write(f"{workdir}/relaxation.in")

    ctx = RelaxationContext(Settings(settings_file=settings_file), workdir, trajectory)
    return ctx.run()
