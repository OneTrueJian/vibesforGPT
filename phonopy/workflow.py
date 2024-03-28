""" Provide a full highlevel phonopy workflow

    Input: geometry.in and settings.in
    Output: geometry.in.supercell and trajectory.son """

from vibes.calculate import calculate_socket
from vibes.calculator.context import CalculatorContext
from vibes.calculator.setup import setup_aims
from vibes.helpers import talk
from vibes.helpers.restarts import restart

from . import metadata2dict
from .postprocess import postprocess


def run_phonopy(**kwargs):
    """ high level function to run phonopy workflow """

    args = bootstrap(**kwargs)
    workdir = args["workdir"]

    talk(f"Run phonopy workflow in working directory\n  {workdir}")

    try:
        postprocess(**args)
        msg = "** Postprocess could be performed from previous calculations. Check"
        msg += f"\n**  {workdir}"
        exit(msg)
    except (FileNotFoundError, RuntimeError):
        completed = calculate_socket(**args)

    if not completed:
        restart(args["settings"])
    else:
        talk("Start postprocess.")
        postprocess(**args)
        talk("done.")


def bootstrap(ctx, dry=False):
    """load settings, prepare atoms, calculator, and phonopy

    Args:
        ctx (PhonopyContext): The context for the calculation
        dry (bool): prepare dry run

    Returns:
        dict: The necessary information to run the workflow with the following items
            atoms_to_calculate (list): list of the displaced supercells
            calculator (ase.calculators.calulator.Calculator):use to calculate forces
            metadata (dict): metadata for the phonon calculation
            workdir (str or Path): working directory for the calculation
            settings (Settings): settings for the workflow
    """
    if ctx.name.lower() == "phonopy":
        from vibes.phonopy.wrapper import preprocess
    elif ctx.name.lower() == "phono3py":
        from vibes.phono3py.wrapper import preprocess

    # Phonopy preprocess
    phonon, supercell, scs = preprocess(atoms=ctx.ref_atoms, **ctx.settings.obj)

    # if calculator not given, create an aims context for this calculation
    if ctx.settings.atoms and ctx.settings.atoms.calc:
        calculator = ctx.settings.atoms.calc
    else:
        aims_ctx = CalculatorContext(settings=ctx.settings, workdir=ctx.workdir)
        # set reference structure for aims calculation and make sure forces are computed
        aims_ctx.ref_atoms = supercell
        aims_ctx.settings.obj["compute_forces"] = True

        calculator = setup_aims(aims_ctx)

    # save metadata
    metadata = metadata2dict(phonon, calculator)

    return {
        "atoms_to_calculate": scs,
        "calculator": calculator,
        "metadata": metadata,
        "workdir": ctx.workdir,
        "settings": ctx.settings,
        "save_input": True,
        "backup_after_calculation": False,
        "dry": dry,
        **ctx.settings.obj,
    }
