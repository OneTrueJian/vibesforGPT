""" Provide a full highlevel phonopy workflow

    Input: geometry.in and settings.in
    Output: geometry.in.supercell and trajectory.son """

from vibes.calculate import calculate_socket
from vibes.helpers import talk
from vibes.helpers.converters import atoms2dict, input2dict
from vibes.helpers.restarts import restart

from .context import CalculatorContext


def run_aims(ctx: CalculatorContext):
    """ high level function to run aims calculation

    Args:
        ctx: The context for the calculation

    """

    args = bootstrap(ctx)

    completed = calculate_socket(**args)

    if not completed:
        restart(ctx.settings)
    else:
        talk("done.")


def bootstrap(ctx: CalculatorContext) -> dict:
    """ load settings, prepare atoms and aims calculator

    Args:
        ctx: The context for the calculation

    Returns:
        All of the necessary objects to run the Aims calculation with the following items

        atoms_to_calculate: list of ase.atoms.Atoms
            The structures to be calculated
        calculator: ase.calculators.calulator.Calculator
            Calculator for all calculations
        metadata: dict
            The Metadata for the calculation
        workdir: str
            Path to the working direcotry
        settings: AimsSettings
            The settings used to generate this task
        backup_after_calculation: bool
            If True back up the calculation folder once completed

    """

    # find geometries
    atoms_to_calculate = ctx.atoms_to_calculate

    if not atoms_to_calculate:
        raise RuntimeError("no structures to compute.")

    calculator = ctx.calculator

    # save metadata
    metadata = input2dict(
        ctx.ref_atoms,
        calculator=calculator,
        settings=ctx.settings,
        primitive=ctx.primitive,
        supercell=ctx.supercell,
    )

    # save input files
    input_files = {}
    for file, atoms in zip(ctx.geometry_files, atoms_to_calculate):
        dct = {f"{file}": atoms2dict(atoms)}
        input_files.update(dct)

    metadata.update({"geometry_files": input_files})

    return {
        "atoms_to_calculate": atoms_to_calculate,
        "calculator": calculator,
        "metadata": metadata,
        "workdir": ctx.workdir,
        "settings": ctx.settings,
        "save_input": True,
        "backup_after_calculation": False,
    }
