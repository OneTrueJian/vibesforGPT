"""stress helpers"""

import numpy as np


def has_stress(atoms):
    """Check if atoms has computed stress

    This is needed because the aims socketio calculator returns an all-zero
    stress if stress computation was disabled for an MD step, as opposed to
    simply not including it in the results.

    The helper is supposed to be run on atoms with a SinglePointCalculator
    attached, not with a "live" calculator.

    Args:
        atoms: ase.Atoms object, with SinglePointCalculator attached.

    Returns:
        Whether atoms.calc has computed stress.

    """

    if "stress" in atoms.calc.results:
        # all zero means it wasn't computed
        return np.abs(atoms.calc.results["stress"]).max() > 1e-14
    else:
        return False
