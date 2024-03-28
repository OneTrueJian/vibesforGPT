""" Helpers for working with kpoint densites. """

import numpy as np

from vibes.helpers.utils import talk


def d2k(atoms, kptdensity=3.5, even=True):
    """Convert k-point density to Monkhorst-Pack grid size.

    inspired by [ase.calculators.calculator.kptdensity2monkhorstpack]

    Parameters
    ----------
    atoms: Atoms object
        Contains unit cell and information about boundary conditions.
    kptdensity: float or list of floats
        Required k-point density.  Default value is 3.5 point per Ang^-1.
    even: bool
        Round up to even numbers.

    Returns
    -------
    list
        Monkhorst-Pack grid size in all directions
    """
    recipcell = atoms.get_reciprocal_cell()
    return d2k_recipcell(recipcell, atoms.pbc, kptdensity, even)


def d2k_recipcell(recipcell, pbc, kptdensity=3.5, even=True):
    """Convert k-point density to Monkhorst-Pack grid size.

    Parameters
    ----------
    recipcell: ASE Cell object
        The reciprocal cell
    pbc: list of Bools
        If element of pbc is True then system is periodic in that direction
    kptdensity: float or list of floats
        Required k-point density.  Default value is 3.5 point per Ang^-1.
    even: bool
        Round up to even numbers.

    Returns
    -------
    list
        Monkhorst-Pack grid size in all directions
    """
    if not isinstance(kptdensity, list) and not isinstance(kptdensity, np.ndarray):
        kptdensity = 3 * [float(kptdensity)]
    kpts = []
    for i in range(3):
        if pbc[i]:
            k = 2 * np.pi * np.sqrt((recipcell[i] ** 2).sum()) * float(kptdensity[i])
            if even:
                kpts.append(2 * int(np.ceil(k / 2)))
            else:
                kpts.append(int(np.ceil(k)))
        else:
            kpts.append(1)
    return kpts


def k2d(atoms, k_grid=[2, 2, 2]):
    """Generate the kpoint density in each direction from given k_grid.

    Parameters
    ----------
    atoms: Atoms
        Atoms object of interest.
    k_grid: list
        k_grid that was used.

    Returns
    -------
    np.ndarray
        density of kpoints in each direction. result.mean() computes average density
    """
    recipcell = atoms.get_reciprocal_cell()
    densities = k_grid / (2 * np.pi * np.sqrt((recipcell ** 2).sum(axis=1)))
    return np.array(densities)


def update_k_grid(atoms, calculator, kptdensity, even=True):
    """Update the k_grid in calculator with the respective density

    Parameters
    ----------
    atoms: ase.atoms.Atoms
        structure that the calculator is attached to
    calculator: ase.calculators.calulator.Calculator
        The calculator
    kptdensity: list of floats
        desired k-point density in all directions
    even: bool
        If True k-grid must be even

    Returns
    -------
    calculator: ase.calculators.calulator.Calculator
        The calculator with updated kgrid
    """

    k_grid = d2k(atoms, kptdensity, even)

    if calculator.name == "aims":
        talk(f"Update aims k_grid with kpt density of {kptdensity} to {k_grid}")
        calculator.parameters["k_grid"] = k_grid
    return calculator


def update_k_grid_calc_dict(calculator_dict, recipcell, kptdensity, even=True):
    """Update k_grid in dictionary representation of a calculator w/ respective density

    Parameters
    ----------
    calculator_dict: dict
        Dictionary representation of calc
    recipcell: np.ndarray
        The reciprocal lattice
    kptdensity: list of floats
        Desired k-point density in all directions
    even: bool
        If True k-grid must be even

    Returns
    -------
    calculator_dict: dict
        The dictionary representation of the calculator with an updated kgrid
    """
    k_grid = d2k_recipcell(recipcell, [True, True, True], kptdensity, even)

    calculator_dict["calculator_parameters"]["k_grid"] = k_grid
    return calculator_dict
