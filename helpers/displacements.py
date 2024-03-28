""" deal with displacements in supercells """

import numpy as np

from .numerics import clean_matrix


def get_dR(atoms, atoms0, wrap_tol=1e-5):
    """Compute and return dR = R - R^0 respecting possibly wrapped atoms

    Args:
        atoms (ase.atoms.Atoms): The distorted structure
        atoms0 (ase.atoms.Atoms): The reference structure
        wrap_tol (float): The tolerance for wrapping atoms at the cell edges

    Returns:
        np.ndarray: R - R^0
    """

    # get fractional coordinates
    fR0 = atoms0.get_scaled_positions()
    fR = atoms.get_scaled_positions()

    fdR = fR - fR0

    # wrap too large displacements
    fdR = (fdR + 0.5 + wrap_tol) % 1 - 0.5 - wrap_tol

    # displacement in cartesian coordinates
    dR = clean_matrix(fdR @ atoms0.cell, eps=1e-12)

    return dR


def get_U(atoms, atoms0, wrap_tol=1e-5):
    """ Compute dR = R - R^0 and return U = sqrt(M) . dR

    Args:
        atoms (ase.atoms.Atoms): The distorted structure
        atoms0 (ase.atoms.Atoms): The reference structure
        wrap_tol (float): The tolerance for wrapping atoms at the cell edges

    Returns:
        np.ndarray: U - U^0
    """

    dR = get_dR(atoms0, atoms, wrap_tol=wrap_tol)

    masses = atoms.get_masses()

    dU = dR * np.sqrt(masses[:, None])

    return dU


def get_dUdt(atoms, wrap_tol=1e-5):
    """ Compute V and return dU/dt = sqrt(M) . V

    Args:
        atoms (ase.atoms.Atoms): The distorted structure
        wrap_tol (float): The tolerance for wrapping atoms at the cell edges

    Returns:
        np.ndarray: sqrt(M) . V
    """

    V = atoms.get_velocities()  # / v_unit

    if V is None:
        V = np.zeros_like(atoms.positions)

    masses = atoms.get_masses()

    dUdt = V * np.sqrt(masses[:, None])

    return dUdt
