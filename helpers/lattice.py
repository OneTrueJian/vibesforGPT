""" Tools for dealing with lattices """

import scipy.linalg as la


def fractional(positions, lattice):
    """ compute fractioal components in terms of lattice

            r = r_frac . lattice

        =>  r_frac = r . inv_lattice

    Parameters
    ----------
    positions: np.ndarray
        positions of the atoms
    lattice: np.ndarray
        The Lattice Vectors

    Returns
    -------
    np.ndarray
        The fractional positions
    """

    return positions @ la.inv(lattice)
