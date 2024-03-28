""" helps to find lattice points in supercell, match positions to images in the
unit cell etc. """

import numpy as np
import scipy.linalg as la

from vibes.helpers.lattice import fractional
from vibes.helpers.supercell import get_lattice_points
from vibes.helpers.utils import Timer


def map_L_to_i(indeces):
    """ Map to atoms belonging to specific lattice point

    Parameters
    ----------
    indeces: list
        map from u_I in supercell to u_iL w.r.t to primitive cell and lattice point

    Returns
    -------
    np.ndarray
        list of masks that single out the atoms that belong to specific lattice point
    """

    n_lattice_points = max(i[1] for i in indeces) + 1
    mappings = []
    for LL in range(n_lattice_points):
        mappings.append([idx[1] == LL for idx in indeces])
    return mappings


def _get_lattice_points(atoms, supercell, lattice_points, extended=False):
    """local wrapper for `get_lattice_points`"""
    if lattice_points is None:
        if extended:
            _, lattice_points = get_lattice_points(atoms.cell, supercell.cell)
        else:
            lattice_points, _ = get_lattice_points(atoms.cell, supercell.cell)

    return lattice_points


def map_I_to_iL(
    in_atoms, in_supercell, lattice_points=None, extended=False, tolerance=1e-5
):
    """Map from supercell index I to (i, L), i is the unit cell index and L lattice p.

    Args:
        in_atoms (ase.atoms.Atoms): primitive cell
        in_supercell (ase.atoms.Atoms): supercell
        lattice_points (list, optional): list of pre-computed lattice points L
        extended (bool, optional): return lattice points at supercell surface
        tolerance (float, optional): tolerance for wrapping

    Returns:
        list, list: I_to_iL map, inverse map
    """
    timer = Timer()

    atoms = in_atoms.copy()
    supercell = in_supercell.copy()
    atoms.wrap()
    supercell.wrap()

    lattice_points = _get_lattice_points(
        atoms, supercell, lattice_points, extended=extended
    )

    # create all positions R = r_i + L
    all_positions = []
    tuples = []
    for ii, pos in enumerate(atoms.positions):
        for LL, lp in enumerate(lattice_points):
            all_positions.append(pos + lp)
            tuples.append((ii, LL))

    # prepare the list of indices
    indices = len(supercell) * [(-1, -1)]
    matches = []

    for satom in supercell:
        spos, ssym, jj = satom.position, satom.symbol, satom.index
        for atom in atoms:
            pos, sym, ii = atom.position, atom.symbol, atom.index
            # discard rightaway if not the correct species
            if ssym != sym:
                continue
            for LL, lp in enumerate(lattice_points):
                if la.norm(spos - pos - lp) < tolerance:
                    indices[jj] = (ii, LL)
                    matches.append(jj)
                    break

    # catch possibly unwrapped atoms
    for satom in supercell:
        spos, ssym, jj = satom.position, satom.symbol, satom.index
        if jj in matches:
            continue
        for LL, lp in enumerate(lattice_points):
            for atom in atoms:
                pos, sym, ii = atom.position, atom.symbol, atom.index
                if ssym != sym:
                    continue
                fpos = fractional(spos - pos - lp, supercell.cell)
                tol = tolerance
                if la.norm((fpos + tol) % 1 - tol) < tolerance:
                    indices[jj] = (ii, LL)
                    matches.append(jj)
                    break

    # sanity checks:
    if len(np.unique(matches)) != len(supercell):
        for ii, _ in enumerate(supercell):
            if ii not in matches:
                print(f"Missing: {ii} {supercell.positions[ii]}")

    assert len(np.unique(indices, axis=0)) == len(supercell), (indices, len(supercell))

    # should never arrive here
    assert not any(-1 in l for l in indices), ("Indices found: ", indices)

    timer(f"matched {len(matches)} positions in supercell and primitive cell")

    inv = _map_iL_to_I(indices)

    return np.array(indices), np.array(inv)


def _map_iL_to_I(I_to_iL_map):
    """ map (i, L) back to supercell index I

    Parameters
    ----------
    I_to_iL_map: np.ndarray
        Map from I to iL

    Returns
    -------
    np.ndarray
        Map back from primitive cell index/lattice point to supercell index

    Raises
    ------
    AssertionError
        If iL2I[I2iL[II][0], I2iL[II][1]] does not equal II
    """

    I2iL = np.array(I_to_iL_map)

    n_atoms = int(I2iL[:, 0].max() + 1)
    n_lps = int(I2iL[:, 1].max() + 1)

    iL2I = np.zeros([n_atoms, n_lps], dtype=int)

    for II, (ii, LL) in enumerate(I_to_iL_map):
        iL2I[ii, LL] = II

    # sanity check:
    for II in range(n_atoms * n_lps):
        iL = I2iL[II]
        I = iL2I[iL[0], iL[1]]
        assert II == I, (II, iL, I)

    return iL2I.squeeze()
