""" a light wrapper for spglib """

import numpy as np
import spglib as spg
from ase.atoms import Atoms

from vibes.helpers.dict import AttributeDict
from vibes.konstanten import symprec as default_symprec
from vibes.structure.convert import to_spglib_cell


def cell_to_Atoms(lattice, scaled_positions, numbers, info=None):
    """convert from spglib cell to Atoms

    Parameters
    ----------
    lattice: np.ndarray
        The lattice vectors of the structure
    scaled_positions: np.ndarray
        The scaled positions of the atoms
    numbers: list
        Atomic numbers of all the atoms in the cell
    info: dict
        additional information on the structure

    Returns
    -------
    ase.atoms.Atoms:
        The ASE Atoms object representation of the material
    """
    atoms_dict = {
        "cell": lattice,
        "scaled_positions": scaled_positions,
        "numbers": numbers,
        "pbc": True,
        "info": info,
    }

    return Atoms(**atoms_dict)


def get_symmetry_dataset(atoms, symprec=default_symprec):
    """return the spglib symmetry dataset

    Parameters
    ----------
    atoms: ase.atoms.Atoms
        The structure to get the dataset of
    symprec: float
        The tolerance for determining symmetry and the space group

    Returns
    -------
    AttributeDict
        The symmetry_dataset for the structure
    """

    dataset = spg.get_symmetry_dataset(to_spglib_cell(atoms), symprec=symprec)

    uwcks, count = np.unique(dataset["wyckoffs"], return_counts=True)
    dataset["wyckoffs_unique"] = [(w, c) for (w, c) in zip(uwcks, count)]

    ats, count = np.unique(dataset["equivalent_atoms"], return_counts=True)
    dataset["equivalent_atoms_unique"] = [(a, c) for (a, c) in zip(ats, count)]

    return AttributeDict(dataset)


def map_unique_to_atoms(atoms, symprec=default_symprec):
    """map each symmetry unique atom to other atoms as used by phonopy PDOS

    Parameters
    ----------
    atoms: ase.atoms.Atoms
        The structure to get the dataset of
    symprec: float
        The tolerance for determining symmetry and the space group

    Returns
    -------
    mapping: np.ndarray
        The mapping of symmetry unique atoms to other atoms
    """

    ds = get_symmetry_dataset(atoms, symprec=symprec)

    uniques = np.unique(ds.equivalent_atoms)

    mapping = [[] for _ in range(len(uniques))]

    for ii, index in enumerate(ds.equivalent_atoms):
        for jj, unique in enumerate(uniques):
            if index == unique:
                mapping[jj].append(ii)

    return mapping


def get_spacegroup(atoms, symprec=default_symprec):
    """return spglib spacegroup

    Parameters
    ----------
    atoms: ase.atoms.Atoms
        The structure to get the dataset of
    symprec: float
        The tolerance for determining symmetry and the space group

    Returns
    -------
    str:
        The spglib space group
    """

    return spg.get_spacegroup(to_spglib_cell(atoms), symprec=symprec)


def refine_cell(atoms, symprec=default_symprec):
    """refine the structure

    Parameters
    ----------
    atoms: ase.atoms.Atoms
        The structure to get the dataset of
    symprec: float
        The tolerance for determining symmetry and the space group

    Returns
    -------
    ase.atoms.Atoms:
        The refined structure of atoms
    """
    lattice, scaled_positions, numbers = spg.refine_cell(to_spglib_cell(atoms), symprec)

    return cell_to_Atoms(lattice, scaled_positions, numbers)


def standardize_cell(
    atoms, to_primitive=False, no_idealize=False, symprec=default_symprec
):
    """wrap spglib.standardize_cell

    Parameters
    ----------
    atoms: ase.atoms.Atoms
        The structure to get the dataset of
    to_primitive: bool
        If True go to the primitive cell
    no_idealize: bool
        If True do not idealize the cell
    symprec: float
        The tolerance for determining symmetry and the space group

    Returns
    -------
    ase.atoms.Atoms
        The standardized structure of atoms
    """

    cell = to_spglib_cell(atoms)
    args = spg.standardize_cell(cell, to_primitive, no_idealize, symprec)

    return cell_to_Atoms(*args)
