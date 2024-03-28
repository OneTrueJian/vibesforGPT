import numpy as np
from ase.atoms import Atoms

from vibes.konstanten import n_db_digits


def to_phonopy_atoms(atoms, wrap=False):
    """Convert ase.atoms.Atoms to PhonopyAtoms

    Args:
      atoms(ase.atoms.Atoms): Atoms to convert
      wrap(bool, optional): If True wrap the scaled positions (Default value = False)

    Returns:
        phonopy.PhonopyAtoms

    """
    from phonopy.structure.atoms import PhonopyAtoms

    phonopy_atoms = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.get_cell(),
        masses=atoms.get_masses(),
        positions=atoms.get_positions(wrap=wrap),
    )

    return phonopy_atoms


def to_spglib_cell(atoms):
    """Convert ase.atoms.Atoms to spglib cell

    Args:
      atoms(ase.atoms.Atoms): Atoms to convert

    Returns:
        tuple

    """
    lattice = atoms.cell
    positions = atoms.get_scaled_positions()
    number = atoms.get_atomic_numbers()
    return (lattice, positions, number)


def to_Atoms(atoms, info=None, pbc=True, db=False):
    """Convert structure to ase.atoms.Atoms

    Args:
      structure(PhonopyAtoms): The structure to convert
      info(dict): Additional information to include in atoms.info (Default value = None)
      pbc(bool): True if the structure is periodic (Default value = True)
      db(bool): remove masses and round positions to 14 digits

    Returns:
        Atoms: structure as atoms object

    """

    if info is None:
        info = {}

    if atoms is None:
        return None

    atoms_dict = {
        "symbols": atoms.get_chemical_symbols(),
        "cell": atoms.get_cell(),
        "masses": atoms.get_masses(),
        "positions": atoms.get_positions(),
        "pbc": pbc,
        "info": info,
    }

    if db:
        del atoms_dict["masses"]
        db_dict = {
            "cell": np.round(atoms.get_cell(), n_db_digits),
            "positions": np.round(atoms.get_positions(), n_db_digits),
        }
        atoms_dict.update(db_dict)

    atoms = Atoms(**atoms_dict)

    return atoms
