""" vibes quality of life """
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.geometry import get_distances
from ase.io import read
from phonopy.file_IO import parse_FORCE_CONSTANTS, read_force_constants_hdf5
from phonopy.structure.atoms import PhonopyAtoms

from vibes.helpers import Timer, progressbar, talk, warn
from vibes.helpers.converters import input2dict
from vibes.helpers.fileformats import last_from_yaml
from vibes.phonopy._defaults import displacement_id_str
from vibes.structure.convert import to_Atoms

_prefix = "phonopy.utils"


def last_calculation_id(trajectory_file):
    """Return the id of the last computed supercell

    Parameters
    ----------
    trajectory_file: str or Path
        Path to the trajectory file with calculated supercells

    Returns
    -------
    disp_id: int
        The id of the last computed supercell
    """
    disp_id = -1

    try:
        dct = last_from_yaml(trajectory_file, allow_empty=True)
        if dct is not None:
            disp_id = dct["info"][displacement_id_str]
    except (FileNotFoundError):
        pass

    return disp_id


def to_phonopy_atoms(atoms):
    """Convert ase.atoms.Atoms to PhonopyAtoms

    Parameters
    ----------
    atoms: ase.atoms.Atoms
        Atoms to convert

    Returns
    -------
    phonopy_atoms: PhonopyAtoms
        The PhonopyAtoms for the same structure as atoms
    """
    phonopy_atoms = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.get_cell(),
        masses=atoms.get_masses(),
        positions=atoms.get_positions(wrap=True),
    )
    return phonopy_atoms


def enumerate_displacements(cells, info_str=displacement_id_str):
    """Assign a displacemt id to every atoms obect in cells.

    Parameters
    ----------
    cells: list
        Atoms objects created by, e.g., phonopy
    info_str: str
        How to name the cell

    Returns
    -------
    list
        cells with id attached to atoms.info (inplace)
    """
    for nn, scell in enumerate(cells):
        if scell is None:
            continue
        scell.info[info_str] = nn


def get_supercells_with_displacements(phonon):
    """Create a phonopy object and supercells etc.

    Parameters
    ----------
    phonon: phonopy.Phonopy
        The phonopy object with displacement_dataset

    Returns
    -------
    phonon: phonopy.Phonopy
        The phonopy object with displacement_dataset, and displaced supercells
    supercell: ase.atoms.Atoms
        The undisplaced supercell
    supercells_with_disps: list of ase.atoms.Atoms
        All of the supercells with displacements
    """

    supercell = to_Atoms(
        phonon.get_supercell(),
        info={
            "supercell": True,
            "supercell_matrix": phonon.get_supercell_matrix().T.flatten().tolist(),
        },
    )

    scells = phonon.get_supercells_with_displacements()

    supercells_with_disps = [to_Atoms(cell) for cell in scells]

    enumerate_displacements(supercells_with_disps)

    return phonon, supercell, supercells_with_disps


def metadata2dict(phonon, calculator):
    """Convert metadata information to plain dict

    Parameters
    ----------
    phonon: phonopy.Phonopy
        Phonopy Object to get metadata for
    calculator: ase.calculators.calulator.Calculator
        The calculator for the force calculation

    Returns
    -------
    dict
        Metadata as a plain dict with the following items

        atoms: dict
            Dictionary representation of the supercell
        calculator: dict
            Dictionary representation of calculator
        Phonopy/Phono3py: dict
            Phonopy/Phono3py metadata with the following items

            version: str
                Version string for the object
            primitive: dict
                dictionary representation of the primitive cell
            supercell_matrix: list
                supercell matrix as a list
            symprec: float
                Tolerance for determining the symmetry/space group of the primitive cell
            displacement_dataset: dict
                The displacement dataset for the phonon calculation
    """

    atoms = to_Atoms(phonon.get_unitcell())

    prim_data = input2dict(atoms)

    phonon_dict = {
        "version": phonon.get_version(),
        "primitive": prim_data["atoms"],
        "supercell_matrix": phonon.get_supercell_matrix().T.astype(int).tolist(),
        "symprec": float(phonon.get_symmetry().get_symmetry_tolerance()),
        "displacement_dataset": phonon.get_displacement_dataset(),
    }

    try:
        displacements = phonon.get_displacements()
        phonon_dict.update({"displacements": displacements})
    except AttributeError:
        pass

    supercell = to_Atoms(phonon.get_supercell())
    supercell_data = input2dict(supercell, calculator)

    return {str(phonon.__class__.__name__): phonon_dict, **supercell_data}


def get_force_constants_from_trajectory(
    trajectory_file, supercell=None, reduce_fc=False, two_dim=False
):
    """Remaps the phonopy force constants into an fc matrix for a new structure

    Parameters
    ----------
    trajectory_file: vibes.Trajectory
        phonopy trajectory
    supercell: ase.atoms.Atoms
        Atoms Object to map force constants onto
    reduce_fc: bool
        return in [N_prim, N_sc, 3, 3]  shape
    two_dim: bool
        return in [3*N_sc, 3*N_sc] shape

    Returns
    -------
    np.ndarray
         new force constant matrix

    Raises
    ------
    IOError
        If both reduce_fc and two_dim are True
    """
    from vibes.phonopy.postprocess import postprocess

    if reduce_fc and two_dim:
        raise IOError("Only one of reduce_fc and two_dim can be True")

    phonon = postprocess(trajectory_file)

    if supercell is None:
        supercell = to_Atoms(phonon.get_supercell())
        if reduce_fc:
            return phonon.get_force_constants()

    return remap_force_constants(
        phonon.get_force_constants(),
        to_Atoms(phonon.get_unitcell()),
        to_Atoms(phonon.get_supercell()),
        supercell,
        reduce_fc,
        two_dim,
    )


def _map2prim(primitive, supercell, tol=1e-5):
    map2prim = []
    primitive = primitive.copy()
    supercell = supercell.copy()

    # represent new supercell in fractional coords of primitive cell
    supercell_with_prim_cell = supercell.copy()

    supercell_with_prim_cell.cell = primitive.cell.copy()

    primitive.wrap(eps=tol)
    supercell_with_prim_cell.wrap(eps=tol)

    # create list that maps atoms in supercell to atoms in primitive cell
    for a1 in supercell_with_prim_cell:
        diff = primitive.positions - a1.position
        map2prim.append(np.where(np.linalg.norm(diff, axis=1) < tol)[0][0])

    # make sure every atom in primitive was matched equally often
    _, counts = np.unique(map2prim, return_counts=True)
    assert counts.std() == 0, counts

    return map2prim


def remap_force_constants(
    force_constants,
    primitive,
    supercell,
    new_supercell=None,
    reduce_fc=False,
    two_dim=False,
    symmetrize=True,
    tol=1e-5,
    eps=1e-13,
    fortran=True,
):
    """remap force constants [N_prim, N_sc, 3, 3] to [N_sc, N_sc, 3, 3]

    Parameters
    ----------
    force_constants: np.ndarray
        force constants in [N_prim, N_sc, 3, 3] shape
    primitive: ase.atoms.Atoms
        primitive cell for reference
    supercell: ase.atoms.Atoms
        supercell for reference
    new_supercell: ase.atoms.Atoms, optional
        supercell to map to (default)
    reduce_fc: bool
        return in [N_prim, N_sc, 3, 3]  shape
    two_dim: bool
        return in [3*N_sc, 3*N_sc] shape
    symmetrize: bool
        make force constants symmetric
    tol: float
        tolerance to discern pairs
    eps: float
        finite zero

    Returns
    -------
    newforce_constants: np.ndarray
        The remapped force constants

    """
    timer = Timer("remap force constants", prefix=_prefix)

    if new_supercell is None:
        new_supercell = supercell.copy()

    # find positions of primitive cell within the (reference) supercell
    primitive_cell = primitive.cell.copy()
    primitive.cell = supercell.cell

    primitive.wrap(eps=tol)
    supercell.wrap(eps=tol)

    n_sc_new = len(new_supercell)

    # make a list of all pairs for each atom in primitive cell
    sc_r = np.zeros((force_constants.shape[0], force_constants.shape[1], 3))
    for aa, a1 in enumerate(primitive):
        diff = supercell.positions - a1.position
        p2s = np.where(np.linalg.norm(diff, axis=1) < tol)[0][0]
        # sc_r[aa] = supercell.get_distances(p2s, range(n_sc), mic=True, vector=True)
        # replace with post 3.18 ase routine:
        spos = supercell.positions
        sc_r[aa], _ = get_distances([spos[p2s]], spos, cell=supercell.cell, pbc=True)

    # find mapping from supercell to origin atom in primitive cell
    primitive.cell = primitive_cell
    map2prim = _map2prim(primitive, new_supercell)

    if fortran:
        from vibes.helpers.supercell.supercell import supercell as fort

        talk(".. use fortran", prefix=_prefix)
        fc_out = fort.remap_force_constants(
            positions=new_supercell.positions,
            pairs=sc_r,
            fc_in=force_constants,
            map2prim=map2prim,
            inv_lattice=new_supercell.get_reciprocal_cell(),
            tol=tol,
            eps=eps,
        )
    else:

        ref_struct_pos = new_supercell.get_scaled_positions(wrap=True)
        sc_temp = new_supercell.get_cell(complete=True)

        fc_out = np.zeros((n_sc_new, n_sc_new, 3, 3))
        for a1, r0 in enumerate(progressbar(new_supercell.positions)):
            uc_index = map2prim[a1]
            for sc_a2, sc_r2 in enumerate(sc_r[uc_index]):
                r_pair = r0 + sc_r2
                r_pair = np.linalg.solve(sc_temp.T, r_pair.T).T % 1.0
                for a2 in range(n_sc_new):
                    r_diff = np.abs(r_pair - ref_struct_pos[a2])
                    # Integer value is the equivalent of 0.0
                    r_diff -= np.floor(r_diff + eps)
                    if np.linalg.norm(r_diff) < tol:
                        fc_out[a1, a2, :, :] += force_constants[uc_index, sc_a2, :, :]

    timer()

    if two_dim:
        fc_out = fc_out.swapaxes(1, 2).reshape(2 * (3 * fc_out.shape[1],))

        # symmetrize
        violation = np.linalg.norm(fc_out - fc_out.T)
        if violation > 1e-5:
            msg = f"Force constants are not symmetric by {violation:.2e}."
            warn(msg, level=1)
            if symmetrize:
                talk("Symmetrize force constants.")
                fc_out = 0.5 * (fc_out + fc_out.T)

        # sum rule 1
        violation = abs(fc_out.sum(axis=0)).mean()
        if violation > 1e-9:
            msg = f"Sum rule violated by {violation:.2e} (axis 1)."
            warn(msg, level=1)

        # sum rule 2
        violation = abs(fc_out.sum(axis=1)).mean()
        if violation > 1e-9:
            msg = f"Sum rule violated by {violation:.2e} (axis 2)."
            warn(msg, level=1)

        return fc_out

    if reduce_fc:
        p2s_map = np.zeros(len(primitive), dtype=int)

        primitive.cell = new_supercell.cell

        new_supercell.wrap(eps=tol)
        primitive.wrap(eps=tol)

        for aa, a1 in enumerate(primitive):
            diff = new_supercell.positions - a1.position
            p2s_map[aa] = np.where(np.linalg.norm(diff, axis=1) < tol)[0][0]

        primitive.cell = primitive_cell
        primitive.wrap(eps=tol)

        return reduce_force_constants(fc_out, p2s_map)

    return fc_out


def reduce_force_constants(fc_full, map2prim):
    """reduce force constants from [N_sc, N_sc, 3, 3] to [N_prim, N_sc, 3, 3]

    Parameters
    ----------
    fc_full: np.ndarray
        The non-reduced force constant matrix
    map2prim: np.ndarray
        map from supercell to unitcell index

    Returns
    -------
    fc_out: np.ndarray
        The reduced force constants
    """
    uc_index = np.unique(map2prim)
    fc_out = np.zeros((len(uc_index), fc_full.shape[1], 3, 3))
    for ii, uc_ind in enumerate(uc_index):
        fc_out[ii, :, :, :] = fc_full[uc_ind, :, :, :]

    return fc_out


def parse_phonopy_force_constants(
    fc_file="FORCE_CONSTANTS",
    primitive="geometry.primitive",
    supercell="geometry.supercell",
    fortran=True,
    symmetrize=True,
    two_dim=True,
    eps=1e-13,
    tol=1e-5,
    format="aims",
):
    """parse phonopy FORCE_CONSTANTS file and return as 2D array

    Args:
        fc_file (Pathlike): phonopy forceconstant file to parse
        primitive (Atoms or Pathlike): either unitcell as Atoms or where to find it
        supercell (Atoms or Pathlike): either supercell as Atoms or where to find it
        symmetrize (bool): make force constants symmetric
        two_dim (bool): return in [3*N_sc, 3*N_sc] shape
        eps: finite zero
        tol: tolerance to discern pairs and wrap
        format: File format for the input geometries

    Returns:
        Force constants in (3*N_sc, 3*N_sc) shape
    """
    if "hdf5" in str(fc_file):
        fc = read_force_constants_hdf5(fc_file)
    else:
        fc = parse_FORCE_CONSTANTS(fc_file)

    if fc.shape[0] == fc.shape[1]:
        if two_dim:
            fc = fc.swapaxes(1, 2).reshape(3 * fc.shape[1], 3 * fc.shape[1])

        return fc

    if two_dim:
        if isinstance(primitive, Atoms):
            uc = primitive
        elif Path(primitive).exists():
            uc = read(primitive, format=format)
        else:
            raise RuntimeError("primitive cell missing")

        if isinstance(supercell, Atoms):
            sc = supercell
        elif Path(supercell).exists():
            sc = read(supercell, format=format)
        else:
            raise RuntimeError("supercell missing")

        fc = remap_force_constants(
            fc,
            uc,
            sc,
            two_dim=two_dim,
            tol=tol,
            eps=eps,
            fortran=fortran,
            symmetrize=symmetrize,
        )

    return fc
