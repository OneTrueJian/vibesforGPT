""" A leightweight wrapper for Phono3py """

import numpy as np
from phono3py import Phono3py, load
from phono3py.api_phono3py import Phono3pyYaml

from vibes import konstanten as const
from vibes.helpers.numerics import get_3x3_matrix
from vibes.phonopy import get_supercells_with_displacements
from vibes.structure.convert import to_phonopy_atoms

from . import _defaults as defaults


def prepare_phono3py(
    atoms,
    supercell_matrix,
    fc2=None,
    fc3=None,
    cutoff_pair_distance=defaults.kwargs.cutoff_pair_distance,
    displacement_dataset=None,
    is_diagonal=defaults.kwargs.is_diagonal,
    q_mesh=defaults.kwargs.q_mesh,
    displacement=defaults.kwargs.displacement,
    symmetrize_fc3q=False,
    symprec=defaults.kwargs.symprec,
    log_level=defaults.kwargs.log_level,
    **kwargs,
):
    """Prepare a Phono3py object

    Args:
        atoms: ase.atoms.Atoms
        supercell_matrix: np.ndarray
        fc2: np.ndarray
        fc3: np.ndarray
        cutoff_pair_distance: float
        displacement_dataset: dict
        is_diagonal: bool
        mesh: np.ndarray
        displacement: float
        symmetrize_fc3q: bool
        symprec: float
        log_level: int

    Returns:
        phono3py.Phono3py
    """

    ph_atoms = to_phonopy_atoms(atoms, wrap=True)

    supercell_matrix = get_3x3_matrix(supercell_matrix)

    phonon3 = Phono3py(
        ph_atoms,
        supercell_matrix=np.transpose(supercell_matrix),
        mesh=q_mesh,
        symprec=symprec,
        is_symmetry=True,
        symmetrize_fc3q=symmetrize_fc3q,
        frequency_factor_to_THz=const.omega_to_THz,
        log_level=log_level,
    )

    if displacement_dataset is not None:
        phonon3.set_displacement_dataset(displacement_dataset)

    phonon3.generate_displacements(
        distance=displacement,
        cutoff_pair_distance=cutoff_pair_distance,
        is_diagonal=is_diagonal,
    )

    if fc2 is not None:
        phonon3.set_fc2(fc2)
    if fc3 is not None:
        phonon3.set_fc3(fc3)

    return phonon3


def preprocess(
    atoms,
    supercell_matrix,
    cutoff_pair_distance=defaults.kwargs.cutoff_pair_distance,
    is_diagonal=defaults.kwargs.is_diagonal,
    q_mesh=defaults.kwargs.q_mesh,
    displacement=defaults.kwargs.displacement,
    symprec=defaults.kwargs.symprec,
    log_level=defaults.kwargs.log_level,
    **kwargs,
):
    """
    Set up a Phono3py object and generate all the supercells necessary for the 3rd order

    Args:
        atoms: ase.atoms.Atoms
        supercell_matrix: np.ndarray
        cutoff_pair_distance: float
        is_diagonal: bool
        q_mesh: np.ndarray
        displacement: float
        symprec: float
        log_level: int

    Returns:
        phonon3: phono3py.Phono3py
        supercell: ase.atoms.Atoms
        supercells_with_disps: list of ase.atoms.Atoms
    """

    phonon3 = prepare_phono3py(
        atoms,
        supercell_matrix=supercell_matrix,
        cutoff_pair_distance=cutoff_pair_distance,
        is_diagonal=is_diagonal,
        q_mesh=q_mesh,
        displacement=displacement,
        symprec=symprec,
        log_level=log_level,
    )

    return get_supercells_with_displacements(phonon3)


def phono3py_save(phonon: Phono3py, file=defaults.phono3py_params_yaml_file):
    """adapted form Phono3py.save"""
    ph3py_yaml = Phono3pyYaml()
    ph3py_yaml.set_phonon_info(phonon)
    with open(file, "w") as w:
        w.write(str(ph3py_yaml))


def phono3py_load(
    file=defaults.phono3py_params_yaml_file,
    mesh=defaults.kwargs.q_mesh,
    log_level=defaults.kwargs.log_level,
    **kwargs,
):
    """load phono3py object from file

    Args:
      mesh: the q mesh
      log_level: log level
      kwargs: kwargs for `Phono3py.load`

    Returns:
      Phono3py

    """
    phonon = load(file, **kwargs)

    return phonon
