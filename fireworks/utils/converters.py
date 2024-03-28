"""
A to convert phonopy/phono3py objects to dictionaries
"""
import copy

import numpy as np

from vibes.ase.db.dict_converters import atoms2dict
from vibes.materials_fp.material_fingerprint import (
    get_phonon_bs_fp,
    get_phonon_dos_fp,
    to_dict,
)
from vibes.structure.convert import to_Atoms


def standardize_sc_matrix(sc_matrix):
    """convert sc_matrix into a list of ints"""
    return list(np.array(sc_matrix, dtype=int).T.flatten())


def phonon_to_dict(phonon, to_mongo=False, add_fc=False):
    """Converts a phonopy object to a dictionary

    Parameters
    ----------
    phonon: phonopy.Phonopy
        The Phonopy object to be converted
    to_mongo: bool
        If True then it is being sent to a mongo database

    Returns
    -------
    dct: dict
        the dictionary representation of phonon
    """
    dct = atoms2dict(to_Atoms(phonon.get_primitive()))
    dct["key_value_pairs"] = {"symprec": phonon._symprec}
    if phonon.get_supercell_matrix() is not None:
        dct["sc_matrix_2"] = standardize_sc_matrix(phonon.get_supercell_matrix())
        dct["natoms_in_sc_2"] = len(dct["numbers"]) * int(
            round(np.linalg.det(phonon.get_supercell_matrix()))
        )
    if add_fc:
        dct["_fc_2"] = np.array(phonon.get_force_constants())
    else:
        displacement_dataset = copy.deepcopy(phonon._displacement_dataset)
        dct["force_2"] = []
        for disp1 in displacement_dataset["first_atoms"]:
            if "forces" in disp1:
                dct["force_2"].append(disp1.pop("forces"))

        dct["force_2"] = np.array(dct["force_2"])

    dct["displacement_dataset_2"] = displacement_dataset

    if phonon.mesh is not None:
        dct["qmesh"] = phonon.mesh.mesh_numbers
    if phonon._total_dos is not None:
        dct["phonon_dos_fp"] = to_dict(get_phonon_dos_fp(phonon))

    if phonon.band_structure is not None:
        dct["qpoints"] = {}
        for ii, q_point in enumerate(phonon.band_structure.qpoints):
            if list(q_point[0]) not in dct["qpoints"].values():
                dct["qpoints"][phonon.band_structure.distances[ii][0]] = list(
                    q_point[0]
                )
            if list(q_point[-1]) not in dct["qpoints"].values():
                dct["qpoints"][phonon.band_structure.distances[ii][-1]] = list(
                    q_point[-1]
                )
        dct["phonon_bs_fp"] = to_dict(
            get_phonon_bs_fp(phonon, dct["qpoints"]), to_mongo
        )
        if to_mongo:
            q_point = {}
            for pt in dct["qpoints"]:
                q_point[str(pt).replace(".", "_")] = dct["qpoints"][pt]
            dct["qpoints"] = q_point
    if phonon.thermal_properties is not None:
        dct["tp_ZPE"] = phonon.thermal_properties.zero_point_energy
        dct["tp_high_T_S"] = phonon.thermal_properties.high_T_entropy
        (
            dct["tp_T"],
            dct["tp_A"],
            dct["tp_S"],
            dct["tp_Cv"],
        ) = phonon.get_thermal_properties()

    return dct


def phonon3_to_dict(phonon3, store_second_order=False, to_mongo=False):
    """Converts a phonopy object to a dictionary

    Parameters
    ----------
    phonon3: phono3py.phonon3.Phono3py
        The Phono3py object to be converted
    store_second_order: bool
        If True store the second order properties of the phonopy object
    to_mongo: bool
        If True then it is being sent to a mongo database

    Returns
    -------
    dct: dict
        the dictionary representation of phonon3
    """
    dct = atoms2dict(to_Atoms(phonon3.get_primitive()))
    dct["symprec"] = phonon3._symprec

    if phonon3.get_supercell_matrix() is not None:
        dct["sc_matrix_3"] = standardize_sc_matrix(phonon3.get_supercell_matrix())
        dct["natoms_in_sc_3"] = len(dct["numbers"]) * int(
            round(np.linalg.det(phonon3.get_supercell_matrix()))
        )

    if store_second_order and phonon3._phonon_supercell_matrix is not None:
        dct["sc_matrix_2"] = standardize_sc_matrix(phonon3._phonon_supercell_matrix)
        dct["natoms_in_sc_2"] = len(dct["numbers"]) * int(
            round(np.linalg.det(phonon3._phonon_supercell_matrix))
        )

    dct["force_3"] = []
    get_forces = True
    displacement_dataset = copy.deepcopy(phonon3._displacement_dataset)
    for disp1 in displacement_dataset["first_atoms"]:
        if "forces" in disp1:
            if disp1["forces"].shape[0] == dct["natoms_in_sc_3"]:
                dct["force_3"].append(disp1.pop("forces"))
            else:
                get_forces = False
                break

    if get_forces:
        for ii, disp1 in enumerate(displacement_dataset["first_atoms"]):
            for disp2 in disp1["second_atoms"]:
                if "delta_forces" in disp2:
                    dct["force_3"].append(
                        disp2.pop("delta_forces") + dct["force_3"][ii]
                    )
        dct["force_3"] = np.array(dct["force_3"])
    else:
        print("Warning not storing forces because of an inconsistent number of atoms")
        dct["force_3"] = np.ndarray(0)

    dct["force_3"] = np.array(dct["force_3"])
    dct["displacement_dataset_3"] = displacement_dataset

    if phonon3.get_thermal_conductivity():
        dct["tp_T"] = phonon3.get_thermal_conductivity().get_temperatures()
        dct["tp_kappa"] = phonon3.get_thermal_conductivity().get_kappa()[0]
        dct["qmesh"] = phonon3.get_thermal_conductivity().get_mesh_numbers()

    return dct
