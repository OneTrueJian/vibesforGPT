import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import full_3x3_to_voigt_6_stress
from ase.geometry import find_mic


class FCCalculator(Calculator):
    """Harmonic Force Constants Calculator

    """

    implemented_properties = ["energy", "energies", "forces", "free_energy"]
    implemented_properties += ["stress", "stresses"]  # bulk properties

    def __init__(self, atoms_reference=None, force_constants=None, fast=True, **kwargs):
        r"""
        Parameters
        ----------
        atoms_reference: Atoms
          reference structure
        force_constants: np.ndarray [3*Na, 3*Na]
          the force contants w.r.t. the reference structure
        fast: bool
          directly compute f_i and sigma_i without detour via f_ij

        When `fast` is chose, compute stresses according to

            s_i = -u_i \otimes f_i

        otherwise according to

            s_i = -1/2 \sum_j (u_i - u_j) \otimes f_ij

        where
            u_i: displacement of atom i
            f_i: force on atom i
            f_ij: force on atom i stemming from atom j

        """
        Calculator.__init__(self, **kwargs)

        self._fast = fast
        self.atoms_reference = atoms_reference
        self.force_constants = force_constants

    def calculate(
        self, atoms=None, properties=None, system_changes=all_changes,
    ):
        if properties is None:
            properties = self.implemented_properties

        n_atoms = len(self.atoms_reference)

        # find displacements
        cell = np.asarray(atoms.cell)
        positions = atoms.positions
        reference_positions = self.atoms_reference.positions
        displacements = find_mic(positions - reference_positions, cell)[0]

        # shorthand
        fc = self.force_constants

        if self._fast:
            stresses = np.zeros((n_atoms, 3, 3))

            forces = -(fc @ displacements.flatten()).reshape(displacements.shape)

            for ii in range(n_atoms):
                stresses[ii] = -np.outer(displacements[ii], forces[ii])

        else:
            forces = np.zeros((n_atoms, n_atoms, 3))
            stresses = np.zeros((n_atoms, n_atoms, 3, 3))
            for (ii, jj) in np.ndindex(n_atoms, n_atoms):
                d_ij = displacements[ii] - displacements[jj]
                p_ij = fc[3 * ii : 3 * (ii + 1), 3 * jj : 3 * (jj + 1)]
                f_ij = p_ij @ d_ij

                forces[ii, jj] = f_ij

                s_ij = -np.outer(d_ij, f_ij) / 2
                stresses[ii, jj] = s_ij

                # sanity check
                if ii == jj:
                    assert np.allclose(s_ij, 0)

            forces = forces.sum(axis=1)
            stresses = stresses.sum(axis=1)

        # assign properties
        # potential energy = - dr * f
        energies = -(displacements * forces).sum(axis=1)
        energy = energies.sum()

        self.results["energy"] = energy
        self.results["energies"] = energies
        self.results["free_energy"] = energy

        self.results["forces"] = forces

        # no lattice, no stress
        if atoms.number_of_lattice_vectors == 3:
            stresses = full_3x3_to_voigt_6_stress(stresses)
            self.results["stress"] = stresses.sum(axis=0) / atoms.get_volume()
            self.results["stresses"] = stresses / atoms.get_volume()
