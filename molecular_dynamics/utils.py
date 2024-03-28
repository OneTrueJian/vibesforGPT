""" helper utilities:
    - FCCalculator for using force constants to compute forces
    - Logger for tracking custom MD """
from pathlib import Path

from ase.calculators.calculator import PropertyNotImplementedError

from vibes import son
from vibes.helpers.converters import input2dict


class MDLogger:
    """ MD logger class to write vibes trajectory files """

    def __init__(self, atoms, trajectory_file, metadata=None, overwrite=False):
        """initialize

        Parameters
        ----------
        atoms: ase.atoms.Atoms
            Atoms of the reference structure
        trajectory_file: str or Path
            path to the trajectory file
        metadata: dict
            metadata for the MD run
        overwrite: bool
            If true overwrite the trajectory file
        """

        if not metadata:
            metadata = {}

        self.trajectory_file = trajectory_file
        if Path(trajectory_file).exists() and overwrite:
            Path(trajectory_file).unlink()
            print(f"** {trajectory_file} deleted.")

        son.dump(
            {**metadata, **input2dict(atoms)}, self.trajectory_file, is_metadata=True
        )

    def __call__(self, atoms, info=None):
        """Log the current step to the trajectory

        Parameters
        ----------
        atoms: ase.atoms.Atoms
            Atoms of the current step
        info: dict
            additional information to add to the update
        """
        if info is None:
            info = {}

        try:
            stress = atoms.get_stress(voigt=False)
        except PropertyNotImplementedError:
            stress = None

        dct = {
            "atoms": {
                "info": info,
                "cell": atoms.cell[:],
                "positions": atoms.positions,
                "velocities": atoms.get_velocities(),
            },
            "calculator": {
                "forces": atoms.get_forces(),
                "energy": atoms.get_kinetic_energy(),
                "stress": stress,
            },
        }

        son.dump(dct, self.trajectory_file)
