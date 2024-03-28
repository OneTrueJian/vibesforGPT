"""Aims workflow context managing"""

from pathlib import Path

from ase import Atoms
from ase.io import read

from vibes import keys

from . import setup

calc_dirname = "calculations"


class CalculatorContext:
    """context for ase calculator"""

    def __init__(self, settings: dict, atoms: Atoms = None, workdir: str = None):
        """Constructor

        Args:
            settings: Settings Object for the Workflow
            atoms: refernce structure
            workdir: Directory to run the calculation in

        """
        self.settings = settings

        assert keys.calculator in settings

        if workdir:
            self.settings[keys.calculator][keys.workdir] = workdir

        self._ref_atoms = atoms
        self._primitive = None
        self._supercell = None
        self._atoms_to_calculate = None

    @property
    def geometry_files(self) -> list:
        """The geometry input files"""
        # find geometries
        files = []
        s = self.settings
        if "files" in s:
            if "geometry" in s.files:
                path = next(Path().glob(s.files.geometry))
                assert path.exists()
                files.append(path)
                self.settings.files["geometry"] = str(path)

        if "geometries" in s.files:
            paths = sorted(Path().glob(s.files.geometries))
            for path in paths:
                assert path.exists(), path
            files.extend(paths)

        return files

    @property
    def atoms_to_calculate(self) -> list:
        """The atoms that are supposed to be computed"""
        if not self._atoms_to_calculate:
            files = self.geometry_files
            atoms_list = [read(file, format="aims") for file in files]
            self._atoms_to_calculate = atoms_list
        return self._atoms_to_calculate

    @property
    def primitive(self):
        """The primitive cell structure"""
        g = self.settings.files
        if not self._primitive and "primitive" in g:
            self._primitive = read(g["primitive"], format="aims")
        return self._primitive

    @property
    def supercell(self):
        """The supercell structure"""
        g = self.settings.files
        if not self._supercell and "supercell" in g:
            self._supercell = read(g["supercell"], format="aims")
        return self._supercell

    @property
    def basisset_location(self):
        """Location of the basis set files"""
        loc = self.settings.machine.basissetloc
        return Path(loc)

    @property
    def ref_atoms(self):
        """The reference structure for the calculation"""
        if not self._ref_atoms:
            self._ref_atoms = read(self.geometry_files[0], format="aims")
        return self._ref_atoms

    @ref_atoms.setter
    def ref_atoms(self, atoms: Atoms):
        """ref_atoms setter

        Args:
            atoms: atoms to set ref_atoms

        """
        assert isinstance(atoms, Atoms)
        self._ref_atoms = atoms

    def get_calculator(self):
        """Get the ASE Calculator based on the context"""
        if self.name == "aims":
            from .aims import setup_aims

            return setup_aims(self)
        else:
            return setup.from_settings(self.settings)

    @property
    def calculator(self):
        """return ASE calculator based on this context"""
        return self.get_calculator()

    @property
    def name(self):
        """The name of the calculation"""
        return self.settings.calculator.get("name")

    @property
    def workdir(self):
        return Path(self.settings.calculator.get(keys.workdir, self.name))
