"""Phonopy workflow context managing"""

import sys

from ase import Atoms
from vibes.context import TaskContext
from vibes.helpers.numerics import get_3x3_matrix
from vibes.structure.misc import get_sysname

from . import metadata2dict
from . import postprocess as postprocess
from . import wrapper as backend
from ._defaults import keys, name, settings_dict


class PhonopyContext(TaskContext):
    """context for phonopy calculation"""

    def __init__(self, settings=None, workdir=None, name=name, template_dict=None):
        """Intializer

        Args:
            settings: Settings for the Workflow
            workdir: The working directory for the workflow
            name: `phonopy` or `phono3py`
            template_dict: for `phono3py`
        """
        template_dict = template_dict or settings_dict
        super().__init__(settings, name, workdir=workdir, template_dict=template_dict)

        if str(self.workdir.stem).lower() == "auto":
            smatrix = self.supercell_matrix.flatten()
            vol = self.atoms.get_volume()
            sysname = get_sysname(self.atoms)
            rep = "_{}_{}{}{}_{}{}{}_{}{}{}_{:.3f}".format(sysname, *smatrix, vol)
            dirname = self.name + rep
            self.workdir = dirname

        self.backend = backend
        self.postprocess = postprocess

        self._primitive = None

    @property
    def supercell_matrix(self):
        """return settings.phonopy.supercell_matrix"""
        return get_3x3_matrix(self.kw["supercell_matrix"])

    @property
    def q_mesh(self):
        """return the q_mesh from settings"""
        return self.kw[keys.q_mesh]

    @property
    def primitive(self):
        """(primitive) unit cell"""
        if self._primitive is None:
            self._primitive = self.settings.read_atoms()
        return self._primitive

    @primitive.setter
    def primitive(self, atoms: Atoms):
        """set the primitive unit cell"""
        assert isinstance(atoms, Atoms)
        self._primitive = atoms

    def preprocess(self):
        return self.backend.preprocess(atoms=self.primitive, **self.kw)

    def bootstrap(self, dry=False):
        """load settings, prepare atoms, calculator, and phonopy object"""

        # Phonopy preprocess
        phonon, supercell, scs = self.preprocess()

        self.atoms = supercell
        calc = self.calculator

        # save metadata
        metadata = metadata2dict(phonon, calc)

        return {
            "atoms_to_calculate": scs,
            "calculator": calc,
            "metadata": metadata,
            "workdir": self.workdir,
            "settings": self.settings,
            "save_input": True,
            "backup_after_calculation": False,
            "dry": dry,
            **self.kw,
        }

    def run(self, dry=False):
        """run phonopy workflow"""
        from vibes.calculate import calculate_socket
        from vibes.helpers import talk
        from vibes.helpers.restarts import restart

        args = self.bootstrap(dry=dry)

        talk(f"Run phonopy workflow in working directory\n  {self.workdir}")

        try:
            self.postprocess(workdir=self.workdir)
            msg = "** Postprocess could be performed from previous calculations. Check"
            msg += f"\n**  {self.workdir}"
            sys.exit(msg)
        except (FileNotFoundError, RuntimeError):
            completed = calculate_socket(**args)

        if not completed:
            restart(args["settings"])
        else:
            talk("Start postprocess.")
            self.postprocess(**args)
            talk("done.")
