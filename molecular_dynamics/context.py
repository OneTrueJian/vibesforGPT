"""Phonopy workflow context managing"""

from pathlib import Path

import numpy as np
from ase import md as ase_md
from ase import units as u
from ase.io import read
from ase.md.md import MolecularDynamics

from vibes import keys, son
from vibes.context import TaskContext
from vibes.helpers import warn
from vibes.helpers.converters import dict2atoms, input2dict

from ._defaults import keys as md_keys
from ._defaults import name, npt_dict, nve_dict, nvt_dict, talk
from .workflow import run_md


class MDContext(TaskContext):
    """context for molecular dynamics calculation"""

    def __init__(
        self, settings=None, workdir=None, trajectory_file=None, ensemble=keys.nve
    ):
        """Context for MD

        Args:
            settings: settings for the MD Workflow
            workdir: Working directory for the MD workflow
            trajectory_file: Path to output trajectory
            ensemble: the thermodynamics ensemble (NVE, NVT, NPT)
        """
        if ensemble.lower() == keys.nve.lower():
            settings_dict = nve_dict
        elif ensemble.lower() == keys.nvt.lower():
            settings_dict = nvt_dict
        elif ensemble.lower() == keys.npt.lower():
            settings_dict = npt_dict
        else:
            raise ValueError(f"{ensemble} not implemented, choose (NVE/NVT/NPT")

        super().__init__(settings, name, workdir=workdir, template_dict=settings_dict)

        self._md = None
        self._primitive = None
        self._supercell = None

        # legacy Langevin
        for key in ("temperature", "friction"):
            if key in self.kw:
                self.kw["kwargs"][key] = self.kw.pop(key)
        # legacy Berendsen
        for key in ("taut", "taup", "pressure", "compressibility"):
            if key in self.kw:
                self.kw["kwargs"][key] = self.kw.pop(key)

    @property
    def maxsteps(self):
        """return the maxsteps from settings"""
        return self.kw["maxsteps"]

    @maxsteps.setter
    def maxsteps(self, steps):
        """set maxsteps"""
        self.kw["maxsteps"] = steps

    @property
    def md(self):
        """set up the MD algorithm and make sure the units are consistent etc"""
        if not self._md:
            self.mkdir()
            obj = self.kw
            md_settings = {
                "atoms": self.atoms,
                "timestep": obj.timestep * u.fs,
                "logfile": Path(self.workdir) / obj["kwargs"].pop("logfile", "md.log"),
            }
            if "verlet" in obj.driver.lower():
                md = ase_md.VelocityVerlet(**md_settings)
            elif "langevin" in obj.driver.lower():
                md = ase_md.Langevin(
                    temperature=obj.kwargs["temperature"] * u.kB,
                    friction=obj.kwargs["friction"],
                    **md_settings,
                )
            elif "berendsen" in obj.driver.lower():
                kw = obj.kwargs.copy()

                if kw.pop(md_keys.inhomogeneous, False):
                    from ase.md.nptberendsen import (
                        Inhomogeneous_NPTBerendsen as Berendsen,
                    )
                else:
                    from ase.md.nptberendsen import NPTBerendsen as Berendsen

                md = Berendsen(
                    **md_settings,
                    taut=kw.pop(md_keys.taut) * u.fs,
                    taup=kw.pop(md_keys.taup) * u.fs,
                    **kw,
                )

            else:
                warn(f"MD driver {obj.driver} not supported.", level=2)

            talk(f"driver: {obj.driver}")
            talk(["settings:", *[f"  {k}: {v}" for k, v in md.todict().items()]])

            self._md = md

        return self._md

    @md.setter
    def md(self, md):
        """set the md class"""
        assert issubclass(md.__class__, MolecularDynamics)
        self._md = md

    @property
    def primitive(self):
        """The primitive cell structure"""
        if not self._primitive and "files" in self.settings:
            g = self.settings.files
            if "primitive" in g:
                self._primitive = read(g["primitive"], format="aims")
        return self._primitive

    @property
    def supercell(self):
        """The supercell structure"""
        if not self._supercell and "files" in self.settings:
            g = self.settings.files
            if "supercell" in g:
                self._supercell = read(g["supercell"], format="aims")
        return self._supercell

    @property
    def compute_stresses(self):
        """controls `compute_stresses` in `md` section of settings"""

        compute_stresses = 0
        if "compute_stresses" in self.kw:
            # make sure compute_stresses describes a step length
            compute_stresses = self.kw["compute_stresses"]
            if compute_stresses is True:
                compute_stresses = 1
            elif compute_stresses is False:
                compute_stresses = 0
            else:
                compute_stresses = int(compute_stresses)

        return compute_stresses

    @property
    def metadata(self):
        """return MD metadata as dict"""

        md_dict = self.md.todict()

        # save time and mass unit
        md_dict.update({"fs": u.fs, "kB": u.kB, "dt": self.md.dt, "kg": u.kg})

        # other stuff
        dct = input2dict(
            self.atoms,
            calculator=self.calculator,
            primitive=self.primitive,
            supercell=self.supercell,
        )

        return {"MD": md_dict, **dct}

    def prepare_from_trajectory(self):
        """Initialize MD from last step of trajectory"""

        trajectory_file = Path(self.trajectory_file)
        if trajectory_file.exists():
            last_atoms = son.last_from(trajectory_file)

            # we can't set self.atoms = dict2atoms because that would
            # break references to it in self.md and self.calculator
            update_atoms(self.atoms, dict2atoms(last_atoms["atoms"]))

            self.md.nsteps = last_atoms["atoms"]["info"]["nsteps"]

            talk(f"Resumed from step {self.md.nsteps} in {trajectory_file}")
            return True

        talk(f"** {trajectory_file} does not exist, nothing to prepare")
        return False

    def resume(self):
        """resume from trajectory"""
        self.prepare_from_trajectory()

    def run(self, timeout=None):
        """run the context workflow"""
        self.resume()
        run_md(self, timeout=timeout)


def update_atoms(atoms, new_atoms):
    """Update atoms with MD-relevant properties"""
    # sanity check
    assert np.array_equal(atoms.get_atomic_numbers(), new_atoms.get_atomic_numbers())

    atoms.set_positions(new_atoms.get_positions())
    atoms.set_velocities(new_atoms.get_velocities())
    atoms.set_pbc(new_atoms.get_pbc())
    atoms.set_cell(new_atoms.get_cell())
