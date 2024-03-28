"""Phonopy workflow context managing"""

from pathlib import Path

from ase import optimize
from ase.constraints import ExpCellFilter

from vibes import keys, son
from vibes.context import TaskContext
from vibes.helpers import talk, warn
from vibes.helpers.converters import input2dict

from ._defaults import name, settings_dict
from .workflow import _prefix, run_relaxation


# duck type ExpCellFilter, see https://gitlab.com/ase/ase/-/issues/603
# is resolved:


class MyExpCellFilter(ExpCellFilter):
    def get_forces(self, apply_constraint=True):
        """overwrite `apply_constraint` from Expcellfilter"""
        return super().get_forces(apply_constraint=apply_constraint)


class RelaxationContext(TaskContext):
    """context for relaxation"""

    def __init__(self, settings=None, workdir=None, trajectory_file=None):
        """Initialization

        Args:
            settings: settings for the relaxation Workflow
            workdir: Working directory for the relaxation workflow
            trajectory_file: Path to output trajectory
        """
        super().__init__(settings, name, workdir=workdir, template_dict=settings_dict)

        self._opt = None

        # legacy
        if "maxstep" in self.kw:
            self.kw["kwargs"]["maxstep"] = self.kw.pop("maxstep")

    @property
    def exp_cell_filter_kw(self):
        """kwargs from self.settings.relaxation for ExpCellFilter"""
        kw = {
            "hydrostatic_strain": self.kw.get("hydrostatic_strain"),
            "constant_volume": self.kw.get("constant_volume"),
            "scalar_pressure": self.kw.get("scalar_pressure"),
            "mask": self.kw.get("mask"),
        }
        if kw["mask"] is not None:
            msg = "`mask` keyword is set, `fmax` might behave unexpectedly. Check!"
            warn(msg, level=1)

        return kw

    @property
    def driver(self):
        return self.kw.get("driver")

    @property
    def unit_cell(self):
        return self.kw.get("unit_cell")

    @property
    def fmax(self):
        return self.kw.get("fmax")

    @property
    def decimals(self):
        return self.kw.get("decimals")

    @property
    def fix_symmetry(self):
        return self.kw.get("fix_symmetry")

    @property
    def symprec(self):
        return self.kw.get("symprec")

    @property
    def opt(self):
        """the relaxation algorithm"""
        if not self._opt:
            self.mkdir()
            obj = self.settings[name].kwargs

            logfile = str(self.workdir / obj.pop("logfile", "relaxation.log"))

            if "bfgs" in self.driver.lower():
                opt = optimize.BFGS(atoms=self.opt_atoms, logfile=logfile, **obj)
            else:
                warn(f"Relaxation driver {obj.driver} not supported.", level=2)

            talk(f"driver: {self.driver}", prefix=_prefix)
            msg = ["settings:", *[f"  {k}: {v}" for k, v in opt.todict().items()]]
            talk(msg, prefix=_prefix)

            self._opt = opt

        return self._opt

    @property
    def opt_atoms(self):
        """return ExpCellFilter(self.atoms, **kwargs) if `unit_cell == True`"""
        kw = self.exp_cell_filter_kw

        msg = ["filter settings:", *[f"  {k}: {v}" for k, v in kw.items()]]
        talk(msg, prefix=_prefix)

        if self.fix_symmetry:
            try:
                from ase.spacegroup.symmetrize import FixSymmetry
            except ModuleNotFoundError:
                msg = (
                    "`ase.spacegroup.symmetrize.FixSymmetry` is available from ASE 3.20,"
                    " please update."
                )
                raise RuntimeError(msg)

            constr = FixSymmetry(self.atoms, symprec=self.symprec)
            self.atoms.set_constraint(constr)

        if self.unit_cell:
            return MyExpCellFilter(self.atoms, **kw)
        else:
            return self.atoms

    @property
    def metadata(self):
        """return relaxation metadata as dict"""

        opt_dict = self.opt.todict()

        # save time and mass unit
        opt_dict.update(**self.settings[name])

        # save kws
        opt_dict.update({keys.relaxation_options: self.kw})
        opt_dict.update({keys.expcellfilter: self.exp_cell_filter_kw})

        # other stuff
        dct = input2dict(self.atoms, calculator=self.calculator)

        return {"relaxation": opt_dict, **dct}

    def resume(self):
        """resume from trajectory"""
        prepare_from_trajectory(self.atoms, self.trajectory_file)

    def run(self, timeout=None):
        """run the context workflow"""
        self.resume()
        run_relaxation(self)


def prepare_from_trajectory(atoms, trajectory_file):
    """Take the last step from trajectory and initialize atoms + md accordingly"""

    trajectory_file = Path(trajectory_file)
    if trajectory_file.exists():
        last_atoms = son.last_from(trajectory_file)

        assert "info" in last_atoms["atoms"]

        atoms.set_cell(last_atoms["atoms"]["cell"])
        atoms.set_positions(last_atoms["atoms"]["positions"])
        talk(f"Resume relaxation from  {trajectory_file}", prefix=_prefix)
        return True

    talk(f"** {trajectory_file} does not exist, nothing to prepare", prefix=_prefix)
    return False
