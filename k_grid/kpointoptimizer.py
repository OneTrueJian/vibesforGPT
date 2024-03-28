""" ASE Optimizer to converge the k-grid """
from ase.optimize.optimize import Dynamics

from vibes.helpers.k_grid import d2k


class KPointOptimizer(Dynamics):
    """ASE Optimizer Class for K-point optimization"""

    def __init__(
        self,
        atoms,
        func=lambda x: x.calc.get_property("energy", x) / len(x),
        loss_func=lambda x: x,
        dfunc_min=1e-6,
        even=True,
        trajectory_file=None,
        logfile="-",
        kpts_density_init=1.0,
    ):
        """Initializes the KPointOptimizer

        Parameters
        ----------
        atoms: ase.atoms.Atoms
            system you want to get optimized k-grids for
        func: function
            Function used to get the property used test if k-grid is converged
        loss_func: function
            Function used calculate if convergence is reached
        dfunc_min: float
            Convergence criteria for the loss function
        even: bool
            If True kgrid must be even valued
        trajecotry_file: str
            file name to store the trajectory
        logfile: str
            file name for the log file
        kpts_density_init: float
            initial k-point density
        """
        Dynamics.__init__(
            self,
            atoms,
            logfile=logfile,
            trajectory=trajectory_file,
            append_trajectory=True,
        )

        self.kpts_density = kpts_density_init
        self.even = even
        kpts_initial = d2k(atoms, self.kpts_density, self.even)
        self.kpts = kpts_initial

        self.step = 0
        self.func = func
        self.loss_func = loss_func
        self.dfunc = 1e12
        self.ref = 0
        self.last = 0
        self.dfunc_min = dfunc_min
        self.n_atoms = len(atoms)

    @property
    def kpts(self):
        """ Accessor function to k_grid"""
        return self.atoms.calc.parameters["k_grid"]

    @kpts.setter
    def kpts(self, kp):
        """ Setter function for the k_grid"""
        self.atoms.calc.parameters["k_grid"] = kp

    def increase_kpts(self):
        """Step function to increase k-grid density"""
        # fkdev: wie vernuenftig loesen?!
        self.atoms.calc.results = {}
        while True:
            self.kpts_density += 1
            kpts = d2k(self.atoms, self.kpts_density, self.even)
            if kpts != self.kpts:
                self.kpts = kpts
                break

    def todict(self):
        """Converts optimizer to a dict"""
        return {"type": "kpoint-optimizer"}

    def log(self):
        """Log the k-point optimization"""
        if self.logfile:
            if self.step == 0:
                self.logfile.write(f"Step, k_points, Energy\n")
            self.logfile.write(
                f"{self.step}, {self.kpts}, {self.atoms.get_potential_energy()}\n"
            )
            self.logfile.flush()

    def irun(self, steps=100):
        """Iterative run functions, advances the calculation by one step

        Parameters
        ----------
        steps: int
            Maximum number of steps

        Yields
        ------
        bool
            True system is converged, False otherwise (If False go to the next step)
        """
        self.ref = self.func(self.atoms)

        for _ in range(steps):
            val = self.func(self.atoms)
            self.dfunc = self.loss_func(self.last - val)
            self.call_observers()
            self.log()
            if self.dfunc < self.dfunc_min:
                yield True
            else:
                yield False
                self.increase_kpts()
                self.last = val

    def run(self, steps=100):
        """Runs the optimizer

        Parameters
        ----------
        steps: int
            Maximum number of steps
        """
        for _ in self.irun(steps=steps):
            pass
