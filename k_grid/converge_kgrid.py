"""Function that performs a k-grid optimization for a structure"""
from pathlib import Path

from ase.calculators.socketio import SocketIOCalculator

from vibes.helpers import talk
from vibes.helpers.converters import input2dict
from vibes.helpers.k_grid import k2d
from vibes.helpers.paths import cwd
from vibes.helpers.socketio import get_socket_info
from vibes.helpers.watchdogs import WallTimeWatchdog as Watchdog
from vibes.k_grid.kpointoptimizer import KPointOptimizer
from vibes.trajectory import metadata2file, step2file


def converge_kgrid(
    atoms,
    calculator,
    func=lambda x: x.calc.get_property("energy", x) / len(x),
    loss_func=lambda x: x,
    dfunc_min=1e-12,
    even=True,
    maxsteps=100,
    trajectory_file="kpt_trajectory.son",
    logfile="kpoint_conv.log",
    walltime=None,
    workdir=".",
    **kwargs,
):
    """Converges the k-grid relative to some loss function

    Parameters
    ----------
    atoms: ase.atoms.Atoms
        geometry of the system you are converging the k-grid on
    calculator: ase.calculators.calulator.Calculator
        calculator for the k-grid convergence
    func: function
        Function used to get the property used test if k-grid is converged
    loss_func: function
        Function used calculate if convergence is reached
    dfunc_min: float
        Convergence criteria for the loss function
    even: bool
        If True kgrid must be even valued
    unit_cell: bool
        if True system is periodic
    maxsteps: int
        maximum steps to run the optimization over
    trajecotry_file: str
        file name to store the trajectory
    logfile: str
        file name for the log file
    walltime: int
        length of the wall time for the job in seconds
    workdir: str
        working directory for the calculation
    kpts_density_init: float
        initial k-point density

    Returns
    -------
    bool
        True if the convergence criteria is met
    """
    watchdog = Watchdog(walltime=walltime)

    workdir = Path(workdir).absolute()
    trajectory_file = workdir / trajectory_file

    kpt_settings = {
        "func": func,
        "loss_func": loss_func,
        "dfunc_min": dfunc_min,
        "even": even,
        "logfile": str(workdir / logfile),
    }
    if "k_grid" in calculator.parameters:
        kpt_settings["kpts_density_init"] = k2d(atoms, calculator.parameters["k_grid"])

    # handle the socketio
    socketio_port, socketio_unixsocket = get_socket_info(calculator)
    socketio = socketio_port is not None

    if socketio is None:
        socket_calc = None
    else:
        socket_calc = calculator

    atoms.calc = calculator
    opt_atoms = atoms

    with SocketIOCalculator(socket_calc, port=socketio_port) as iocalc, cwd(
        workdir / "calculation", mkdir=True
    ):
        if socketio:
            atoms.calc = iocalc

        opt = KPointOptimizer(opt_atoms, **kpt_settings)
        # log very initial step and metadata
        if opt.nsteps == 0 and not trajectory_file.exists():
            metadata = input2dict(atoms, calculator)
            metadata["geometry_optimization"] = opt.todict()
            metadata2file(metadata, trajectory_file)

        converged = False
        for _converged in opt.irun(steps=maxsteps):
            step2file(atoms, atoms.calc, trajectory_file)
            converged = _converged
            if watchdog():
                break
            if converged:
                talk("k-grid optimized.", prefix="k_grid")
                break

    return converged, opt.kpts_density, calculator
