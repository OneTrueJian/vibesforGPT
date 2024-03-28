""" run molecular dynamics simulations using the ASE classes """
import numpy as np
from ase.calculators.socketio import SocketIOCalculator

from vibes import keys, son
from vibes.helpers import warn
from vibes.helpers.aims import get_aims_uuid_dict
from vibes.helpers.backup import backup_folder as backup
from vibes.helpers.backup import default_backup_folder
from vibes.helpers.paths import cwd
from vibes.helpers.restarts import restart
from vibes.helpers.socketio import get_socket_info, get_stresses
from vibes.helpers.utils import Timeout
from vibes.helpers.watchdogs import SlurmWatchdog as Watchdog
from vibes.helpers.virials import virials_off, virials_on, supports_virials
from vibes.trajectory import metadata2file, step2file

from ._defaults import calculation_timeout, talk


_calc_dirname = "calculations"
# _socket_timeout = 60


def run_md(ctx, timeout=None):
    """ high level function to run MD """

    converged = run(ctx)

    if not converged:
        talk("restart")
        restart(ctx.settings, trajectory_file=ctx.trajectory_file)
    else:
        talk("done.")


def run(ctx, backup_folder=default_backup_folder):  # noqa: C901
    """run MD for a specific time

    Green-Kubo runs, we treat calculators as follows:

    Ideally, a Calculator should implement a `.virials` property that
    enables/disables the computation of virials. If `calc.virials=True`,
    they should be computed when `.calculate()` is  called. The `virials`
    should be written into `results`. They will be logged. If `virials=True`,
    the `stress` property should also be computed for legacy reasons.

    Calculators that do not implement this are expected to write `stresses`
    into their `results` when `calculate()` is called. Turning `stresses`
    computation on and off every n-th step is not supported for those, so
    make sure that for a GK run, the calculator is set up to compute them.

    FHI-aims with socketio is a special case. Here turning on and off is
    supported through custom means.

    Args:
        ctx (MDContext): context of the MD
        backup_folder (str or Path): Path to the back up folders
    Returns:
        bool: True if hit max steps or completed
    """

    # extract things from context
    atoms = ctx.atoms
    calculator = ctx.calculator
    md = ctx.md
    maxsteps = ctx.maxsteps
    compute_virials = ctx.compute_stresses
    settings = ctx.settings

    # create watchdog with buffer size of 3
    watchdog = Watchdog(buffer=3)

    # create working directories
    workdir = ctx.workdir
    trajectory_file = ctx.trajectory_file
    calc_dir = workdir / _calc_dirname
    backup_folder = (workdir / backup_folder).absolute()

    # prepare the socketio stuff
    socketio_port, socketio_unixsocket = get_socket_info(calculator)

    iocalc = None
    if socketio_port is None:
        atoms.calc = calculator
    else:
        kw = {"port": socketio_port, "unixsocket": socketio_unixsocket}
        iocalc = SocketIOCalculator(calculator, **kw)
        atoms.calc = iocalc

    is_socketio = socketio_port is not None  # do we have to do aims+socketio workarounds?
    use_virials = supports_virials(atoms.calc)  # do we need to use stresses?

    # does it make sense to start everything?
    if md.nsteps >= maxsteps:
        msg = f"run already finished, please inspect {workdir.absolute()}"
        talk(msg)
        return True

    # is the calculation similar enough?
    metadata = ctx.metadata
    if trajectory_file.exists():
        old_metadata, _ = son.open(trajectory_file)
        check_metadata(metadata, old_metadata)

    # backup previously computed data
    backup(calc_dir, target_folder=backup_folder)

    # back up settings
    if settings:
        with cwd(workdir, mkdir=True):
            settings.write()

    # start a timeout
    timeout = Timeout(calculation_timeout)

    with cwd(calc_dir, mkdir=True):
        if is_socketio:
            # workaround to deal with the fact that the socket server only exists
            # *after* calculate is called for the first time, i.e. we can't turn
            # virials on before calculating someting.
            # we have to *only* do it for socketio because other calculators would
            # not re-run their computations when virials are enabled, and so they
            # would be missing later

            if not get_forces(atoms):
                return False

        # log initial step and metadata
        if md.nsteps == 0:

            # log metadata
            metadata2file(metadata, file=trajectory_file)

            if compute_virials:
                virials_on(atoms.calc)

                if use_virials:
                    # trigger calculation, which should compute virials
                    get_forces(atoms)
                else:
                    # if we're using socketio this will compute stresses and retrieve them
                    # if not, it'll just retrieve stresses from calc.results
                    stresses_to_results(atoms)
            else:
                if not is_socketio:
                    # case: non-GK MD, we need to compute properties for 0th step
                    get_forces(atoms)

            # log initial structure computation
            log_step(atoms, md, trajectory_file)

        while not watchdog() and md.nsteps < maxsteps:

            if compute_virials_next(compute_virials, md.nsteps):
                talk("switch virials computation on")
                virials_on(atoms.calc)
            else:
                talk("switch virials computation off")
                virials_off(atoms.calc)

            # reset timeout
            timeout()

            # actually run md step
            if not md_step(md):
                break

            talk(f"Step {md.nsteps} finished, log.")

            if not use_virials:
                if compute_virials_now(compute_virials, md.nsteps):
                    stresses_to_results(atoms)  # properly format stresses
                else:
                    # socketio returns dummy stresses,
                    # which we do not want to log
                    if is_socketio and "stresses" in atoms.calc.results:
                        del atoms.calc.results["stresses"]

            log_step(atoms, md, trajectory_file)

        # close socket
        if iocalc is not None:
            talk("Close the socket")
            iocalc.close()

        talk("Stop.\n")

    # restart
    if md.nsteps < maxsteps:
        return False
    return True


def log_step(atoms, md, trajectory_file):
    atoms.info.update({keys.nsteps: md.nsteps, keys.dt: md.dt})
    meta = get_aims_uuid_dict()  # peek into aims file and grep for uuid
    step2file(atoms, atoms.calc, trajectory_file, metadata=meta)


def compute_virials_now(compute_virials, nsteps):
    """Return if virials should be computed in this step"""
    return compute_virials and (nsteps % compute_virials == 0)


def compute_virials_next(compute_virials, nsteps):
    """Return if virials should be computed in the NEXT step"""
    return compute_virials_now(compute_virials, nsteps + 1)


def stresses_to_results(atoms):
    """Write stresses to results

    This has two functions:

    1. For the aims socketio calculator, it makes sure that stresses actually
    are written into results, which does not happen automatically: They need
    to be explicitly requested via the socket.

    2. Ensure that the stresses are in 3x3 form and extensive.

    The latter is also important for calculators that aren't socketio+aims,
    and the former doesn't hurt, so we always do both.
    """
    atoms.calc.results["stresses"] = get_stresses(atoms)


def check_metadata(new_metadata, old_metadata):
    """Sanity check if metadata sets coincide"""
    om, nm = old_metadata["MD"], new_metadata["MD"]

    # check if keys coincide:
    # sanity check values:
    check_keys = ("md-type", "timestep", "temperature", "friction", "fs")
    keys = [k for k in check_keys if k in om.keys()]
    for key in keys:
        ov, nv = om[key], nm[key]
        if isinstance(ov, float):
            assert np.allclose(ov, nv, rtol=1e-10), f"{key} changed from {ov} to {nv}"
        else:
            assert ov == nv, f"{key} changed from {ov} to {nv}"

    # calculator
    om = old_metadata["calculator"]["calculator_parameters"]
    nm = new_metadata["calculator"]["calculator_parameters"]

    # sanity check values:
    for key in ("xc", "k_grid", "relativistic"):
        if key not in om and key not in nm:
            continue
        ov, nv = om[key], nm[key]
        if isinstance(ov, float):
            assert np.allclose(ov, nv, rtol=1e-10), f"{key} changed from {ov} to {nv}"
        else:
            assert ov == nv, f"{key} changed from {ov} to {nv}"


def get_forces(atoms):
    try:
        _ = atoms.get_forces()
        return True
    except OSError as error:
        warn(f"Error during force computation:")
        print(error, flush=True)
        return False


def md_step(md):
    try:
        md.run(1)
        return True
    except OSError as error:
        warn(f"Error during MD step:")
        print(error, flush=True)
        return False
