from pathlib import Path

from ase.calculators.socketio import SocketIOCalculator

from vibes.filenames import filenames
from vibes.helpers import talk
from vibes.helpers.paths import cwd
from vibes.helpers.restarts import restart
from vibes.helpers.socketio import get_socket_info
from vibes.helpers.structure import clean_atoms
from vibes.helpers.watchdogs import SlurmWatchdog as Watchdog
from vibes.spglib.wrapper import get_spacegroup
from vibes.trajectory import metadata2file, step2file

from ._defaults import name

_prefix = name
_calc_dirname = "calculation"
_temp_geometry_file = filenames.atoms_next


def run_relaxation(ctx):
    """ high level function to run relaxation"""

    converged = run(ctx)

    if not converged:
        talk("restart", prefix=_prefix)
        restart(ctx.settings, trajectory_file=ctx.trajectory_file)
    else:
        talk("done.", prefix=_prefix)


def run(ctx, backup_folder="backups"):
    """ run a relaxation with ASE"""

    watchdog = Watchdog()

    # extract things from context
    atoms = ctx.atoms
    calculator = ctx.calculator
    opt = ctx.opt
    fmax = ctx.fmax

    workdir = ctx.workdir
    trajectory_file = ctx.trajectory_file
    calc_dir = workdir / _calc_dirname

    socketio_port, socketio_unixsocket = get_socket_info(calculator)
    if socketio_port is None:
        socket_calc = None
    else:
        socket_calc = calculator

    atoms.calc = calculator

    opt_atoms = ctx.opt_atoms

    # is a filter used?
    filter = len(atoms) < len(opt_atoms)

    with SocketIOCalculator(
        socket_calc, port=socketio_port, unixsocket=socketio_unixsocket
    ) as iocalc, cwd(calc_dir, mkdir=True):
        if socketio_port is not None:
            atoms.calc = iocalc

        if Path(opt.restart).exists():
            talk(f"Restart optimizer from {opt.restart}.", prefix=_prefix)
            opt.read()
        else:
            opt.initialize()
            # log very initial step and metadata
            metadata2file(ctx.metadata, trajectory_file)

        talk(f"Start step {opt.nsteps}", prefix=_prefix)
        for ii, converged in enumerate(opt.irun(fmax=fmax)):
            if converged:
                talk("Relaxation converged.", prefix=_prefix)
                break

            # residual forces (and stress)
            forces = opt_atoms.get_forces(apply_constraint=True)
            na = len(atoms)
            res_forces = (forces[:na] ** 2).sum(axis=1).max() ** 0.5 * 1000
            if filter:
                res_stress = (forces[na:] ** 2).sum(axis=1).max() ** 0.5 * 1000

            # log if it's not the first step from a resumed relaxation
            if not (ii == 0 and opt.nsteps > 0):
                talk(f"Step {opt.nsteps} finished.", prefix=_prefix)
                talk(f".. residual force:  {res_forces:.3f} meV/AA", prefix=_prefix)
                if filter:
                    msg = f".. residual stress: {res_stress:.3f} meV/AA**3"
                    talk(msg, prefix=_prefix)

                # spacegroup
                sg = get_spacegroup(atoms)
                talk(f".. Space group:     {sg}")

                talk("clean atoms before logging", prefix=_prefix)
                log_atoms = clean_atoms(atoms, decimals=ctx.decimals)
                log_atoms.info.update({"nsteps": opt.nsteps})

                talk(f".. log", prefix=_prefix)
                step2file(log_atoms, atoms.calc, trajectory_file)

                info_str = [
                    f"Relaxed with BFGS, fmax={fmax*1000:.3f} meV/AA",
                    f"nsteps = {opt.nsteps}",
                    f"residual force  = {res_forces:.6f} meV/AA",
                ]
                if filter:
                    info_str.append(f"residual stress = {res_stress:.6f} meV/AA")

                log_atoms.write(
                    workdir / _temp_geometry_file,
                    format="aims",
                    scaled=True,
                    wrap=False,
                    info_str=info_str,
                )

            if watchdog():
                break

    return converged
