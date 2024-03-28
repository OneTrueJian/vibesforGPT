"""
 Functions to run several related calculations with using trajectories as cache
"""
import sys
from pathlib import Path

from ase import Atoms
from ase.calculators.socketio import Calculator, SocketIOCalculator

from vibes.filenames import filenames
from vibes.helpers import talk, warn
from vibes.helpers.aims import get_aims_uuid_dict
from vibes.helpers.backup import backup_folder as backup
from vibes.helpers.hash import hash_atoms
from vibes.helpers.lists import expand_list
from vibes.helpers.paths import cwd
from vibes.helpers.socketio import get_socket_info
from vibes.helpers.utils import Spinner
from vibes.helpers.watchdogs import SlurmWatchdog as Watchdog
from vibes.son import son
from vibes.trajectory import get_hashes_from_trajectory_file, metadata2file, step2file


calc_dirname = "calculations"


def calculate(atoms: Atoms, calculator: Calculator, workdir: str = ".") -> Atoms:
    """Perform a dft calculation with ASE

    Args:
        atoms: The structure to calculate
        calculator: The calculator to used to get the properties
        workdir: Path to the working directory

    Returns:
        atoms with all properties calculated

    """
    if atoms is None:
        warn("No atoms given, do nothing.")
        return atoms

    calc_atoms = atoms.copy()
    calc_atoms.calc = calculator

    with cwd(workdir, mkdir=True):
        calc_atoms.calc.calculate(calc_atoms)

    return calc_atoms


def calculate_socket(
    atoms_to_calculate: list,
    calculator: Calculator,
    metadata: dict = None,
    settings: dict = None,
    trajectory_file: str = filenames.trajectory,
    workdir: str = "calculations",
    save_input: bool = False,
    backup_folder: str = "backups",
    backup_after_calculation: bool = True,
    check_settings_before_resume: bool = True,
    dry: bool = False,
    **kwargs,
) -> bool:
    """perform calculations for a set of atoms objects, while able to use the socket

    Args:
        atoms_to_calculate: list with atoms to calculate
        calculator: calculator to use
        metadata: metadata information to store to trajectory
        settings: the settings used to set up the calculation
        trajectory_file: path to write trajectory to
        workdir: working directory
        backup_folder: directory to back up calculations to
        check_settings_before_resume: only resume when settings didn't change
        dry: only create working directory and write metadata to trajectory

    Returns:
        Wether all structures were computed or not

    """
    # create watchdog
    watchdog = Watchdog()

    # create working directories
    workdir = Path(workdir).absolute()
    trajectory_file = workdir / trajectory_file
    backup_folder = workdir / backup_folder
    calc_dir = workdir / calc_dirname

    # save fist atoms object for computation
    atoms = atoms_to_calculate[0].copy()

    # handle the socketio (initially introduced in MD workflow)
    socketio_port, socketio_unixsocket = get_socket_info(calculator)

    iocalc = None
    if socketio_port is None:
        atoms.calc = calculator
    else:
        kw = {"port": socketio_port, "unixsocket": socketio_unixsocket}
        iocalc = SocketIOCalculator(calculator, **kw)
        atoms.calc = iocalc

    # fix for EMT calculator
    fix_emt(atoms)

    # perform backup if calculation folder exists
    backup(calc_dir, target_folder=backup_folder)

    # append settings to metadata
    if settings:
        metadata["settings"] = settings.to_dict()
        if save_input:
            with cwd(workdir, mkdir=True):
                settings.write()

    # fetch list of hashes from trajectory
    precomputed_hashes = get_hashes_from_trajectory_file(trajectory_file)

    # perform calculation
    n_cell = -1

    with cwd(calc_dir, mkdir=True):
        # log metadata and sanity check
        if check_settings_before_resume:
            try:
                old_metadata, _ = son.open(trajectory_file)
                check_metadata(metadata, old_metadata)
                talk(f"resume from {trajectory_file}")
            except FileNotFoundError:
                metadata2file(metadata, trajectory_file)

        if dry:
            talk("dry run requested, stop.")
            sys.exit()

        # launch socket
        for n_cell, cell in enumerate(atoms_to_calculate):
            # skip if cell is None or already computed
            if cell is None:
                talk("`atoms is None`, skip.")
                continue

            if check_precomputed_hashes(cell, precomputed_hashes, n_cell):
                continue

            # make sure a new calculation is started
            atoms.calc.results = {}

            # update calculation_atoms and compute force
            atoms.info = cell.info
            atoms.cell = cell.cell  # update cell
            atoms.positions = cell.positions  # update positions

            # when socketio is used: calculate here, backup was already performed
            # else: calculate in subfolder and backup if necessary
            if iocalc is None and len(atoms_to_calculate) > 1:
                wd = f"{n_cell:05d}"
                # perform backup if calculation folder exists
                backup(wd, target_folder=f"{backup_folder}")
            else:
                wd = "."

            # compute and save the aims UUID
            msg = f"{'[vibes]':15}Compute structure "
            msg += f"{n_cell + 1} of {len(atoms_to_calculate)}"

            with cwd(wd, mkdir=True):
                with Spinner(msg):
                    atoms.calc.calculate(atoms, system_changes=["positions"])

                # log the step including aims_uuid if possible
                meta = get_aims_uuid_dict()
                step2file(atoms, atoms.calc, trajectory_file, metadata=meta)

            if watchdog():
                break

        # close socket
        if iocalc is not None:
            talk("Close the socket")
            iocalc.close()

    # backup
    if backup_after_calculation:
        backup(calc_dir, target_folder=f"{backup_folder}")

    if n_cell < len(atoms_to_calculate) - 1:
        return False

    return True


def check_metadata(new_metadata: dict, old_metadata: dict, keys: list = ["calculator"]):
    """check if metadata sets coincide and sanity check geometry

    Args:
        new_metadata: The metadata of the current calculation
        old_metadata: The metadata from the trajectory.son file
        keys: Keys to check if the metadata agree with

    """
    nm = new_metadata
    om = old_metadata

    if "atoms" in new_metadata:
        new_atoms = new_metadata["atoms"]
        old_atoms = old_metadata["atoms"]
        s1 = expand_list(new_atoms["symbols"])
        s2 = expand_list(old_atoms["symbols"])
        assert s1 == s2, ("symbols changed:", s1, s2)

    for key in keys:
        if key == "walltime":
            continue
        if isinstance(nm[key], dict):
            check_metadata(nm[key], om[key], keys=nm[key].keys())
        if key not in om:
            warn(f"{key} not in previous metadata. Check?", level=1)
        elif nm[key] != om[key]:
            msg = f"{key} changed: from {nm[key]} to {om[key]}"
            warn(msg, level=1)


def fix_emt(atoms: Atoms):
    """fix when using EMT calculator with socket"""
    try:
        atoms.calc.initialize(atoms)
        talk("calculator initialized.")
    except AttributeError:
        pass


def check_precomputed_hashes(
    atoms: Atoms, precomputed_hashes: list, index: int
) -> bool:
    """check if atoms was computed before"""
    hash = hash_atoms(atoms)
    try:
        pre_hash = precomputed_hashes[index]
    except IndexError:
        return False

    if hash == pre_hash:
        talk(f"Structure {index + 1} already computed, skip.")
        return True
