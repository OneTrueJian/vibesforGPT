"""a wrapper for TDEP"""
from pathlib import Path
from subprocess import run

import numpy as np
from ase import Atoms
from ase.io import read

from vibes.helpers import Timer
from vibes.helpers.paths import cwd
from vibes.phonopy.postprocess import extract_results
from vibes.phonopy.utils import remap_force_constants
from vibes.trajectory import reader


def convert_phonopy_to_tdep(
    phonon, workdir="output_tdep", logfile="convert_phonopy_to_tdep.log"
):
    """Convert phonopy FORCE_CONSTANTS to tdep outfile.forceconstant

    Parameters
    ----------
        phonon: phonopy.Phonopy
            The Phonopy Object with force constants calculated
        workdir: str or Path
            working directory for the tdep binary
        logfile: str or Path
            log file
    """
    command = ["convert_phonopy_to_forceconstant", "--truncate"]

    print(workdir)

    with cwd(workdir, mkdir=True), open(logfile, "w") as file:
        extract_results(
            phonon,
            tdep=True,
            output_dir=Path().absolute(),
            write_geometries=False,
            plot_bandstructure=False,
        )

        run(command, stdout=file)
        print(f".. outfile.converted_forceconstant created.")

        fc_file = Path("infile.forceconstant")
        if not fc_file.exists():
            fc_file.symlink_to("outfile.converted_forceconstant")
            print(f".. and symlinked to {fc_file}")


def canonical_configuration(
    phonon=None,
    workdir="tdep",
    temperature=300,
    n_sample=5,
    quantum=False,
    logfile="canon_conf.log",
):
    """wrapper for tdep canconical_configuration

    Parameters
    ----------
    phonon: phonopy.Phonopy
        The Phonopy Object with force constants calculated
    workdir: str or Path
        working directory for the tdep binary
    temperature: float
        temperature to run the calculation on
    n_samples: int
        number of samples to generate
    quantum: bool
        If True use Bose-Einstein populations
    logfile: str or Path
        log file

    Returns
    -------
    list of ase.atoms.Atoms
        thermally displaced structures
    """
    if phonon:
        convert_phonopy_to_tdep(phonon, workdir)
    command = ["canonical_configuration"]
    if quantum:
        command.append(f"--quantum")
    command.extend("-of 4".split())
    command.extend(f"-n {n_sample}".split())
    command.extend(f"-t {temperature}".split())
    with cwd(workdir, mkdir=True), open(logfile, "w") as file:
        run(command, stdout=file)
    outfiles = Path(workdir).glob("aims_conf*")
    return [read(of, format="aims") for of in outfiles]


def extract_forceconstants_from_trajectory(
    trajectory_file, workdir="tdep", rc2=10, logfile="fc.log", skip=0, **kwargs
):
    """wrapper for tdep `extract_forceconstants`

    Parameters
    ----------
    trajectory_file: str or Path
        location of trajectory
    workdir: str or Path
        working directory for tdep binary
    rc2: float
        cutoff for force constants in Angstrom
    logfile: str or Path
        logfile
    skip: int
        number of steps to skip (from beginning of trajectory)
    kwargs: dict
        extra keywords
    """

    trajectory = reader(trajectory_file)

    trajectory.to_tdep(folder=workdir, skip=skip)
    extract_forceconstants(workdir, rc2, logfile, **kwargs)


def parse_tdep_remapped_forceconstant(file="infile.forceconstant", force_remap=False):
    """parse the remapped forceconstants from TDEP

    Parameters
    ----------
    file: str or Path
        tdep force constants file
    force_remap: bool
        If True the force constants are remapped

    Returns
    -------
    force_constants: np.ndarray
        The force constant matrix
    lattice_points: np.ndarray
        List of lattice points inside the supercell the force constants were calculated in
    """
    timer = Timer()

    remapped = force_remap
    if "remap" in str(file):
        remapped = True

    print(f"Parse force constants from\n  {file}")
    print(f".. remap representation for supercell: ", remapped)

    with open(file) as fo:
        n_atoms = int(next(fo).split()[0])
        cutoff = float(next(fo).split()[0])

        print(f".. Number of atoms:   {n_atoms}")
        print(rf".. Real space cutoff: {cutoff:.3f} \AA")

        # Not yet clear how many lattice points / force constants we will get
        lattice_points = []
        force_constants = []

        for i1 in range(n_atoms):
            n_neighbors = int(next(fo).split()[0])
            for _ in range(n_neighbors):
                fc = np.zeros([n_atoms, 3, n_atoms, 3])

                # neighbour index
                i2 = int(next(fo).split()[0]) - 1

                # lattice vector
                lp = np.array(next(fo).split(), dtype=float)

                # the force constant matrix for pair (i1, i2)
                phi = np.array([next(fo).split() for _ in range(3)], dtype=float)

                fc[i1, :, i2, :] = phi

                lattice_points.append(lp)
                # force_constants.append(fc.reshape((3 * n_atoms, 3 * n_atoms)))
                force_constants.append(fc)

        n_unique = len(np.unique(lattice_points, axis=0))
        print(f".. Number of lattice points: {len(lattice_points)} ({n_unique} unique)")

    timer()

    if remapped:
        force_constants = np.sum(force_constants, axis=0)
        return force_constants.reshape(2 * (3 * n_atoms,))

    return np.array(force_constants), np.array(lattice_points)


def parse_tdep_forceconstant(
    fc_file="infile.forceconstants",
    primitive="infile.ucposcar",
    supercell="infile.ssposcar",
    fortran=True,
    two_dim=True,
    symmetrize=False,
    reduce_fc=True,
    eps=1e-13,
    tol=1e-5,
    format="vasp",
):
    """Parse the the TDEP force constants into the phonopy format

    Args:
        fc_file (Pathlike): phonopy forceconstant file to parse
        primitive (Atoms or Pathlike): either unitcell as Atoms or where to find it
        supercell (Atoms or Pathlike): either supercell as Atoms or where to find it
        two_dim (bool): return in [3*N_sc, 3*N_sc] shape
        reduce_fc (bool): return in [N_uc, N_sc, 3, 3] shape similar to phonopy
        eps: finite zero
        tol: tolerance to discern pairs and wrap
        format: File format for the input geometries

    Returns:
        Force constants in (3*N_sc, 3*N_sc) shape
    """
    if isinstance(primitive, Atoms):
        uc = primitive
    elif Path(primitive).exists():
        uc = read(primitive, format=format)
    else:
        raise RuntimeError("primitive cell missing")

    if isinstance(supercell, Atoms):
        sc = supercell
    elif Path(supercell).exists():
        sc = read(supercell, format=format)
    else:
        raise RuntimeError("supercell missing")

    uc.wrap(eps=tol)
    sc.wrap(eps=tol)
    n_uc = len(uc)
    n_sc = len(sc)

    force_constants = np.zeros((len(uc), len(sc), 3, 3))

    with open(fc_file) as fo:
        n_atoms = int(next(fo).split()[0])
        cutoff = float(next(fo).split()[0])

        # make sure n_atoms coincides with number of atoms in supercell
        assert n_atoms == n_uc, f"n_atoms == {n_atoms}, should be {n_uc}"

        print(f".. Number of atoms:   {n_atoms}")
        print(rf".. Real space cutoff: {cutoff:.3f} \AA")

        for i1 in range(n_atoms):
            n_neighbors = int(next(fo).split()[0])
            for _ in range(n_neighbors):
                # neighbour index
                i2 = int(next(fo).split()[0]) - 1
                # lattice vector
                lp = np.array(next(fo).split(), dtype=float)
                phi = np.array([next(fo).split() for _ in range(3)], dtype=float)
                r_target = uc.positions[i2] + np.dot(lp, uc.cell[:])
                for ii, r1 in enumerate(sc.positions):
                    r_diff = np.abs(r_target - r1)
                    sc_temp = sc.get_cell(complete=True)
                    r_diff = np.linalg.solve(sc_temp.T, r_diff.T).T % 1.0
                    r_diff -= np.floor(r_diff + eps)
                    if np.sum(r_diff) < tol:
                        force_constants[i1, ii, :, :] += phi

    if not reduce_fc or two_dim:
        force_constants = remap_force_constants(
            force_constants, uc, sc, symmetrize=symmetrize
        )

    if two_dim:
        return force_constants.swapaxes(2, 1).reshape(2 * (3 * n_sc,))

    return force_constants


def extract_forceconstants(
    workdir="tdep", rc2=10, logfile="fc.log", create_symlink=False, **kwargs
):
    """run tdep's extract_forceconstants in the working directory

    Parameters
    ----------
    workdir: Path or str
        The working directory with all input files
    rc2: float
        raidus cutoff to calculate the force constants at
    logfile: str or Path
        log file to put tdep std out
    create_symlink: bool
        make a symbolic link between outfile.forceconstant and infile.forceconstant
    """

    timer = Timer()

    print(f"Extract force constants with TDEP from input files in\n  {workdir}")

    command = ["extract_forceconstants", "--verbose"]

    command.extend(f"-rc2 {rc2}".split())

    with cwd(workdir), open(logfile, "w") as file:
        run(command, stdout=file)
        timer()

        # create the symlink of force constants
        if create_symlink:
            print(f".. Create symlink to forceconstant file")
            outfile = Path("outfile.forceconstant")
            infile = Path("infile" + outfile.suffix)

            if infile.exists():
                proceed = input(f"Symlink {infile} exists. Proceed? (y/n) ")
                if proceed.lower() == "y":
                    infile.unlink()
                else:
                    print(".. Symlink NOT created.")
                    return

            infile.symlink_to(outfile)
            print(f".. Symlink {infile} created.")


def phonon_dispersion_relations(workdir="tdep", gnuplot=True, logfile="dispersion.log"):
    """run tdep's phonon_dispersion_relations in working directory

    Parameters
    ----------
    workdir: Path or str
        The working directory with all input files
    gnuplot: bool
        If True use gnuplot to plot the phonon dispersion into a pdf file
    logfile: str or Path
        log file to put tdep std out

    Raises
    ------
    FileNotFoundError
        If infile is missing
    """

    timer = Timer(f"Run TDEP phonon_dispersion_relations in {workdir}")

    with cwd(workdir):
        # check if input files are present
        for file in ("forceconstant", "ucposcar", "ssposcar"):
            path = Path("infile." + file)
            if not path.exists():
                raise FileNotFoundError(f"{path} missing in ./{workdir}.")

        # plot if ipnut files are present
        command = "phonon_dispersion_relations -p".split()

        with open(logfile, "w") as file:
            run(command, stdout=file)

            if gnuplot:
                print(f".. use gnuplot to plot dispersion to pdf")
                command = "gnuplot -p outfile.dispersion_relations.gnuplot_pdf".split()
                run(command, stdout=file)
    timer()
