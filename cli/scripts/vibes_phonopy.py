""" Summarize output from ASE.md class (in md.log) """

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from vibes import defaults
from vibes.filenames import filenames
from vibes.helpers.geometry import get_cubicness, inscribed_sphere_in_box
from vibes.helpers.pickle import pread
from vibes.phonopy.context import PhonopyContext
from vibes.phonopy.postprocess import extract_results, postprocess
from vibes.phonopy.wrapper import summarize_bandstructure


def preprocess(
    file: str = None,
    settings_file: str = None,
    dimension: np.ndarray = None,
    format: str = defaults.format.geometry,
    write_supercell: bool = False,
):
    """inform about a phonopy calculation a priori

    Args:
        file: geometry input file
        settings_file: settings file (phonopy.in)
        dimensions: supercell dimension
        format: geometry file format (aims)
        write_supercell: write the supercell to geometry.in.supercell

    """
    from ase.io import read
    from vibes.settings import Settings
    from vibes.phonopy.wrapper import preprocess

    ctx = PhonopyContext(Settings(settings_file))
    settings = ctx.settings

    if file:
        atoms = read(file, format=format)
    else:
        atoms = settings.read_atoms(format=format)

    _, _, scs_ref = preprocess(atoms, **{**settings.phonopy, "supercell_matrix": 1})

    if dimension is not None:
        phonon, sc, scs = preprocess(atoms, supercell_matrix=dimension)
    else:
        phonon, sc, scs = preprocess(atoms, **settings.phonopy)
        print("vibes phonopy workflow settings (w/o configuration):")
        settings.print(only_settings=True)

    smatrix = phonon.get_supercell_matrix().T
    sc_str = np.array2string(smatrix.flatten(), separator=", ")
    bash_str = " ".join(str(l) for l in smatrix.flatten())
    print("Phonopy Information")
    print(f"  Supercell matrix:        {sc_str}")
    print(f"  .. for make_supercell:   -d {bash_str}")
    print(f"  Superlattice:")
    for latvec in sc.cell:
        lv_str = "{:-6.2f} {:-6.2f} {:-6.2f}".format(*latvec)
        print(f"                         {lv_str}")

    print(f"  Number of atoms in SC:   {len(sc)}")
    print(f"  Number of displacements: {len(scs)} ({len(scs_ref)})")

    cub = get_cubicness(sc.cell)
    print(f"  Cubicness:               {cub:.3f} ({cub**3:.3f})")
    sh = inscribed_sphere_in_box(sc.cell)
    print(f"  Largest Cutoff:          {sh:.3f} AA")

    if write_supercell:
        sc.write(filenames.supercell, format=format)


def main():
    """ main routine """
    parser = ArgumentParser(description="information about phonopy task")
    parser.add_argument("infile", help="primitive structure or pickled phonopy")
    parser.add_argument("--dim", type=int, nargs="*", default=None)
    parser.add_argument("--config_file", default="settings.in")
    parser.add_argument("--format", default="aims")
    parser.add_argument("--fp_file", default=None, help="File to store the fingerprint")
    parser.add_argument("--dos", action="store_true")
    parser.add_argument("--pdos", action="store_true")
    parser.add_argument("--tdep", action="store_true")
    parser.add_argument("--born", default=None, help="BORN file")
    parser.add_argument("--full_fc", action="store_true")
    args = parser.parse_args()

    extract_settings = {"plot_dos": args.dos, "plot_pdos": args.pdos, "tdep": args.tdep}

    suffix = Path(args.infile).suffix
    if suffix == ".in":
        preprocess(args.infile, args.config_file, args.dim, args.format)
        return

    if suffix == ".yaml":
        phonon = postprocess(
            args.infile,
            born_charges_file=args.born,
            calculate_full_force_constants=args.full_fc,
        )
    elif suffix in (".pick", ".gz"):
        phonon = pread(args.infile)
    else:
        print("*** Nothing happened.")

    extract_results(phonon, **extract_settings)
    summarize_bandstructure(phonon, fp_file=args.fp_file)


if __name__ == "__main__":
    main()
