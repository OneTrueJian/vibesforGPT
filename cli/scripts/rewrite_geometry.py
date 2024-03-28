"""clean and rewrite a geometry"""

from argparse import ArgumentParser as argpars

from vibes.helpers import talk
from vibes.helpers.structure import clean_atoms
from vibes.io import read, write
from vibes.structure.io import inform


def rewrite_geometry(file, align, frac, format):
    """clean and rewrite geometry in FILENAME"""
    atoms = read(file, format=format)
    inform(atoms, verbosity=0)

    atoms = clean_atoms(atoms, align=align)

    outfile = f"{file}.cleaned"
    write(atoms, outfile, format=format, scaled=frac)

    talk(f"\nCleaned geometry written to {outfile}")


def main():
    """ print geometry information """
    parser = argpars(description="Read geometry and print some symmetry info")
    parser.add_argument("geom", type=str, help="geometry input file")
    parser.add_argument("--align", action="store_true", help="align the lattice")
    parser.add_argument("--format", default="aims")
    parser.add_argument("--frac", action="store_true")
    args = parser.parse_args()

    rewrite_geometry(args.geom, args.align, args.frac, args.format)


if __name__ == "__main__":
    main()
