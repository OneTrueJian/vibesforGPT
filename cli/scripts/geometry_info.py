from argparse import ArgumentParser as argpars

from vibes.io import inform, read
from vibes.spglib.wrapper import get_symmetry_dataset
from vibes.structure.misc import get_sysname


def main():
    """ print geometry information """
    parser = argpars(description="Read geometry and print some symmetry info")
    parser.add_argument("geom", type=str, help="geometry input file")
    parser.add_argument("-t", "--tolerance", type=float, default=1e-5)
    parser.add_argument("--format", default="aims")
    parser.add_argument("-s", "--short", action="store_true", help="Only print name")
    args = parser.parse_args()

    file = args.geom
    cell = read(file, format=args.format)

    if args.short:
        sds = get_symmetry_dataset(cell)
        print(get_sysname(cell, sds))
    else:
        inform(cell, file=file, symprec=args.tolerance, verbosity=0)


if __name__ == "__main__":
    main()
