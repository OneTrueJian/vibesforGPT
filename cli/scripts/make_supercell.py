"""
Script to compute supercells from inputself.
Similar to generate_structure from TDEP.
"""
from argparse import ArgumentParser as argpars

import numpy as np
from vibes.helpers import Timer
from vibes.helpers import supercell as sc
from vibes.helpers.geometry import get_cubicness, inscribed_sphere_in_box
from vibes.helpers.numerics import get_3x3_matrix
from vibes.io import get_info_str, read
from vibes.spglib.wrapper import get_spacegroup
from vibes.structure.io import inform


def print_matrix(matrix, indent=2):
    ind = indent * " "
    rep = [" [{}, {}, {}]".format(*elem) for elem in matrix]
    # join to have a comma separated list
    rep = f",\n{ind}".join(rep)
    # add leading [ and trailing ]
    rep = f"{ind}[{rep[1:]}"
    rep += "]"
    print(rep)


def make_supercell(
    file,
    dimension,
    n_target,
    deviation,
    dry,
    format,
    scaled=True,
    wrap=False,
    outfile=None,
):
    """create or find a supercell

    Parameters
    ----------
    file: str
        Input primitive (for the super cell) cell geometry file (default: geometry.in)
    dimension: list of ints
        the supercell matrix in any format (defined by 1, 3, or 9 ints)
    n_target: int
        Target number of atoms in the supercell
    deviation: float
        Allowed deviation from n_target (default: 0.2)
    dry: bool
        If True Do not include symmetry information
    format: str
        Format of the input geometry file (default: aims)
    scaled: bool
        If True use fractional coordinates
    wrap: bool
        wrap atoms back to cell
    """
    from vibes.phonopy.wrapper import preprocess

    timer = Timer()
    print(f"Find supercell for")
    cell = read(file, format=format)
    inform(cell, verbosity=0)
    print()

    if n_target:
        print("\nSettings:")
        print(f"  Target number of atoms: {n_target}")
        supercell, smatrix = sc.make_cubic_supercell(
            cell, n_target, deviation=deviation
        )
    elif dimension:
        smatrix = get_3x3_matrix(dimension)
        supercell = sc.make_supercell(cell, smatrix, wrap=wrap)
    else:
        exit("Please specify either a target cell size or a supercell matrix")

    # find number of phonopy displacements
    _, _, scs_ref = preprocess(cell, supercell_matrix=1)
    _, _, scs = preprocess(cell, supercell_matrix=smatrix)

    print(f"\nSupercell matrix:")
    print(" python:  {}".format(np.array2string(smatrix.flatten(), separator=", ")))
    print(" cmdline: {}".format(" ".join([f"{el}" for el in smatrix.flatten()])))
    print(" 2d:")
    print_matrix(smatrix, indent=0)

    print(f"\nSuperlattice:")
    print(supercell.cell.array)
    print(f"\nNumber of atoms:  {len(supercell)}")
    cub = get_cubicness(supercell.cell)
    cutoff = inscribed_sphere_in_box(supercell.cell)
    print(f"  Cubicness:               {cub:.3f} ({cub**3:.3f})")
    print(f"  Largest Cutoff:          {cutoff:.3f} AA")
    print(f"  Number of displacements: {len(scs)} ({len(scs_ref)})")

    if not dry:
        spacegroup = get_spacegroup(cell)
        if not outfile:
            outfile = f"{file}.supercell_{len(supercell)}"
        info_str = get_info_str(supercell, spacegroup)
        info_str += [f"Supercell matrix:    {smatrix.flatten()}"]
        supercell.write(outfile, format=format, scaled=scaled, info_str=info_str)
        print(f"\nSupercell written to {outfile}")

    timer()


def main():
    parser = argpars(description="Read geometry create supercell")
    parser.add_argument("geom", type=str, help="geometry input file")
    parser.add_argument("-n", type=int, help="target size")
    parser.add_argument("-d", "--dim", type=int, nargs="+", help="supercell matrix")
    parser.add_argument("--deviation", type=float, default=0.2)
    parser.add_argument("--dry", action="store_true", help="Do not write output file")
    parser.add_argument("--format", default="aims")
    parser.add_argument("--frac", action="store_true")
    args = parser.parse_args()

    make_supercell(
        args.geom, args.dim, args.n, args.deviation, args.dry, args.format, args.scaled
    )


if __name__ == "__main__":
    main()
