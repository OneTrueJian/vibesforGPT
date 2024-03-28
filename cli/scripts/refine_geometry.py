"""Symmetry refinement"""

from argparse import ArgumentParser as argpars

import numpy as np

from vibes.helpers import bold, talk, warn
from vibes.io import inform, read, write
from vibes.spglib.wrapper import refine_cell, standardize_cell


def refine_geometry(
    file, primitive, conventional, center, origin, cartesian, format, symprec
):
    """refine geometry with spglib and write to file

    Parameters
    ----------
    file: str
        Input geometry file (default: gemoetry.in)
    primitive: bool
        If True output the primitive cell structure to file.primitive
    conventional: bool
        If True ouput the conventional structure to file.conventional
    center: bool
        If True center the cell to the center of mass and output to
        file.(primitive, conventional, or refined).center
    origin: bool
        If True center the cell to the origin and output to
        file.(primitive, conventional, or refined).origin
    cartesian: bool
        If True use Cartesian coordinates
    format: str
        Format of the input geometry file (default: aims)
    symprec: float
        Precision for space group/symmetry operation determination
    """
    atoms = read(file, format=format)

    talk(f"Perfom geometry refinement for")

    inform(atoms, symprec=symprec)

    outfile = f"{file}"

    if primitive:
        atoms = standardize_cell(atoms, to_primitive=True, symprec=symprec)
        outfile += ".primitive"
    elif conventional:
        atoms = standardize_cell(atoms, to_primitive=False, symprec=symprec)
        outfile = f"{file}.conventional"
    else:
        atoms = refine_cell(atoms, symprec=symprec)
        outfile += ".refined"

    if center:
        atoms = center_atoms(atoms)
        outfile += ".center"
    elif origin:
        atoms = center_atoms(atoms, origin=True)
        outfile += ".origin"

    write(atoms, outfile, format=format, spacegroup=True, scaled=not cartesian)

    coords = "fractional coordinates"
    if cartesian:
        coords = "cartesian coordinates"

    print()
    talk(f"New structure written in {bold(coords)} and {format} format to")
    talk(f"  {bold(outfile)}")


def center_com(atoms):
    """ centre the center of mass """
    midpoint = atoms.cell.sum(axis=0) / 2
    atoms.positions -= atoms.get_center_of_mass()
    atoms.positions += midpoint

    return atoms


def center_pos(atoms):
    """ centre the average position"""
    midpoint = atoms.cell.sum(axis=0) / 2
    atoms.positions -= atoms.positions.mean(axis=0)
    atoms.positions += midpoint

    return atoms


def center_atoms(atoms, origin=False):
    """ Center atoms: Move center of mass, then average position to cell midpoint """

    warn("This is an experimental feature!")

    # 0) move 1. atom to origin
    atoms.positions -= atoms[0].position
    atoms.wrap()

    # i) move center of mass to midpoint
    atoms = center_com(atoms)
    for _ in range(10):
        temp_pos = atoms.positions.copy()
        atoms.wrap()
        if np.linalg.norm(temp_pos - atoms.positions) > 1e-12:
            atoms = center_com(atoms)
            continue
        else:
            break
        # if we end up here, this was not succesful
        warn("FIXME", level=2)

    # ii) move average position to midpoint
    atoms = center_pos(atoms)
    for _ in range(10):
        temp_pos = atoms.positions.copy()
        atoms.wrap()
        if np.linalg.norm(temp_pos - atoms.positions) > 1e-12:
            atoms = center_pos(atoms)
            continue
        else:
            break
        # if we end up here, this was not succesful
        warn("FIXME", level=2)

    if origin:
        atoms.positions -= atoms[0].position
        atoms.wrap()

    return atoms


def main():
    """main routine, deprecated since CLI"""
    parser = argpars(description="Read geometry and use spglib to refine")
    parser.add_argument("geometry")
    parser.add_argument("--prim", action="store_true", help="store primitive cell")
    parser.add_argument("--conv", action="store_true", help="store conventional cell")
    parser.add_argument("-t", "--tolerance", type=float, default=1e-5)
    parser.add_argument("--format", default="aims")
    parser.add_argument("--cart", action="store_true")
    parser.add_argument("--center", action="store_true", help="center average position")
    parser.add_argument("--origin", action="store_true", help="first atom to origin")
    args = parser.parse_args()

    refine_geometry(
        args.geometry,
        args.prim,
        args.conv,
        args.center,
        args.origin,
        args.cart,
        args.format,
        args.tolerance,
    )


if __name__ == "__main__":
    main()
