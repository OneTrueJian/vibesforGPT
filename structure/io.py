"""read, format, and write structures and inform about them"""

import datetime

import numpy as np

from vibes.helpers.brillouinzone import get_special_points
from vibes.helpers.geometry import (
    get_cubicness,
    inscribed_sphere_in_box,
    bounding_sphere_of_box,
)
from vibes.helpers.numerics import clean_matrix
from vibes.helpers.utils import talk
from vibes.konstanten import n_geom_digits, symprec, v_unit
from vibes.spglib.wrapper import get_symmetry_dataset
from vibes.structure.misc import get_sysname


def _get_decoration_string(atoms, symprec=symprec, verbose=True):
    """Decoration for geometry.in files

    Args:
      atoms: the structure
      symprec: symmetry precision
      verbose: be verbose

    Returns:
        str: decoration

    """
    if verbose:
        sds = get_symmetry_dataset(atoms, symprec=symprec)
        string = "#=====================================================\n"
        string += f"# libflo:  geometry.in \n"
        # string += '#   Material: {:s}\n'.format(cell.get_chemical_formula())
        date = datetime.datetime.now().isoformat(" ", timespec="seconds")
        string += f"#   Date:    {date}\n"
        string += "#=====================================================\n"
        string += f"#   Material:          {atoms.get_chemical_formula()}\n"
        string += f"#   No. atoms:         {atoms.n_atoms}\n"
        string += f"#   Spacegroup:        {sds.number:d}\n"

        if any(atoms.pbc):
            string += f"#   Unit cell volume:  {atoms.get_volume():f} AA^3\n"
        if hasattr(atoms, "tags"):
            for ii, tag in enumerate(atoms.tags):
                string += f"#   Tag {ii+1:2d}:            {tag}\n"
        # Supercell
        if hasattr(atoms, "smatrix"):
            string += f"#   Supercell matrix:  {atoms.smatrix.flatten()}\n"
    else:
        string = ""

    return string


def get_aims_string(atoms, decorated=True, scaled=None, velocities=False, wrap=False):
    """print the string that is geometry.in

    Args:
      atoms(ase.atoms.Atoms): the structure
      decorated(bool, optional): add decoration (Default value = True)
      scaled(bool, optional): use scaled positions (Default value = None)
      velocities(bool, optional): write velocities (Default value = False)
      wrap(bool, optional): write wrapped positions (Default value = False)

    Returns:
      str: geometry.in

    """
    if scaled is None:
        if "supercell" in atoms.tags:
            scaled = False
        else:
            scaled = True

    string = _get_decoration_string

    if any(atoms.pbc):
        latvecs = clean_matrix(atoms.get_cell())
        for latvec, constraint in zip(latvecs, atoms.constraints_lv):

            if decorated:
                string += f"  lattice_vector "
            else:
                string += f"lattice_vector "
            #
            string += f"{latvec[0]: .{n_geom_digits}e} "
            string += f"{latvec[1]: .{n_geom_digits}e} "
            string += f"{latvec[2]: .{n_geom_digits}e}\n"
            if constraint:
                string += "constrain_relaxation .true.\n"

    # if not pbc: direct positions!
    else:
        scaled = False

    # Write (preferably) scaled positions
    symbols = atoms.get_chemical_symbols()
    if scaled:
        positions = clean_matrix(atoms.get_scaled_positions(wrap=wrap))
        atompos = "atom_frac"
    else:
        # if decorated:
        #     string += '\n# Cartesian positions:\n'
        #
        positions = clean_matrix(atoms.get_positions())
        atompos = "atom"

    if velocities:
        vels = atoms.get_velocities()

    for ii, (pos, sym) in enumerate(zip(positions, symbols)):
        if decorated:
            string += f"  {atompos:9s}  "
        else:
            string += f"{atompos:s}  "
        #
        string += f"{pos[0]: .{n_geom_digits}e} "
        string += f"{pos[1]: .{n_geom_digits}e} "
        string += f"{pos[2]: .{n_geom_digits}e}  "
        string += f"{sym:3s}\n"
        if velocities:
            vel = vels[ii]
            if decorated:
                string += "    velocity "
            else:
                string += "velocity "
            string += f"{vel[0]: .{n_geom_digits}e} "
            string += f"{vel[1]: .{n_geom_digits}e} "
            string += f"{vel[2]: .{n_geom_digits}e}\n"

    return string


def inform(atoms, file=None, verbosity=1, symprec=symprec):
    """print geometry information to screen

    Args:
      atoms: the structure
      file:  (Default value = None)
      verbosity:  (Default value = 1)
      symprec:  (Default value = symprec)

    """
    unique_symbols, multiplicity = np.unique(atoms.symbols, return_counts=True)
    # Structure info:
    talk(f"Geometry info")
    print(f"  input geometry:    {get_sysname(atoms)}")
    if file:
        print(f"  from:              {file}")
    print(f"  Symmetry prec.:    {symprec}")
    print(f"  Number of atoms:   {len(atoms)}")

    msg = ", ".join([f"{s} ({m})" for (s, m) in zip(unique_symbols, multiplicity)])
    print(f"  Species:           {msg}")
    print(f"  Periodicity:       {atoms.pbc}")
    if verbosity > 0 and any(atoms.pbc):
        print(f"  Lattice:  ")
        for vec in atoms.cell:
            print(f"    {vec}")
        cub = get_cubicness(atoms.cell)
        print(f"  Cubicness:           {cub:.3f} ({cub**3:.3f})")
        r = inscribed_sphere_in_box(atoms.cell)
        print(f"  Largest Cutoff:      {r:.3f} AA")
        r = bounding_sphere_of_box(atoms.cell)
        print(f"  Bounding box rad.:   {r/2:.3f} AA")

    print("")

    if symprec is not None:
        sds = get_symmetry_dataset(atoms, symprec=symprec)

        print(f"  Spacegroup:          {sds.international} ({sds.number})")
        if sds.number > 1:
            msg = "  Wyckoff positions:   "
            print(msg + ", ".join(f"{c}*{w}" for (w, c) in sds.wyckoffs_unique))
            msg = "  Equivalent atoms:    "
            print(msg + ", ".join(f"{c}*{a}" for (a, c) in sds.equivalent_atoms_unique))

        if verbosity > 1:
            print(f"  Standard lattice:  ")
            for vec in sds.std_lattice:
                print(f"    {vec}")

        if verbosity > 1:
            print(f"  Special k points:")
            for key, val in get_special_points(atoms).items():
                print(f"    {key}: {val}")

    # Info
    for (key, val) in atoms.info.items():
        print(f"  {key:10s}: {val}")

    # lengths and angles
    if verbosity > 0:
        la = atoms.get_cell_lengths_and_angles()
        print("\nCell lengths and angles [\u212B, °]:")
        print("  a, b, c: {}".format(" ".join([f"{l:11.4f}" for l in la[:3]])))
        angles = "  \u03B1, \u03B2, \u03B3: "
        values = "{}".format(" ".join([f"{l:11.4f}" for l in la[3:]]))
        print(angles + values)
        print(f"  Volume:           {atoms.get_volume():11.4f} \u212B**3")
        print(f"  Volume per atom:  {atoms.get_volume() / len(atoms):11.4f} \u212B**3")

        if atoms.get_velocities() is not None:
            v = atoms.get_momenta().sum(axis=0) / v_unit / atoms.get_masses().sum()
            print(f"\n Net velocity: {v} \u212B/ps")
