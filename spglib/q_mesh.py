""" use spglib to symmetrize q points """

import numpy as np
import spglib as spg

from vibes.helpers import Timer
from vibes.helpers.numerics import clean_matrix
from vibes.structure.convert import to_spglib_cell


def get_ir_reciprocal_mesh(
    q_mesh, primitive, supercell=None, is_time_reversal=True, symprec=1e-5
):
    r""" reduce the given q_mesh by symmetry

    Remarks
    Maybe it would be nice to return the respective rotations as well?

    Parameters
    ----------
    q_mesh:
        list of q points in reduced coordinates
    primitive:
        reference structure to determine symmetry from
    supercell:
        reference supercell in case we need to wrap back into BZ
    is_time_reversal: bool
        If True time reversal symmetry is preserved
    symprec: float
        Tolerance for determining the symmetry/space group for the primitive cell

    Returns
    -------
    my_ir_mapping: np.ndarray
        which q points maps to which irreducible
    my_ir_grid: list
        the reduced set of q-points

    Example
    -------
    https://gitlab.com/flokno/vibes/blob/devel/examples/harmonic_analysis/
    irreducible_q_points/ir_qpoints.ipynb

    """

    timer = Timer()

    my_mesh = q_mesh.copy()

    n_lp = len(my_mesh)

    # find the highest q-point in each reciprocal direction
    n_qmesh = my_mesh.max(axis=0) - my_mesh.min(axis=0) + 1

    if supercell is not None:
        cell = to_spglib_cell(supercell)
    else:
        cell = to_spglib_cell(primitive)
    sym_settings = {"is_time_reversal": is_time_reversal, "symprec": symprec}

    # use spglib to create a big enough mesh
    mapping, spg_mesh = spg.get_ir_reciprocal_mesh(2 * n_qmesh, cell, **sym_settings)

    # list the unique grid points
    ir_grid = spg_mesh[np.unique(mapping)]

    # dict translates between full and reduced grid point indices
    dct = {}
    for nn, ii in enumerate(np.unique(mapping)):
        dct[ii] = nn

    # mapping to the unique grid points
    ir_mapping = np.array([dct[ii] for ii in mapping])

    # match the q points from my list with the ones from spg
    match_list = -np.ones(n_lp, dtype=int)
    my_mapping = -np.ones(n_lp, dtype=int)
    for i1, q1 in enumerate(my_mesh):
        for i2, q2 in enumerate(spg_mesh):
            # if np.linalg.norm(q1 % n_qmesh - q2 % n_qmesh) < 1e-12:
            if np.linalg.norm(q1 - q2) < 1e-12:
                # exchange the q point in spg list with my one
                # ir_grid[ir_mapping[i2]] = q1
                match_list[i1] = i2
                my_mapping[i1] = ir_mapping[i2]

    # sanity checks
    assert -1 not in match_list, match_list
    assert -1 not in my_mapping, my_mapping
    assert len(np.unique(match_list)) == n_lp, (len(np.unique(match_list)), match_list)

    # `my_mapping` now maps my grid points to the unique grid points from spg
    # now I want to list only the grid points relevant for me
    # -> create the list of irreducible q points for my list of q points:

    # create dict that translates between my grid points and my unique grid points
    dct = {}
    for nn, ii in enumerate(np.unique(my_mapping)):
        dct[ii] = nn

    my_ir_mapping = np.array([dct[ii] for ii in my_mapping])

    if supercell is None:
        my_ir_grid = ir_grid[np.unique(my_mapping)]
    else:
        # take care of q points that now lie outside of BZ
        # the ir grid in fractional coords w.r.t. to primitive unitcell
        ir_grid_frac_prim = (
            ir_grid[np.unique(my_mapping)]
            @ supercell.get_reciprocal_cell()
            @ primitive.cell.T
        )

        my_ir_grid_temp = clean_matrix(ir_grid_frac_prim)

        # wrap atoms back
        my_ir_grid_temp_cleaned = clean_matrix((my_ir_grid_temp + 0.001) % 1 - 0.001)

        # kick out irreducible q points that can be mapped back into BZ
        my_ir_grid = []
        # ir_temp_mapping = []
        for ii, ir_q_temp in enumerate(my_ir_grid_temp_cleaned):
            for jj, ir_q in enumerate(my_ir_grid):
                if np.linalg.norm(ir_q_temp - ir_q) < 1e-12:
                    # they are the same:
                    my_ir_mapping[my_ir_mapping == ii] = jj
                    break
            else:
                # otherwise add to list
                my_ir_grid.append(ir_q_temp)

        # back to frac positions w.r.t. supercell
        my_ir_grid = np.array(my_ir_grid)
        my_ir_grid = clean_matrix(
            my_ir_grid @ primitive.get_reciprocal_cell() @ supercell.cell.T
        )

    # verify that my_ir_grid indeed containts the reduced q points
    # for i, q in enumerate(my_mapping):
    #     assert (my_ir_grid[my_ir_mapping[i]] % n_qmesh == ir_grid[q] % n_qmesh).all()

    assert len(my_ir_grid) == len(np.unique(my_ir_grid, axis=0)), my_ir_grid

    timer(
        f"number of q points reduced from {len(my_ir_mapping)} to "
        f"{len(np.unique(my_ir_mapping))}"
    )

    return my_ir_mapping, my_ir_grid
