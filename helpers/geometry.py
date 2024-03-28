import numpy as np

from vibes.helpers.numerics import clean_matrix


def get_deformation(cell1: np.array, cell2: np.array) -> np.array:
    """Compute the matrix that deforms cell1 into cell2

    Rationale:
              A2 = D A1
        <=> A2.T = A1.T D.T
         =>    D = np.linalg.solve(A1.T, A2.T).T

    """
    A1 = np.asarray(cell1)
    A2 = np.asarray(cell2)

    return np.linalg.solve(A1.T, A2.T).T


def inscribed_sphere_in_box(cell):
    """Find the radius of an inscribed sphere in a unit cell

    Args:
      cell(np.ndarray): Cell where the sphere should be inscribed

    Returns:
      float: The radius of the inscribed sphere

    """

    # the normals of the faces of the box
    na = np.cross(cell[1, :], cell[2, :])
    nb = np.cross(cell[2, :], cell[0, :])
    nc = np.cross(cell[0, :], cell[1, :])
    na /= np.linalg.norm(na)
    nb /= np.linalg.norm(nb)
    nc /= np.linalg.norm(nc)
    # distances between opposing planes
    rr = 1.0e10
    rr = min(rr, abs(na @ cell[0, :]))
    rr = min(rr, abs(nb @ cell[1, :]))
    rr = min(rr, abs(nc @ cell[2, :]))
    rr *= 0.5
    return rr


def bounding_sphere_of_box(cell):
    """Find the radius of the sphere bounding a box

    Args:
      cell(np.ndarray): Cell where the sphere should be wrapped around

    Returns:
      float: The radius of the inscribed sphere

    """
    a, b, c = np.asarray(cell)

    d1 = a + b + c
    d2 = a + b - c
    d3 = a - b + c
    d4 = a - b - b

    return max(np.linalg.norm([d1, d2, d3, d4], axis=0)) / 2


def get_cubicness(cell):
    """Quantify the cubicness of a cell

    Quantify 'how cubic' a given lattice or cell is by comparing the largest
    sphere that fits into the cell to a sphere that fits into a cubic cell
    of similar size

    Args:
      cell(np.ndarray): Lattice of the cell

    Returns:
        float: cubicness

    """

    # perfect radius: 1/2 * width of the cube
    radius_perfect = np.linalg.det(cell) ** (1 / 3) * 0.5
    radius_actual = inscribed_sphere_in_box(cell)

    # volume = vol_sphere * inscribed_sphere_in_box(cell)**3 / np.linalg.det(cell)
    # Fill = namedtuple('Fill', ['volume', 'radius'])
    # fill = Fill(volume=volume, radius=radius)

    return radius_actual / radius_perfect


def get_rotation_matrix(phi, axis, radians=False):
    """Get the rotation matrix for a given rotation

    Args:
      phi(float): The angle to rotate by
      axis(int): 0-2 axis to rotate ove
      radians(bool, optional): If True phi is in radians (Default value = False)

    Returns:
        np.ndarray: rotation matrix

    """
    if not radians:
        phi = phi / 180 * np.pi

    # Norm the rotation axis:
    axis = axis / np.linalg.norm(axis)

    cp = np.cos(phi)
    sp = np.sin(phi)
    r1, r2, r3 = axis
    Rm = np.array(
        [
            [
                r1 ** 2 * (1 - cp) + cp,
                r1 * r2 * (1 - cp) - r3 * sp,
                r1 * r3 * (1 - cp) + r2 * sp,
            ],
            [
                r1 * r2 * (1 - cp) + r3 * sp,
                r2 ** 2 * (1 - cp) + cp,
                r2 * r3 * (1 - cp) - r1 * sp,
            ],
            [
                r3 * r1 * (1 - cp) - r2 * sp,
                r2 * r3 * (1 - cp) + r1 * sp,
                r3 ** 2 * (1 - cp) + cp,
            ],
        ]
    )

    # clean small values
    Rm = clean_matrix(Rm)

    return Rm
