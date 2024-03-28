# flake8: noqa
from ase import units

from .einheiten import *

v_unit = units.Ang / (1000.0 * units.fs)

# Symmetry
symprec = 1e-5

# io
n_db_digits = 14
n_geom_digits = 12
n_yaml_digits = 14

# maths
perfect_fill = 0.523598775598299
vol_sphere = 4.0 * pi / 3.0
