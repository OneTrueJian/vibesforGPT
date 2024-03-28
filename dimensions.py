"""Naming the dimensions

    Atom labels: I, J
    Cartesian coordinates: a, b
"""
I, J, a, b = "I", "J", "a", "b"

# composite
Ia, Jb = "Ia", "Jb"

vec = (a,)
tensor = (a, b)

time = "time"
time_atom = (time, I)
time_vec = (time, *vec)
time_atom_vec = (time, I, a)
time_tensor = (time, *tensor)
time_atom_tensor = (time, I, *tensor)
# time_atom_atom_tensor = (time, I, J, a, b)
# taat = time_atom_atom_tensor

lattice = tensor
positions = (I, a)

fc_remapped = (Ia, Jb)

dim_replace = {"a1": a, "a2": b, "I1": I, "I2": J}
