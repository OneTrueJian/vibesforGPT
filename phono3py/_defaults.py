""" vibes Phono3py defaults """

from vibes.phonopy._defaults import kwargs as defaults_fc2

name = "phono3py"

mandatory = {
    "mandatory_keys": ["machine", "control", "geometry"],
    "mandatory_obj_keys": ["supercell_matrix"],
}

kwargs = defaults_fc2.copy()
kwargs.update(
    {
        "displacement": 0.03,
        "is_diagonal": True,
        "cutoff_pair_distance": 100.0,
        "q_mesh": [21, 21, 21],
        "log_level": 2,
        "workdir": name,
    }
)

fc2_file = "fc2.hdf5"
fc3_file = "fc3.hdf5"
disp_fc3_yaml_file = "disp_fc3.yaml"
phono3py_params_yaml_file = "phono3py_params.yaml"

settings_dict = {name: kwargs}
