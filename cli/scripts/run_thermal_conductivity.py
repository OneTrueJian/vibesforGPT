from phono3py import load

from vibes.phono3py import _defaults as defaults


def run_thermal_conductivity_in_folder(
    folder,
    phono3py_yaml_file=defaults.phono3py_params_yaml_file,
    mesh=defaults.kwargs.q_mesh,
):
    """run thermal conductivity for Phono3py object reconstructed within a folder.

    The folder must contain:
        * phono3py_params.yaml
        * disp_fc3.yaml
        * fc2.hdf5
        * fc3.hdf5

    Args:
        folder: folder that contains the respective files
        phono3py_yaml_file: name of the Phono3py YAML file
        mesh: the q-mesh to use

    """
    phonon = load(phono3py_yaml_file, mesh=mesh, log_level=2)
    phonon.run_thermal_conductivity(write_kappa=True)
