""" Provide a full highlevel phonopy workflow """
from pathlib import Path

from phonopy.file_IO import write_FORCE_CONSTANTS, write_FORCE_SETS
from phonopy.interface.phonopy_yaml import PhonopyYaml

from vibes.filenames import filenames
from vibes.helpers import Timer as _Timer
from vibes.helpers import talk as _talk
from vibes.helpers import warn
from vibes.helpers.brillouinzone import get_special_points
from vibes.helpers.converters import dict2atoms
from vibes.helpers.paths import cwd
from vibes.io import write
from vibes.phonopy import _defaults as defaults
from vibes.phonopy import wrapper
from vibes.structure.convert import to_Atoms
from vibes.trajectory import reader

from . import displacement_id_str


_prefix = "phonopy.postprocess"
_tdep_fnames = {"primitive": "infile.ucposcar", "supercell": "infile.ssposcar"}


def talk(msg):
    return _talk(msg, prefix=_prefix)


def Timer(msg=None):
    return _Timer(msg, prefix=_prefix)


def postprocess(
    trajectory_file=filenames.trajectory,
    workdir=".",
    calculate_full_force_constants=False,
    born_charges_file=None,
    enforce_sum_rules=False,
    verbose=True,
    **kwargs,
):
    """Phonopy postprocess

    Parameters
    ----------
    trajectory_file: str or Path
        The trajectory file to process
    workdir: str or Path
        The working directory where trajectory is stored
    calculate_full_force_constants: bool
        If True calculate the full force constant matrix
    born_charges_file: str or Path
        Path to the born charges file
    verbose: bool
        If True be verbose

    Returns
    -------
    phonon: phonopy.Phonopy
        The Phonopy object with the force constants calculated
    """

    timer = Timer("Start phonopy postprocess:")

    trajectory_file = Path(workdir) / trajectory_file

    calculated_atoms, metadata = reader(trajectory_file, get_metadata=True)

    # make sure the calculated atoms are in order
    for nn, atoms in enumerate(calculated_atoms):
        atoms_id = atoms.info[displacement_id_str]
        if atoms_id == nn:
            continue
        warn(f"Displacement ids are not in order. Inspect {trajectory_file}!", level=2)

    for disp in metadata["Phonopy"]["displacement_dataset"]["first_atoms"]:
        disp["number"] = int(disp["number"])
    primitive = dict2atoms(metadata["Phonopy"]["primitive"])
    supercell = dict2atoms(metadata["atoms"])
    supercell_matrix = metadata["Phonopy"]["supercell_matrix"]
    supercell.info = {"supercell_matrix": str(supercell_matrix)}
    symprec = metadata["Phonopy"]["symprec"]

    phonon = wrapper.prepare_phonopy(primitive, supercell_matrix, symprec=symprec)
    phonon._displacement_dataset = metadata["Phonopy"]["displacement_dataset"].copy()

    force_sets = [atoms.get_forces() for atoms in calculated_atoms]
    phonon.produce_force_constants(
        force_sets, calculate_full_force_constants=calculate_full_force_constants
    )

    if enforce_sum_rules:
        from vibes.hiphive import enforce_rotational_sum_rules

        timer = Timer("Enforce rotational sum rules with hiphive")
        enforce_rotational_sum_rules(phonon, only_project=True)
        timer()

    # born charges?
    if born_charges_file:
        from phonopy.file_IO import get_born_parameters

        prim = phonon.get_primitive()
        psym = phonon.get_primitive_symmetry()
        if verbose:
            talk(f".. read born effective charges from {born_charges_file}")
        nac_params = get_born_parameters(open(born_charges_file), prim, psym)
        phonon.set_nac_params(nac_params)

    # Set qmesh if it is in settings, but default to None
    q_mesh = metadata.get("settings", {}).get("phonopy", {}).get("q_mesh")
    if q_mesh is not None:
        phonon.init_mesh(q_mesh)

    if verbose:
        timer("done")
    return phonon


def check_negative_frequencies(phonon, tol=1e-8):
    """check if there are negative frequencies in the spectrum"""
    primitive = to_Atoms(phonon.get_unitcell())
    special_points = primitive.cell.get_bravais_lattice().get_special_points()
    for k in special_points:
        sp = special_points[k]
        f = phonon.get_frequencies(sp)
        if any(f < tol):
            warn(f"Negative frequencies found at {k} = {sp}:")
            print("# Mode   Frequency")
            for ii, fi in enumerate(f[f < 0]):
                print(f"  {ii+1:3d} {fi:12.5e} THz")


def print_frequencies_at_gamma(phonon):
    """print gamma point frequencies"""
    talk("\nFrequencies at Gamma point:")
    phonon.run_mesh([1, 1, 1])
    qpoints, weights, frequencies, _ = phonon.get_mesh()
    for q, w, f in zip(qpoints, weights, frequencies):
        print(f"q = {q} (weight= {w})")
        print("# Mode   Frequency")
        for ii, fi in enumerate(f):
            print(f"  {ii+1:3d} {fi:12.7f} THz")


def extract_results(
    phonon,
    minimal_output=True,
    thermal_properties=False,
    bandstructure=False,
    dos=False,
    pdos=False,
    debye=False,
    bz_path=None,
    animate=False,
    animate_q=None,
    q_mesh=None,
    remap_fc=False,
    output_dir="phonopy_output",
    verbose=False,
):
    """Extract results from phonopy object and present them.

    Args:
        phonon (phonopy.Phonopy): The Phonopy Object with calculated force constants
        minimal_output (bool, optional): write geometries and force_constants
        thermal_properties (bool, optional): write and plot thermal properties
        bandstructure (bool, optional): write and plot bandstructure
        dos (bool, optional): write and plot DOS
        pdos (bool, optional): write and plot projected DOS
        debye (bool, optional): write Debye Temperature
        bz_path (list, optional): Brillouin zone path for bandstructure
        animate (bool, optional): write animation.ascii files
        animate_q (list, optional): write animation.ascii file for given q
        q_mesh (list, optional): q_mesh for DOS etc.
        remap_fc (bool, optional): write remapped force_constants
        output_dir (str, optional): ]. Defaults to "phonopy_output".
        verbose (bool, optional): be verbose
    """
    timer = Timer("\nExtract phonopy results:")
    if not q_mesh:
        if phonon.mesh_numbers is None:
            q_mesh = defaults.kwargs.q_mesh.copy()
        else:
            q_mesh = list(phonon.mesh_numbers)

    talk(f".. q_mesh:   {q_mesh}")

    primitive = to_Atoms(phonon.get_unitcell())
    supercell = to_Atoms(phonon.get_supercell())

    if remap_fc:
        talk(f".. write force constants (remapped)")
        fc_file = "FORCE_CONSTANTS_remapped_phonopy"
        p2s_map = None
    else:
        talk(f".. write force constants")
        fc_file = "FORCE_CONSTANTS"
        p2s_map = phonon.get_primitive().get_primitive_to_supercell_map()

    fc = phonon.get_force_constants()

    with cwd(output_dir, mkdir=True):
        if minimal_output:
            talk(f"Extract basic results:")
            talk(f".. write primitive cell")
            write(primitive, filenames.primitive)
            talk(f".. write supercell")
            write(supercell, filenames.supercell)

            talk(f".. write force constants to {fc_file}")
            write_FORCE_CONSTANTS(fc, filename=fc_file, p2s_map=p2s_map)
            write_FORCE_SETS(phonon.dataset)

            # yaml
            phpy_yaml = PhonopyYaml()
            phpy_yaml.set_phonon_info(phonon)
            with open("phonopy.yaml", "w") as f:
                f.write(str(phpy_yaml))

        if bandstructure:
            talk(f"Extract bandstructure")
            talk(f".. write yaml")
            wrapper.set_bandstructure(phonon, paths=bz_path)
            phonon.write_yaml_band_structure()

        if dos or thermal_properties or debye:
            talk(f"Run mesh")
            phonon.run_mesh(q_mesh)

        if thermal_properties:
            talk(f"Extract thermal properties")
            talk(f".. write yaml")
            phonon.run_thermal_properties()
            phonon.write_yaml_thermal_properties()

        if debye:
            talk("Extract Debye Temperatur")
            debye_temp = wrapper.get_debye_temperature(phonon)
            with open("debye.dat", "w") as f:
                f.write(str(debye_temp[0]))
            talk(f".. Debye temperature: {debye_temp[0]:.2f}K written to file.")

        if dos:
            talk(f"Extract DOS:")
            talk(f".. write")
            dos = wrapper.get_dos(phonon, write=True)

        if pdos:
            talk(f"Extract projected DOS")
            talk(f".. write")
            phonon.run_mesh(q_mesh, with_eigenvectors=True, is_mesh_symmetry=False)
            pdos = wrapper.get_dos(phonon, total=False, write=True)

        animate_q_points = {}
        if animate:
            animate_q_points = get_special_points(primitive)

        elif animate_q:
            for q_point in animate_q:
                key = "_".join(str(q) for q in q_point)
                animate_q_points.update({f"{key}": q_point})

        for key, val in animate_q_points.items():
            path = Path("animation")
            path.mkdir(exist_ok=True)
            outfile = path / f"animation_{key}.ascii"
            wrapper.get_animation(phonon, val, outfile)
            talk(f".. {outfile} written")

    timer(f"all files written to {output_dir}")

    # checks
    check_negative_frequencies(phonon)

    if verbose:
        print_frequencies_at_gamma(phonon)


def plot_results(
    phonon,
    thermal_properties=False,
    bandstructure=False,
    dos=False,
    pdos=False,
    bz_path=None,
    output_dir="phonopy_output",
    run_mesh=False,
):
    """Plot results from phonopy object and present them.

    Args:
        phonon (phonopy.Phonopy): The Phonopy Object with calculated force constants
        thermal_properties (bool, optional): write and plot thermal properties
        bandstructure (bool, optional): write and plot bandstructure
        dos (bool, optional): write and plot DOS
        pdos (bool, optional): write and plot projected DOS
        bz_path (list, optional): Brillouin zone path for bandstructure
        output_dir (str, optional): ]. Defaults to "phonopy_output".
        run_mesh (bool): re-run the mesh
    """
    timer = Timer("\nPlot phonopy results:")

    with cwd(output_dir, mkdir=True):

        if thermal_properties:
            talk(f"Plot thermal properties")
            wrapper.plot_thermal_properties(phonon)

        if bandstructure:
            talk(f"Plot bandstructure")
            wrapper.plot_bandstructure(phonon, paths=bz_path)

        if dos:
            talk(f"Plot DOS:")
            wrapper.plot_bandstructure_and_dos(phonon, run_mesh=run_mesh)

        if pdos:
            talk(f"Plot projected DOS")
            wrapper.plot_bandstructure_and_dos(phonon, partial=True, run_mesh=run_mesh)

    timer(f"all files written to {output_dir}")
