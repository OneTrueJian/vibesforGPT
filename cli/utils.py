"""vibes CLI utils"""

from vibes import keys
from vibes.filenames import filenames

from .misc import AliasedGroup, ClickAliasedGroup, Path, click, complete_files


xrange = range

# from click 7.1 on
_default_context_settings = {"show_default": True}


@click.group(cls=ClickAliasedGroup)
def utils():
    """tools and utilities"""


@utils.command(context_settings=_default_context_settings)
@click.argument("file", type=complete_files)
@click.option("--dry", is_flag=True, help="Only print hash to stdout")
@click.option("-o", "--outfile", default="hash.toml")
def hash(file, dry, outfile):
    """create SHA1 hash for FILE"""
    import time

    from vibes.helpers.hash import hash_file
    from vibes.helpers.utils import talk

    timestr = time.strftime("%Y/%m/%d_%H:%M:%S")

    hash = hash_file(file)
    talk(f'Hash for "{file}":\n  {hash}')

    if not dry:
        with open(outfile, "a") as f:
            f.write(f"\n# {timestr}")
            f.write(f'\n"{file}": "{hash}"')
        talk(f".. written to {outfile}")


@utils.command(cls=AliasedGroup)
def geometry():
    """utils for manipulating structures (wrap, refine, etc.)"""
    ...


@geometry.command(context_settings=_default_context_settings)
@click.argument("file", type=complete_files)
@click.option("-d", "--density", default=3.5)
@click.option("--uneven", is_flag=True)
@click.option("--format", default="aims")
def suggest_k_grid(file, density, uneven, format):
    """suggest a k_grid for geometry in FILENAME based on density"""
    from .scripts.suggest_k_grid import suggest_k_grid

    click.echo("vibes CLI: suggest_k_grid")
    suggest_k_grid(file, density, uneven, format)


@geometry.command(context_settings=_default_context_settings)  # aliases=['gd'])
@click.argument("files", nargs=2, type=complete_files)
@click.option("-sc_file", "--supercell_file", type=Path)
@click.option("--outfile", default=filenames.deformation, type=Path)
@click.option("--dry", is_flag=True)
@click.option("--format", default="aims")
def get_deformation(files, supercell_file, outfile, dry, format):
    """get matrix D that deformes lattice of geometry1 to lattice of geometry2, where
        A_2 = D A_1

    For Supercells:
        S = M A  =>  S_2 = M D A_1  = M D M^-1 S_1  = D_s S_1
    """
    import numpy as np
    from ase.io import read

    from vibes.helpers.geometry import get_deformation

    atoms1 = read(files[0], format=format)
    atoms2 = read(files[1], format=format)

    D = get_deformation(atoms1.cell, atoms2.cell).round(14)

    # supercell?
    if supercell_file is not None:
        click.echo(f"Reference supercell from: {supercell_file}")
        sc = read(supercell_file, format=format)
        M = get_deformation(atoms1.cell, sc.cell).round(14)

        D = M @ D @ np.linalg.inv(M)

        outfile = f"{outfile.stem}_supercell{outfile.suffix}"

    click.echo("Deformation tensor:")
    click.echo(D)

    if not dry:
        import numpy as np

        click.echo(f".. write to {outfile}")
        np.savetxt(outfile, D)


@geometry.command(context_settings=_default_context_settings)  # aliases=['ad'])
@click.argument("file", type=complete_files)
@click.option("--scalar", type=float, default=None)
@click.option("--deformation", type=complete_files, default=filenames.deformation)
@click.option("--outfile", type=Path)
@click.option("--dry", is_flag=True)
@click.option("--cartesian", is_flag=True)
@click.option("--format", default="aims")
def apply_deformation(file, scalar, deformation, outfile, dry, cartesian, format):
    """apply deformation tensor D to geometry in FILE"""
    import numpy as np
    from ase.io import read

    if scalar is not None:
        click.echo(f"Apply scalar deformation w/ factor {scalar}:")
        D = np.eye(3) * scalar
    else:
        click.echo(f"Apply deformation tensor from {deformation}:")
        D = np.loadtxt(deformation)
    atoms = read(file, format=format)

    click.echo(D)

    new_atoms = atoms.copy()
    new_atoms.set_cell(D @ atoms.cell[:], scale_atoms=True)

    velocities = False
    if atoms.get_velocities() is not None:
        velocities = True
        new_atoms.set_velocities(atoms.get_velocities())
        click.echo("** velocities are not scaled")

    info_str = f"Deformation tensor: {D.tolist()}"

    if not dry:
        outfile = outfile or file + ".deformed"
        click.echo(f".. write structure to {outfile}")
        kw = {
            "info_str": info_str,
            "scaled": not cartesian,
            "format": format,
            "velocities": velocities,
        }
        new_atoms.write(outfile, **kw)


@geometry.command(context_settings=_default_context_settings)
@click.argument("files", nargs=2, type=complete_files)
@click.option("--cartesian", is_flag=True)
@click.option("--format", default="aims")
def set_cell(files, cartesian, format):
    """set cell of geometry in file1 to geometry in file2"""
    from ase.io import read

    atoms1 = read(files[0], format=format)
    atoms2 = read(files[1], format=format)

    velocities = atoms2.get_velocities()

    click.echo(f"Replace lattice in {files[1]} by lattice from {files[0]}")
    atoms2.set_cell(atoms1.cell[:], scale_atoms=True)
    if velocities is not None:
        click.echo("** velocities are not scaled")

    click.echo(f".. write structure to {files[1]}")
    kw = {
        "scaled": not cartesian,
        "format": format,
        "velocities": velocities is not None,
    }
    atoms2.write(files[1], **kw)


@geometry.command("2frac", context_settings=_default_context_settings)
@click.argument("file", default=filenames.atoms, type=complete_files)
@click.option("-o", "--outfile")
@click.option("--format", default="aims")
def to_fractional(file, outfile, format):
    """rewrite geometry in fractional coordinates"""
    from ase.io import read

    if not outfile:
        outfile = file + ".fractional"

    atoms = read(file, format=format)
    atoms.write(outfile, format=format, scaled=True, geo_constrain=True)

    click.echo(f"Geometry written to {outfile}")


@geometry.command("2cart", context_settings=_default_context_settings)
@click.argument("file", default=filenames.atoms, type=complete_files)
@click.option("-o", "--outfile")
@click.option("--format", default="aims")
def to_cartesian(file, outfile, format):
    """rewrite geometry in cartesian coordinates"""
    from ase.io import read

    if not outfile:
        outfile = file + ".cartesian"

    atoms = read(file, format=format)
    atoms.write(outfile, format=format, scaled=False)

    click.echo(f"Geometry written to {outfile}")


@geometry.command("refine", context_settings=_default_context_settings)
@click.argument("file", type=complete_files)
@click.option("-prim", "--primitive", is_flag=True)
@click.option("-conv", "--conventional", is_flag=True)
@click.option("--center", is_flag=True)
@click.option("--origin", is_flag=True)
@click.option("-cart", "--cartesian", is_flag=True)
@click.option("--format", default="aims")
@click.option("-t", "--symprec", default=1e-5)
def geometry_refine(*args, **kwargs):
    """vibes.scripts.refine_geometry"""
    from .scripts.refine_geometry import refine_geometry

    refine_geometry(*args, **kwargs)


@geometry.command(context_settings=_default_context_settings)
@click.argument("file", default=filenames.atoms, type=complete_files)
@click.option("-o", "--outfile")
@click.option("--format", default="aims")
def wrap(file, outfile, format):
    """wrap atoms in FILE to the cell"""
    from ase.io import read

    if not outfile:
        outfile = file + ".wrapped"

    atoms = read(file, format=format)
    atoms.positions -= [0.1, 0.1, 0.1]
    atoms.wrap(pretty_translation=True)
    atoms.positions += [0.1, 0.1, 0.1]
    atoms.wrap()
    atoms.write(outfile, format=format, scaled=True, geo_constrain=True, wrap=False)

    click.echo(f"Wrapped geometry written to {outfile}")


@utils.command(context_settings=_default_context_settings)
@click.argument("file", default=filenames.atoms, type=complete_files)
@click.option("-d", "--dimension", type=int, nargs=9, help="9 values for SC matrix")
@click.option("-dd", "--diagonal_dimension", type=int, nargs=3, help="3 values")
@click.option("-n", "--n_target", type=int, help="target size of the supercell")
@click.option("-o", "--outfile")
@click.option("--deviation", default=0.2)
@click.option("--dry", is_flag=True)
@click.option("--format", default="aims")
def make_supercell(
    file,
    dimension,
    diagonal_dimension,
    outfile,
    n_target,
    deviation,
    dry,
    format,
):
    """create a supercell of desired shape or size"""
    from .scripts.make_supercell import make_supercell

    if diagonal_dimension:
        dimension = diagonal_dimension

    make_supercell(
        file,
        dimension,
        n_target,
        deviation,
        dry,
        format,
        outfile=outfile,
    )


@utils.command(context_settings=_default_context_settings)
@click.argument("filename", type=complete_files)
@click.option("-T", "--temperature", type=float, help="Temperature in Kelvin")
@click.option("-n", "--n_samples", type=int, default=1, help="number of samples")
@click.option("-fc", "--fc_file", type=complete_files, help="remapped force constants")
@click.option("--rattle", type=float, help="atoms.rattle(stdev=X) (ASE default: 0.001)")
@click.option("--quantum", is_flag=True, help="use Bose-Einstein distribution")
# @click.option("--deterministic", is_flag=True, help="create a deterministic sample")
@click.option("--zacharias", is_flag=True, help="Zacharias one-shot sampling")
@click.option("--ignore_negative", is_flag=True, help="freeze imaginary modes")
@click.option("-seed", "--random_seed", type=int, help="seed for random numbers")
@click.option("--propagate", type=float, help="propagate this many fs")
@click.option("--format", default="aims")
def create_samples(filename, **kwargs):
    """create samples from geometry in FILENAME"""
    from .scripts.create_samples import create_samples

    click.echo("vibes CLI: create_samples")
    create_samples(atoms_file=filename, **kwargs)


@utils.group(aliases=["fc"])
def force_constants():
    """utils for working with force constants"""
    ...


@force_constants.command(context_settings=_default_context_settings)
@click.argument("file", default="FORCE_CONSTANTS", type=complete_files)
@click.option("-uc", "--primitive", default=filenames.primitive)
@click.option("-sc", "--supercell", default=filenames.supercell)
@click.option("-nsc", "--new_supercell")
@click.option("-o", "--outfile")
@click.option("--symmetrize", is_flag=True)
@click.option("--format", default="aims")
def remap(
    file,
    primitive,
    supercell,
    new_supercell,
    outfile,
    symmetrize,
    format="aims",
):
    """remap phonopy force constants in FILENAME to [3N, 3N] shape"""
    # copy: from vibes.scripts.remap_phonopy_forceconstants import remap_force_constants
    import numpy as np
    from ase.io import read

    from vibes.io import parse_force_constants
    from vibes.phonopy.utils import remap_force_constants

    uc = read(primitive, format=format)
    sc = read(supercell, format=format)

    nsc = None
    if new_supercell:
        nsc = read(new_supercell, format=format)

    kwargs = {
        "primitive": uc,
        "supercell": sc,
    }

    fc = parse_force_constants(fc_file=file, two_dim=False, **kwargs)

    kwargs.update({"new_supercell": nsc, "two_dim": True, "symmetrize": symmetrize})

    fc = remap_force_constants(fc, **kwargs)

    if not outfile:
        outfile = f"{file}_remapped"

    msg = f"remapped force constants from {file}, shape [{fc.shape}]"
    np.savetxt(outfile, fc, header=msg)

    click.echo(f".. remapped force constants written to {outfile}")


@force_constants.command(context_settings=_default_context_settings)
@click.argument("file", default="FORCE_CONSTANTS_remapped", type=complete_files)
@click.option("-sc", "--supercell", default=filenames.supercell)
@click.option("-n", "--show_n_frequencies", default=6, type=int)
@click.option("-o", "--outfile", default="frequencies.dat")
@click.option("--symmetrize", is_flag=True)
@click.option("--format", default="aims")
def frequencies(file, supercell, show_n_frequencies, outfile, symmetrize, format):
    """compute the frequencies for remapped force constants"""
    import numpy as np
    from ase.io import read

    from vibes.harmonic_analysis.dynamical_matrix import get_frequencies

    atoms = read(supercell, format=format)
    fc = np.loadtxt(file)

    w2 = get_frequencies(fc, masses=atoms.get_masses())

    if show_n_frequencies:
        nn = show_n_frequencies
        print(f"The first {nn} frequencies:")
        for ii, freq in enumerate(w2[:nn]):
            print(f" {ii + 1:4d}: {freq}")

        print(f"Highest {nn} frequencies")
        for ii, freq in enumerate(w2[-nn:]):
            print(f" {len(w2) - ii:4d}: {freq }")

    if isinstance(outfile, str):
        np.savetxt(outfile, w2)
        click.echo(f".. frequencies written to {outfile}")


@utils.group(cls=ClickAliasedGroup)
def trajectory():
    """trajectory utils"""


@trajectory.command("join", context_settings=_default_context_settings)
@click.argument("files", nargs=-1, type=complete_files)
@click.option("--dim", default=keys.trajectory)
@click.option("-o", "--outfile", default="joined.nc")
def join_xr(files, dim, outfile):
    """join Datasets in FILES on dim=dim"""
    import xarray as xr

    dss = [xr.load_dataset(f) for f in files]

    ds = xr.concat(dss, dim=dim)
    ds.attrs = dss[0].attrs

    ds.to_netcdf(outfile)
    click.echo(f"Joined datasets written to {outfile}")


@trajectory.command(context_settings=_default_context_settings)
@click.argument("file", default=filenames.trajectory, type=complete_files)
@click.option("-n", "--n_steps", type=int, help="discard this many steps")
def discard(file, n_steps):
    """discard this many steps from trajectory"""
    from vibes.trajectory import reader

    traj = reader(file)

    click.echo(f"Discard {n_steps} steps")
    traj = traj[n_steps:-1]

    p = Path(file)
    outfile = f"{p.stem}_discarded{p.suffix}"

    traj.write(outfile)


@trajectory.command("2tdep", context_settings=_default_context_settings)
@click.argument("file", default=filenames.trajectory, type=complete_files)
@click.option("-s", "--skip", default=1, help="skip this many steps from trajectory")
@click.option("--folder", default="tdep", help="folder to store input")
def t2tdep(file, skip, folder):
    """extract tdep input files from trajectory in FILENAME"""
    from vibes.trajectory import reader

    traj = reader(file)
    traj.to_tdep(folder=folder, skip=skip)


@trajectory.command("2xyz", context_settings=_default_context_settings)
@click.argument("file", default=filenames.trajectory, type=complete_files)
@click.option("-o", "--outfile", default="trajectory.xyz")
def t2xyz(file, outfile):
    """extract trajectory in FILENAME and store as xyz file"""
    from vibes.trajectory import reader

    traj = reader(file)
    traj.to_xyz(file=outfile)


@trajectory.command("2db", context_settings=_default_context_settings)
@click.argument("file", default=filenames.trajectory, type=complete_files)
@click.option("-o", "--outfile", default="trajectory.db")
def t2db(file, outfile):
    """extract trajectory in FILENAME and store as ase db"""
    from vibes.trajectory import reader

    traj = reader(file)
    traj.to_db(outfile)


@trajectory.command("2traj", context_settings=_default_context_settings)
@click.argument("file", default=filenames.trajectory, type=complete_files)
@click.option("-o", "--outfile", default="trajectory.traj")
def t2traj(file, outfile):
    """extract trajectory in FILENAME and store as ase trajectory"""
    from vibes.trajectory import reader

    traj = reader(file)
    traj.to_traj(outfile)


@trajectory.command("2csv", context_settings=_default_context_settings)
@click.argument("file", default=filenames.trajectory, type=complete_files)
@click.option("-o", "--outfile", default="trajectory.csv")
def t2csv(file, outfile):
    """extract and store 1D data from trajectory in FILENAME """
    from vibes.trajectory import reader

    traj = reader(file)
    df = traj.dataframe

    click.echo(f"Write trajectory data to {outfile}")
    df.to_csv(outfile)


@trajectory.command(context_settings=_default_context_settings)
@click.argument("file", default=filenames.trajectory, type=complete_files)
@click.option("-uc", help="Add a (primitive) unit cell", type=complete_files)
@click.option("-sc", help="Add the respective supercell", type=complete_files)
@click.option("-fc", help="Add the force constants", type=complete_files)
@click.option("-rfc", help="Add remapped force constants", type=complete_files)
@click.option("-o", "--outfile")
@click.option("--format", default="aims")
def update(file, uc, sc, fc, rfc, outfile, format):
    """update reference data in trajectory file"""
    # copy: from vibes.scripts.update_md_trajectory import update_trajectory
    import shutil

    from ase.io import read

    from vibes.io import parse_force_constants
    from vibes.trajectory import reader

    traj = reader(file, fc_file=fc)

    if uc:
        atoms = read(uc, format=format)
        traj.primitive = atoms

    if sc:
        atoms = read(sc, format=format)
        traj.supercell = atoms
        traj.set_displacements()

    if rfc:
        fc = parse_force_constants(rfc)
        print(fc)
        traj.set_force_constants_remapped(fc)

    if not outfile:
        suffix = Path(file).suffix
        new_trajectory = f"temp{suffix}"
        fname = f"{file}.bak"
        click.echo(f".. back up old trajectory to {fname}")
        shutil.copy(file, fname)

    else:
        new_trajectory = outfile

    traj.write(file=new_trajectory)

    if not outfile:
        click.echo(f".. move new trajectory to {file}")
        shutil.move(new_trajectory, file)


@trajectory.command(aliases=["pick"], context_settings=_default_context_settings)
@click.argument("file", default=filenames.trajectory, type=complete_files)
@click.option("-o", "--outfile", type=Path)
@click.option("-n", "--number", default=0)
@click.option(
    "-r", "--range", type=int, nargs=3, help="start, stop, step", default=None
)
@click.option("-cart", "--cartesian", is_flag=True, help="write cart. coords")
def pick_samples(file, outfile, number, range, cartesian):
    """pick samples from trajectory"""
    from vibes.trajectory import reader

    click.echo(f"Read trajectory from {file}:")
    traj = reader(file)

    if range is not None:
        rge = xrange(*range)
    else:
        rge = [number]

    for number in rge:
        number_positive = number
        if number < 0:
            number_positive += len(traj)
        click.echo(f"Extract sample {number_positive}:")
        file = outfile or f"{filenames.atoms}.{number_positive:05d}"
        atoms = traj[number]
        info_str = f"Sample no.: {number_positive:7d}"
        kw = {"velocities": True, "scaled": not cartesian, "info_str": info_str}
        atoms.write(file, format="aims", **kw)
        click.echo(f".. sample written to {file}")


@trajectory.command("average", context_settings=_default_context_settings)
@click.argument("file", default=filenames.trajectory, type=complete_files)
@click.option("-r", "--range", type=int, nargs=3, help="start, stop, step")
@click.option("-cart", "--cartesian", is_flag=True, help="write cart. coords")
@click.option("-o", "--outfile", default="geometry.in.average")
def average_trajectory(file, range, cartesian, outfile):
    """average positions"""
    from vibes.trajectory import reader

    click.echo(f"Read trajectory from {file}:")
    traj = reader(file)

    if len(range) == 3:
        rge = xrange(*range)
        traj = traj[rge]

    atoms = traj.average_atoms
    atoms.wrap(pretty_translation=True)
    atoms.write(outfile, format="aims", scaled=not cartesian)
    click.echo(f".. geometry with averaged positions written to {outfile}")


@trajectory.command(context_settings=_default_context_settings)
@click.argument("file", default=filenames.trajectory, type=complete_files)
def clean_nsteps(file):
    """remove duplicate nsteps from trajectory.son"""
    from vibes import keys
    from vibes.trajectory import Trajectory, reader

    click.echo(f"Read trajectory from {file}:")
    traj = reader(file)

    click.echo(f"Clean trajectory")
    list_atoms = []
    list_steps = []
    for atoms in traj:
        nsteps = atoms.info.get(keys.nsteps)
        if nsteps not in list_steps:
            list_steps.append(nsteps)
            list_atoms.append(atoms)
        else:
            click.echo(f".. discard step {nsteps}")

    new_traj = Trajectory(list_atoms, metadata=traj.metadata)

    outfile = str(file) + ".cleaned"

    click.echo(f"Write to {outfile}")
    new_traj.write(outfile)


@utils.command("backup", context_settings=_default_context_settings)
@click.argument("folder", type=complete_files)
@click.option("--target", default=keys.default_backup_folder)
@click.option("--nozip", is_flag=True)
def perform_backup(folder, target, nozip):
    """backup FOLDER to TARGET"""
    from vibes.helpers.backup import backup_folder

    backup_folder(folder, target_folder=target, zip=not nozip)


@utils.group(aliases=["dev"])
def developer():
    """developer utils for FHI-vibes. Use at own risk"""
    ...


@developer.group()
def phono3py():
    """utils for working with phono3py"""
    ...


@phono3py.command(context_settings=_default_context_settings)
@click.argument("folder", default="output", type=complete_files)
@click.option("--q_mesh", nargs=3, help="q_mesh")
@click.option("--outfile", default="kappa_QMESH.log")
@click.pass_obj
def run_thermal_conductivity(obj, folder, q_mesh, outfile):
    """run a phono3py thermal conductivity calculation in FOLDER"""
    import sys

    from vibes.cli.scripts import run_thermal_conductivity as rtc
    from vibes.helpers.utils import talk
    from vibes.phono3py._defaults import kwargs

    if q_mesh is None:
        q_mesh = kwargs.q_mesh

    outfile = outfile.replace("QMESH", ".".join((str(q) for q in q_mesh)))

    talk("Run thermal conductivity")
    talk(f"Log will be written to {outfile}")

    sys.stdout = open(outfile, "w")

    rtc.run_thermal_conductivity_in_folder(folder, mesh=q_mesh)
