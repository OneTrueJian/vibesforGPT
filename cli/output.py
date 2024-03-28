"""`vibes output` part of the CLI"""
from pathlib import Path

import click

from vibes import defaults
from vibes.filenames import filenames

from .misc import ClickAliasedGroup as AliasedGroup
from .misc import complete_files, default_context_settings


_default_context_settings = {"show_default": True}


@click.command(cls=AliasedGroup)
def output():
    """produce output of vibes workfow"""


@output.command(aliases=["md"], context_settings=default_context_settings)
@click.argument("file", default=filenames.trajectory, type=complete_files)
@click.option("-hf", "--heat_flux", is_flag=True, help="write heat flux dataset")
@click.option("-d", "--discard", type=int, help="discard this many steps")
@click.option("--minimal", is_flag=True, help="omit redundant information")
@click.option("-fc", "--fc_file", type=Path, help="add force constants from file")
@click.option("-o", "--outfile", default="auto", show_default=True)
@click.option("--force", is_flag=True, help="enfore parsing of output file")
def trajectory(file, heat_flux, discard, minimal, fc_file, outfile, force):
    """write trajectory data in FILE to xarray.Dataset"""
    from vibes import keys
    from vibes.trajectory import reader
    from vibes.trajectory.dataset import get_trajectory_dataset

    if "auto" in outfile.lower():
        outfile = Path(file).stem
        outfile += ".nc"
    outfile = Path(outfile)

    file_size = Path(file).stat().st_size
    if not force and outfile.exists():
        import xarray as xr

        click.echo(f"Check if {file} has been parsed already")
        file_size_is = xr.open_dataset(outfile).attrs.get(keys.st_size)

        if file_size == file_size_is:
            click.echo(f".. file size has not changed, skip.")
            click.echo(".. (use --force to parse anyway)")
            return
        else:
            click.echo(".. file size has changed, parse the file.")

    click.echo(f"Extract Trajectory dataset from {file}")
    traj = reader(file=file, fc_file=fc_file)

    if discard:
        traj = traj.discard(discard)

    # harmonic forces?
    if fc_file:
        traj.set_forces_harmonic()

    if heat_flux:
        traj.compute_heat_flux_from_stresses()

    DS = get_trajectory_dataset(traj, metadata=True)
    # attach file size
    DS.attrs.update({keys.st_size: file_size})
    # write to disk
    DS.to_netcdf(outfile)
    click.echo(f"Trajectory dataset written to {outfile}")


@output.command(context_settings=default_context_settings)
@click.argument("file", default=filenames.trajectory, type=complete_files)
@click.option("-bs", "--bandstructure", is_flag=True, help="plot bandstructure")
@click.option("--dos", is_flag=True, help="plot DOS")
@click.option("--full", is_flag=True, help="include thermal properties and animation")
@click.option("--q_mesh", nargs=3, default=None, help="use this q-mesh")
@click.option("--debye", is_flag=True, help="compute Debye temperature")
@click.option("-pdos", "--projected_dos", is_flag=True, help="plot projected DOS")
@click.option("--born", type=complete_files, help="include file with BORN charges")
@click.option("--sum_rules", is_flag=True, help="enfore sum rules with hiphive")
@click.option("-v", "--verbose", is_flag=True, help="print frequencies at gamma point")
@click.pass_obj
def phonopy(
    obj,
    file,
    bandstructure,
    dos,
    full,
    q_mesh,
    debye,
    projected_dos,
    born,
    sum_rules,
    verbose,
):
    """perform phonopy postprocess for trajectory in FILE"""
    from vibes.phonopy import _defaults
    from vibes.phonopy.postprocess import extract_results, plot_results, postprocess

    phonon = postprocess(
        trajectory_file=file,
        born_charges_file=born,
        enforce_sum_rules=sum_rules,
    )
    if not q_mesh:
        if phonon.mesh_numbers is None:
            q_mesh = _defaults.kwargs.q_mesh.copy()
            click.echo(f"q_mesh not given, use default {q_mesh}")
        else:
            q_mesh = list(phonon.mesh_numbers)
            click.echo(f"q_mesh not given, use values stored in {file}: {q_mesh}")

    folder = "output"
    if sum_rules:
        folder += "_sum_rules"
    output_directory = Path(file).parent / folder

    kwargs = {
        "minimal_output": True,
        "thermal_properties": full,
        "bandstructure": bandstructure or full,
        "dos": dos or full,
        "debye": debye,
        "pdos": projected_dos,
        "q_mesh": q_mesh,
        "output_dir": output_directory,
        "animate": full,
        "verbose": verbose,
    }

    extract_results(phonon, **kwargs)

    kwargs = {
        "thermal_properties": full,
        "bandstructure": bandstructure or full,
        "dos": dos or full,
        "pdos": projected_dos,
        "output_dir": output_directory,
    }
    plot_results(phonon, **kwargs)


@output.command()
@click.argument("file", default="trajectory.son", type=complete_files)
# necessary?
@click.option("--q_mesh", nargs=3, default=None)
@click.pass_obj
def phono3py(obj, file, q_mesh):
    """perform phono3py postprocess for trajectory in FILE"""
    from vibes.phono3py._defaults import kwargs
    from vibes.phono3py.postprocess import extract_results, postprocess

    if not q_mesh:
        q_mesh = kwargs.q_mesh.copy()
        click.echo(f"q_mesh not given, use default {q_mesh}")

    phonon = postprocess(trajectory=file)

    output_directory = Path(file).parent / "output"

    extract_results(phonon, output_dir=output_directory)


@output.command(aliases=["gk"], context_settings=_default_context_settings)
@click.argument("file", default="trajectory.nc")
@click.option("-o", "--outfile", default="greenkubo.nc", type=Path)
@click.option("-w", "--window_factor", default=defaults.window_factor)
@click.option("--filter_prominence", default=defaults.filter_prominence)
@click.option("--total", is_flag=True, help="compute total flux")
# @click.option("-d", "--discard", default=0)
def greenkubo(file, outfile, window_factor, filter_prominence, total):
    """perform greenkubo analysis for dataset in FILE"""
    import xarray as xr

    import vibes.green_kubo as gk

    ds = xr.open_dataset(file)

    ds_gk = gk.get_gk_dataset(
        ds,
        window_factor=window_factor,
        filter_prominence=filter_prominence,
        total=total,
    )

    if total:
        outfile = outfile.parent / f"{outfile.stem}.total.nc"

    click.echo(f".. write to {outfile}")

    ds_gk.to_netcdf(outfile)
