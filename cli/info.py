"""`vibes info` backend"""
from pathlib import Path

import numpy as np

from vibes import defaults, keys
from vibes.filenames import filenames

from .misc import ClickAliasedGroup, click, complete_files


# from click 7.1 on
_default_context_settings = {"show_default": True}


@click.command(cls=ClickAliasedGroup)
def info():
    """inform about content of a file"""


@info.command()
@click.argument("file", type=complete_files)
@click.pass_obj
def settings(obj, file):
    """write the settings in FILE *including* the configuration"""
    from vibes.settings import Settings

    click.echo(f"List content of {file} including system-wide configuration")

    settings = Settings(file)
    settings.print()


@info.command()
@click.argument("file", default=filenames.atoms, type=complete_files)
@click.option("--format", default="aims", show_default=True)
@click.option("-t", "--symprec", default=1e-5, show_default=True)
@click.option("-v", "--verbose", is_flag=True, help="increase verbosity")
@click.pass_obj
def geometry(obj, file, format, symprec, verbose):
    """inform about a structure in a geometry input file"""
    from ase.io import read

    from vibes.structure.io import inform

    atoms = read(file, format=format)

    verbosity = 1
    if verbose:
        verbosity = 2

    inform(atoms, symprec=symprec, verbosity=verbosity)


@info.command(context_settings=_default_context_settings)
@click.argument("file", default=filenames.trajectory, type=complete_files)
@click.option("-p", "--plot", is_flag=True, help="plot a summary")
@click.option("--avg", default=100, help="window size for running avg")
@click.option("-v", "--verbose", is_flag=True, help="be verbose")
def md(file, plot, avg, verbose):
    """inform about MD simulation in FILE"""
    import xarray as xr

    from vibes.trajectory import analysis as al
    from vibes.trajectory import reader

    from .scripts.md_sum import md_sum

    file = Path(file)

    if file.suffix in (".son", ".yaml", ".bz", ".gz"):
        trajectory = reader(file)
        DS = trajectory.dataset
    elif file.suffix in (".nc"):
        DS = xr.load_dataset(file)
    elif file.suffix in (".log"):
        md_sum(file, plot, avg, verbose)
    else:
        raise click.FileError(f"File format of {file} not known.")

    click.echo(f"Dataset summary for {file}:")
    al.summary(DS, plot=plot, avg=avg)


@info.command(context_settings=_default_context_settings)
@click.argument("file", default="phonopy.in", type=complete_files)
@click.option("--write_supercell", is_flag=True, help="write the supercell to file")
def phonopy(file, write_supercell):
    """inform about a phonopy calculation based on the input FILE"""
    from .scripts.vibes_phonopy import preprocess

    preprocess(settings_file=file, write_supercell=write_supercell)


@info.command()
@click.argument("file", default=filenames.trajectory, type=complete_files)
def trajectory(file):
    """print metadata from trajectory in FILE"""
    from vibes import son
    from vibes.settings import Settings

    metadata, _ = son.open(file)

    click.echo(f"Summary of metadata in {file}:\n")
    click.echo("Keys:")
    click.echo(f"  {list(metadata.keys())}\n")
    if "settings" in metadata:
        settings = Settings.from_dict(metadata["settings"])
        click.echo("Settings:")
        settings.print()


@info.command()
@click.argument("file", type=complete_files)
@click.option("--key", type=str, default=None, help="print DS[key]")
def netcdf(file, key):
    """show contents of netCDF FILE"""
    import xarray as xr

    DS = xr.open_dataset(file)

    if key is not None:
        print(DS[key])
    else:
        print(DS)


@info.command(context_settings=_default_context_settings)
@click.argument("file", type=complete_files)
@click.option("--describe", is_flag=True, help="print description of data")
@click.option("--half", is_flag=True, help="print only the second half of data")
@click.option("--to_json", type=Path, help="Write to json file")
def csv(file, describe, half, to_json):
    """show contents of csv FILE"""
    import json

    import pandas as pd

    df = pd.read_csv(file)

    pd.options.display.max_rows = len(df)
    pd.options.display.max_columns = len(df.columns)

    if half:
        df = df.iloc[len(df) // 2 :]

    if describe:
        df = df.describe()

    click.echo(df)

    if to_json is not None:
        click.echo(f".. write to {to_json}")
        with open(to_json, "w") as f:
            json.dump(df.to_dict(), f, indent=1)


@info.command(aliases=["gk"], context_settings=_default_context_settings)
@click.argument("files", nargs=-1)
@click.option("-p", "--plot", is_flag=True, help="plot summary")
@click.option("--logx", is_flag=True)
@click.option("--xlim", type=float, default=None)
@click.option("--cmap", default="colorblind", help="the matplotlib colormap")
@click.option("--outfile", default="greenkubo_summary.pdf")
def greenkubo(files, plot, logx, xlim, cmap, outfile):
    """visualize heat flux and thermal conductivity"""
    import xarray as xr

    if len(files) < 1:
        files = ["greenkubo.nc"]

    datasets = [xr.load_dataset(f) for f in files]

    # concatentate trajectories
    ds_gk = xr.concat(datasets, dim=keys.trajectory)
    attrs = {ii: d.attrs for (ii, d) in enumerate(datasets)}
    ds_gk.attrs = attrs

    ks = np.array([np.diag(k) for k in ds_gk.kappa]).flatten()
    k_mean = ks.mean()
    k_err = (ks.var() / (len(ks))) ** 0.5
    click.echo(f"Kappa:    {k_mean:.3f} +/- {k_err:.3f}")
    click.echo(f"Kappa^ab: {ds_gk.kappa.mean(axis=0)}")

    if plot:
        import seaborn as sns
        from matplotlib import pyplot as plt

        from vibes.helpers.xarray import xtrace

        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

        colors = sns.color_palette(cmap, n_colors=3)  # plt.get_cmap(cmap)

        cutoff_time = ds_gk.cutoff_time.mean(axis=0)
        j_raw = ds_gk.heat_flux_autocorrelation.mean(axis=0)
        k_raw = ds_gk.heat_flux_autocorrelation_cumtrapz.mean(axis=0)
        j_filtered = ds_gk.heat_flux_autocorrelation_filtered.mean(axis=0)
        k_filtered = ds_gk.heat_flux_autocorrelation_cumtrapz_filtered.mean(axis=0)

        k_total = xtrace(k_raw) / 3  # .mean(axis=1)
        k_total_filteres = xtrace(k_filtered) / 3  # .mean(axis=1)

        for ii in range(3):
            c = colors[ii]

            j = j_filtered[:, ii, ii]
            j.to_series().plot(ax=ax1, c=c, lw=2)

            # unfiltered kappa
            k = k_raw[:, ii, ii]
            k.to_series().plot(ax=ax2, c=c, alpha=0.75)

            k = k_filtered[:, ii, ii]
            k.to_series().plot(ax=ax2, c=c)

            ta = cutoff_time[ii, ii]
            ax1.axvline(ta, c=c, lw=2)
            ax2.axvline(ta, c=c, lw=2)

            # unfiltered hfacf (and restore ylim)
            ylim = ax1.get_ylim()
            j = j_raw[:, ii, ii]
            j.to_series().plot(ax=ax1, c=c, lw=0.1, zorder=-1)
            ax1.set_ylim(ylim)

        # mean of k
        k_total.to_series().plot(ax=ax2, c="k", alpha=0.75)
        k_total_filteres.to_series().plot(ax=ax2, c="k")

        ax1.axhline(0, c="k")
        ax1.set_ylim([j_filtered.min(), 1.2 * j_filtered.max()])

        # plot final kappa
        ax2.axhline(k_mean, c="k")
        ax2.fill_between(j.time, k_mean + k_err, k_mean - k_err, color="k", alpha=0.1)

        ax1.set_ylabel("$J_{\\rm corr} (t)$")
        ax2.set_ylabel("$\\kappa (t)$")

        kappa_str = f"$\\kappa$: {k_mean:.2f} +/- {k_err:.2f} W/mK"

        ax1.set_title(kappa_str)

        tmax = 3 * np.diag(cutoff_time).max()

        if xlim is None:
            xlim = tmax

        if logx:
            ax2.set_xlim([0.01 * xlim, xlim])
            ax2.set_xscale("log")
        else:
            ax2.set_xlim([0, xlim])

        fig.savefig(outfile, bbox_inches="tight")
        click.echo(f".. summary plotted to {outfile}")


@info.command(context_settings=_default_context_settings)
@click.argument("file", default=filenames.trajectory_dataset, type=complete_files)
@click.option("-o", "--output_file", default="vdos.csv")
@click.option("-p", "--plot", is_flag=True, help="plot the DOS")
@click.option("--prominence", default=defaults.filter_prominence, help="for find_peaks")
@click.option("-mf", "--max_frequency", type=float, help="max. freq. in THz")
@click.option("--npad", default=10000, help="number of zeros for padding")
def vdos(file, output_file, plot, prominence, max_frequency, npad):
    """compute and write velocity autocorrelation function to output file"""
    import xarray as xr

    from vibes.green_kubo.velocities import get_vdos, simple_plot

    click.echo(f"Read {file} and extract velocities")
    velocities = xr.open_dataset(file).velocities

    vdos = get_vdos(velocities=velocities, hann=False, npad=npad, verbose=True)

    # sum atoms and coordinates
    df = vdos.real.sum(axis=(1, 2)).to_series()

    if plot:
        simple_plot(df, prominence=prominence, max_frequency=max_frequency)

    click.echo(f".. write VDOS to {output_file}")
    df.to_csv(output_file, index_label="omega", header=True)


@info.command(context_settings=_default_context_settings)
@click.argument("file", default=filenames.trajectory, type=complete_files)
@click.option("-v", "--verbose", is_flag=True, help="show more information")
@click.pass_obj
def relaxation(obj, file, verbose):
    """summarize geometry optimization in FILE"""
    from ase.constraints import full_3x3_to_voigt_6_stress

    from vibes.relaxation._defaults import keys, kwargs, name, relaxation_options
    from vibes.relaxation.context import MyExpCellFilter as ExpCellFilter
    from vibes.spglib.wrapper import get_spacegroup
    from vibes.trajectory import reader

    traj, metadata = reader(file, get_metadata=True, verbose=False)

    relaxation_kwargs = metadata[name].get(relaxation_options, {})

    try:
        fmax = relaxation_kwargs[keys.fmax]
    except KeyError:
        fmax = kwargs[keys.fmax]
        click.echo(f"** fmax not found in {file}, use default value {fmax}")

    try:
        fix_symmetry = relaxation_kwargs[keys.fix_symmetry]
    except KeyError:
        fix_symmetry = kwargs[keys.fix_symmetry]
        msg = f"** `fix_symmetry` not found in {file}, use default value {fix_symmetry}"
        click.echo(msg)

    scalar_pressure = relaxation_kwargs.get(
        keys.scalar_pressure, kwargs[keys.scalar_pressure]
    )

    atoms_ref = traj[0]
    na = len(atoms_ref)

    energy_ref = atoms_ref.get_potential_energy()

    click.echo(f"Relaxation info for {file}:")
    if verbose:
        import json

        click.echo("Metadata for relaxation:")
        click.echo(json.dumps(metadata[name], indent=2))

    if fix_symmetry:
        from ase.spacegroup.symmetrize import FixSymmetry

        click.echo("fix_symmetry:     True")

    if scalar_pressure:
        click.echo(f"scalar_pressure: {scalar_pressure*1000: .3e} meV/A**3")

    click.echo(f"fmax:            {fmax*1000: .3e} meV/AA")
    click.echo(
        "# Step |   Free energy   |   F-F(1)   | max. force |  max. stress |"
        + "  Volume  |  Spacegroup  |"
        + "\n"
        + "#      |       [eV]      |    [meV]   |  [meV/AA]  |  [meV/AA^3]  |"
        + "  [AA^3]  |              |"
        + "\n"
    )

    for ii, atoms in enumerate(traj[1:]):

        energy = atoms.get_potential_energy()
        de = 1000 * (energy - energy_ref)

        opt_atoms = ExpCellFilter(atoms, scalar_pressure=scalar_pressure)

        forces = opt_atoms.get_forces()
        stress = full_3x3_to_voigt_6_stress(forces[na:])
        forces = forces[:na]  # drop the stress

        # optionally: symmetrize forces and stress
        if fix_symmetry:
            constr = FixSymmetry(atoms, symprec=kwargs[keys.symprec])
            constr.adjust_forces(atoms, forces)
            constr.adjust_stress(atoms, stress)

        res_forces = (forces ** 2).sum(axis=1).max() ** 0.5 * 1000
        res_stress = abs(stress).max() * 1000

        vol_str = f"{atoms.get_volume():10.3f}"
        # sg_str = f"{get_spacegroup(atoms):5d}"
        sg_str = get_spacegroup(atoms)

        msg = "{:5d}   {:16.8f}  {:12.6f} {:12.4f} {:14.4f} {}   {}".format(
            ii + 1, energy, de, res_forces, res_stress, vol_str, sg_str,
        )
        click.echo(msg)

    if max(res_forces, res_stress) < fmax * 1000:
        click.echo("--> converged.")


@info.command(context_settings=_default_context_settings)
@click.argument("files", nargs=-1, type=complete_files)
@click.option("-o", "--outfile")
@click.option("--per_sample", is_flag=True, help="analyze per sample")
@click.option("--per_mode", is_flag=True, help="analyze per mode (no --per_sample)")
@click.option("--describe", is_flag=True)
@click.option("--dry", is_flag=True, help="don't write output files")
def anharmonicity(files, outfile, per_sample, per_mode, describe, dry):
    """Compute sigmaA for trajectory dataset in FILE"""
    import pandas as pd
    import xarray as xr

    from vibes import keys
    from vibes.anharmonicity_score import get_dataframe, get_sigma_per_mode

    if len(files) < 1:
        files = [filenames.trajectory_dataset]

    dfs = []
    for file in files:
        click.echo(f"Compute anharmonicity measure for {file}:")

        click.echo(f" parse {file}")

        DS = xr.open_dataset(file)

        name = DS.attrs[keys.system_name]
        if per_mode:
            df = get_sigma_per_mode(DS)
            outfile = outfile or f"sigmaA_mode_{name}.csv"
            index_label = keys.omega
        else:
            df = get_dataframe(DS, per_sample=per_sample)
            outfile = outfile or f"sigmaA_{name}.csv"
            index_label = "material"
        dfs.append(df)

    df = pd.concat(dfs)

    click.echo("\nDataFrame:")
    click.echo(df)
    if describe:
        click.echo("\nDataFrame.describe():")
        click.echo(df.describe())

    if outfile is not None and not dry:
        df.to_csv(outfile, index_label=index_label, float_format="%15.12e")
        click.echo(f"\n.. Dataframe for {name} written to {outfile}")
