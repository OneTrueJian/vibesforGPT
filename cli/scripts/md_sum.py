""" Summarize output from ASE.md class (in md.log) """

from argparse import ArgumentParser
from pathlib import Path
from warnings import warn

import numpy as np
from ase import units

from vibes.trajectory import reader


def parse_log(file):
    """ parse the ASE logfile, typically md.log """
    e_kin = []
    e_pot = []
    temp = []
    time = []
    with open(file) as f:
        for line in f:
            if line.strip() == "":
                continue
            if "Time" in line:
                continue

            t, _, ep, ek, T = (float(l) for l in line.split())
            time.append(t)
            temp.append(T)
            e_kin.append(ek)
            e_pot.append(ep)

    return e_kin, e_pot, temp, time


def md_sum(file, plot, avg, verbose):
    """summarize the MD trajectory in FILE"""
    infile = Path(file)
    msg = "\nConsider to use `vibes output md` to create `trajectory.nc` "
    msg += "and plot via `vibes info md trajectory.nc`"
    warn(msg, FutureWarning, stacklevel=2)

    natoms = -1
    if infile.suffix in (".yaml", ".son", ".bz2", ".gz"):
        trajectory = reader(infile)
        e_kin = trajectory.kinetic_energy
        e_pot = trajectory.potential_energy
        temp = trajectory.temperatures
        time = trajectory.times / 1000
        natoms = len(trajectory[0])

    elif "log" in infile.suffix:
        e_kin, e_pot, temp, time = parse_log(infile)

    if verbose:
        n_range = min(20, len(trajectory) - 2)
        print(f"step   time   temperature")
        for ii in range(n_range):
            print(
                f'{trajectory[-n_range+ii].info["nsteps"]:5d} '
                f"{time[-n_range+ii]:10.4f} "
                f"{temp[-n_range+ii]:10.4f} "
            )

    # divide into three thirds
    len_3 = len(temp) // 3
    temp_1 = temp[:len_3]
    temp_2 = temp[len_3 : 2 * len_3]
    temp_3 = temp[2 * len_3 :]

    print(f"Simulation time:         {time[-1] - time[0]:.4f} ps ({len(time)} steps)")
    print(f"Temperature:             {np.mean(temp):.2f} +/- {np.std(temp):.2f}K")
    print(f"Temperature (1st 1/3):   {np.mean(temp_1):.2f} +/- {np.std(temp_1):.2f}K")
    print(f"Temperature (2nd 1/3):   {np.mean(temp_2):.2f} +/- {np.std(temp_2):.2f}K")
    print(f"Temperature (3rd 1/3):   {np.mean(temp_3):.2f} +/- {np.std(temp_3):.2f}K")
    print(f"Kinetic energy:          {np.mean(e_kin):.2f} +/- {np.std(e_kin):.2f}eV")
    print(f"Potential energy:        {np.mean(e_pot):.2f} +/- {np.std(e_pot):.2f}eV")

    if plot:
        # create pandas DataFrame and plot things from it
        import pandas as pd

        data = {"temp": temp, "e_kin": e_kin, "e_pot": e_pot}
        df = pd.DataFrame(data, index=time)
        plot_summary(df, avg, natoms)


def main():
    """ main routine """
    parser = ArgumentParser(description="Read md.log and make simple statistics")
    parser.add_argument("file", help="md.log or trajectory.son input file")
    parser.add_argument("-p", "--plot", action="store_true", help="plot to pdf")
    parser.add_argument("--avg", type=int, help="running avg in plot", default=100)
    parser.add_argument("-v", "--verbose", action="store_true", help="give more info")
    args = parser.parse_args()

    md_sum(args.file, args.plot, args.avg, args.verbose)


if __name__ == "__main__":
    main()


def plot_summary(dataframe, avg, natoms=None):
    """plot a summary of the data in DATAFRAGE

    Args:
        dataframe (pandas.Dataframe): MD data
        avg (int): window size for averaging
        natoms (int): number of atoms
    """
    import matplotlib

    matplotlib.use("pdf")

    from matplotlib import pyplot as plt

    from vibes.helpers.plotting import tableau_colors as tc

    # plot temperatures
    temp = dataframe.temp
    e_kin = dataframe.e_kin
    e_pot = dataframe.e_pot
    e_pot -= e_pot.min()

    fig, (ax, ax2) = plt.subplots(ncols=2)

    # settings for the immediate plot
    plot_settings = {"alpha": 0.4, "linewidth": 1.0, "label": ""}
    avg_settings = {"linewidth": 1.5}

    temp.plot(color=tc[0], title="Nuclear Temperature", ax=ax, **plot_settings)

    if natoms:
        e_temp = (e_kin + e_pot) / natoms / 3 / units.kB
        e_temp.plot(color=tc[1], ax=ax, **plot_settings)
        e_temp.rolling(window=avg, min_periods=0).mean().plot(
            color=tc[1], label=f"E_tot", ax=ax, **avg_settings
        )

    roll = temp.rolling(window=avg, min_periods=0).mean()
    roll.plot(color=tc[0], label=f"T_nucl", ax=ax, **avg_settings)

    # exp = data.iloc[min(len(data) // 2, args.avg) :].expanding().mean()
    # exp.plot( color=tc[5], label=f"Expanding mean ({args.avg}", ax=ax)

    ax.set_xlabel("Time [ps]")
    ax.set_ylabel("Nucl. Temperature [K]")
    ax.legend()

    # fig.savefig("temp.pdf")

    # plot energies in one plot
    # fig, ax = plt.subplots()

    e_tot = e_pot + e_kin
    e_dif = e_pot - e_kin

    e_tot.plot(color=tc[0], title="Total Energy", ax=ax2, **plot_settings)
    roll = e_tot.rolling(window=avg, min_periods=0).mean()
    roll.plot(color=tc[0], ax=ax2, label="E_tot", **avg_settings)

    e_pot.plot(color=tc[3], ax=ax2, **plot_settings)
    roll = e_pot.rolling(window=avg, min_periods=0).mean()
    roll.plot(color=tc[3], ax=ax2, label="E_pot", **avg_settings)

    ax2.axhline(0, linewidth=1, color="k")
    e_dif.plot(color=tc[1], ax=ax2, **plot_settings)
    exp = e_dif.rolling(min_periods=0, window=avg).mean()
    exp.plot(color=tc[1], ax=ax2, label="E_pot - E_kin", **avg_settings)

    ax2.legend()
    ax2.set_xlabel("Time [ps]")
    ax2.set_ylabel("Energy [eV]")

    # fig.tight_layout()
    file = "md_summary.pdf"
    fig.savefig(file)
    print(f".. summary plotted to {file}")
