# USAGE:  ./get_relaxation_info.py  aims.out (aims.out.2 ...)
#
# Revision 2018/08: FK
# 23/4/2019: give max. force in meV/AA instead of eV/AA
# 12/6/2019: refactor for vibes cli
# 27/6/2019: Add docstrings


# Find the optimizer type
def get_optimizer(f):
    """Find the optimzer type

    Parameters
    ----------
    f: str
        file to search through

    Returns
    -------
    int
        Optimizer type, 1 for Textbook BFGS, 2 for TRM, -1 for undefined
    """
    try:
        line = next(l for l in f if "Geometry relaxation:" in l)
    except StopIteration:
        exit("Optimizer not found -- is this output from a relaxation?")

    if "Textbook BFGS" in line:
        return 1
    if "TRM" in line:
        return 2
    return -1


# find energy
def get_energy(f):
    """Find the total energy for the calculation

    Parameters
    ----------
    f: str
        file to search through

    Returns
    -------
    total_energy: float
        the total energy corrected of the structure
    free_energy: float
        the electronic free energy of the structure
    """
    spacegroup = None
    total_energy = None
    for line in f:
        if "Space group" in line:
            spacegroup = int(line.split()[4])
        if "| Total energy corrected        :" in line:
            total_energy = float(line.split()[5])
            break
        if "| Electronic free energy        :" in line:
            free_energy = float(line.split()[5])

    if not total_energy:
        raise StopIteration

    return total_energy, free_energy, spacegroup


# get max_force
def get_forces(f):
    """Find the maximum force component

    Parameters
    ----------
    f: str
        file to search through

    Returns
    -------
    float
        The maximum force component of the structure
    """
    line = next(l for l in f if "Maximum force component" in l)
    return float(line.split()[4])


# get current volume
def get_volume(f):
    """Find the volume of the structure

    Parameters
    ----------
    f: str
        file to search through

    Returns
    -------
    float
        The structures volume
    """
    for line in f:
        if "| Unit cell volume " in line:
            return float(line.split()[5])
        if "Begin self-consistency loop:" in line:
            return -1
        if "Final output of selected total energy values:" in line:
            return -1
    return -1


# parse info of one step
def parser(f, n_init=0, optimizer=2):
    """Parse info of one step

    Parameters
    ----------
    f: str
        file to search through
    n_init: int
        The initial step
    optimizer: int
        Optimizer type, 1 for Textbook BFGS, 2 for TRM, -1 for undefined

    Yields
    ------
    n_rel: int
        Current relaxation step
    energy: float
        The total energy corrected of the step
    free_energy: float
        The electronic free energy of the step
    max_force: float
        The maximum force component of the step
    volume: float
        The volume of the step
    status: int
        status of the step, 0 is normal, 1 is unproductive step, 2 is optimizer is stuck
    converged: bool
        If True the relaxation is converged
    abort: int
        If 1 the relaxation is aborting
    """
    n_rel = n_init
    converged = 0
    abort = 0
    volume = -1
    while not converged and not abort:
        n_rel += 1
        status = 0
        try:
            energy, free_energy, spacegroup = get_energy(f)
            max_force = get_forces(f)
        except StopIteration:
            break

        for line in f:
            if "Present geometry is converged." in line:
                converged = 1
                break
            elif "Advancing" in line:
                pass
            elif "Aborting optimization" in line:
                abort = 1
            elif "Counterproductive step -> revert!" in line:
                status = 1
            elif "Optimizer is stuck" in line:
                status = 2
            #            elif '**' in line:
            #                status = 3
            elif "Finished advancing geometry" in line:
                volume = get_volume(f)
                break
            elif "Updated atomic structure" in line:
                volume = get_volume(f)
                break
        yield (
            n_rel,
            energy,
            free_energy,
            max_force,
            volume,
            spacegroup,
            status,
            converged,
            abort,
        )


def print_status(
    n_rel, energy, de, free_energy, df, max_force, volume, spacegroup, status_string
):
    """Print the status line, skip volume if not found

    Parameters
    ----------
    n_rel: int
        Current relaxation step
    energy: float
        The total energy corrected of the step
    de: float
        Change in total energy
    free_energy: float
        The electronic free energy of the step
    df: float
        Change in electronic free energy
    max_force: float
        The maximum force component of the step
    volume: float
        The volume of the step
    status_string: str
        The status of the relaxation
    """

    if volume and volume > 0:
        vol_str = f"{volume:15.4f}"
    else:
        vol_str = ""

    if spacegroup:
        sg_str = f"{spacegroup:5d}"
    else:
        sg_str = ""

    print(
        "{:5d}   {:16.8f}   {:16.8f} {:14.6f} {:20.6f} {} {} {}".format(
            n_rel,
            energy,
            free_energy,
            df,
            max_force * 1000,
            vol_str,
            status_string,
            sg_str,
        )
    )


def get_relaxation_info(files):
    """print information about relaxation performed with FHIaims

    Parameters
    ----------
    files: list of str
        The file paths of the aims.out files to analyze
    """
    init, n_rel, converged, abort = 4 * (None,)
    status_string = [
        "",
        "rejected.",
        "rejected: force <-> energy inconsistency?",
        "stuck.",
    ]

    # Run
    print(
        "\n# Step Total energy [eV]   Free energy [eV]   F-F(1)"
        + " [meV]   max. force [meV/AA]  Volume [AA^3]  Spacegroup\n"
    )

    converged, abort = False, False
    for infile in files:
        with open(infile) as f:
            # Check optimizer
            optimizer = get_optimizer(f)
            ps = parser(f, n_init=n_rel or 0, optimizer=optimizer)
            for (n_rel, ener, free_ener, fmax, vol, sg, status, _conv, _abort) in ps:
                if not init:
                    first_energy, first_free_energy = ener, free_ener
                    init = 1
                print_status(
                    n_rel,
                    ener,
                    1000 * (ener - first_energy),
                    free_ener,
                    1000 * (free_ener - first_free_energy),
                    fmax,
                    vol,
                    sg,
                    status_string[status],
                )
                converged, abort = _conv, _abort

    if converged:
        print("--> converged.")
    if abort:
        print("*--> aborted, too many steps.")


def main():
    """wrap get_relaxation_info"""
    from argparse import ArgumentParser as argpars

    parser = argpars(description="Summarize the relaxation path")
    parser.add_argument("aimsouts", type=str, nargs="+", help="aims output files")
    args = parser.parse_args()

    get_relaxation_info(args.aimsouts)


if __name__ == "__main__":
    main()
