"""compute and analyze heat fluxes"""
import pandas as pd
import scipy.signal as sl

from vibes import defaults
from vibes.correlation import get_autocorrelationNd
from vibes.fourier import get_fourier_transformed
from vibes.integrate import trapz

from . import Timer, _talk


def get_velocity_autocorrelation(velocities=None, trajectory=None, verbose=True):
    """LEGACY: compute velocity autocorrelation function from xarray"""
    return get_autocorrelationNd(velocities, normalize=True, hann=False)


def get_vdos(velocities=None, hann=False, normalize=False, npad=10000, verbose=True):
    r"""compute vibrational DOS for trajectory

    vdos(w) = FT{\sum_i corr(v_i, v_i)(t)}(w)

    Args:
        velocities (xarray.DataArray [N_t, N_a, 3]): the velocities
        hann: use Hann window when computing the autocorrelation
        normalize: normalize VDOS to 1
        npad: number of zeros for zero padding
    Returns:
        vdos (xarray.DataArray [N_t, N_a, 3])
    """
    timer = Timer("Get VDOS", verbose=verbose)
    v_corr = get_autocorrelationNd(velocities, normalize=True, hann=hann)
    df_vdos = get_fourier_transformed(v_corr, npad=npad)

    if normalize:
        norm = trapz(df_vdos)
        _talk(f"Normalize with {norm}")
        df_vdos /= norm

    timer()

    return df_vdos


def simple_plot(
    series: pd.Series,
    file: str = "vdos.pdf",
    prominence: float = defaults.filter_prominence,
    threshold_freq: float = 0.1,
    max_frequency: float = None,
    logy: bool = False,
):
    """simple plot of VDOS for overview purpose

    Args:
        series: Intensity vs. omega
        file: file to store the plot to
        prominence: for peak detection with `scipy.signal.find_peaks`
        threshold:_freq: neglect data up to this freq in THz (default: 0.1 THz)
        max_frequency (float): max. frequency in THz
        logy (bool): use semilogy
    """
    # normalize peaks
    series -= series.min()
    series /= series[series.index > threshold_freq].max()

    # find peaks:
    peaks, *_ = sl.find_peaks(series, prominence=prominence)

    _talk(f".. {len(peaks)} peaks found w/ prominence={prominence}")

    high_freq = series.index[series > 0.05].max()
    _talk(f".. highest non-vanishin freq. found at   {high_freq:.2f} THz")

    ax = series.plot()

    # plot peaks
    ax.scatter(series.index[peaks], series.iloc[peaks], marker=".", c="k", zorder=5)

    if logy:
        ax.set_yscale("log")

    if max_frequency is None:
        max_frequency = 1.2 * high_freq

    ax.set_xlim([0, max_frequency])
    ax.set_xlabel("Omega (THz)")
    ax.set_ylabel("VDOS (1)")

    fig = ax.get_figure()
    fig.savefig(file, bbox_inches="tight")

    _talk(f".. max. frequency for plot:  {high_freq:.2f} THz")
    _talk(f".. plot saved to {file}")
