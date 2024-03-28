"""Fourier Transforms"""
import numpy as np
import xarray as xr

from vibes import dimensions as dims
from vibes import keys
from vibes.helpers import Timer, talk, warn
from vibes.konstanten.einheiten import THz_to_cm


_prefix = "Fourier"


def _talk(msg, **kwargs):
    return talk(msg, prefix=_prefix, **kwargs)


def get_timestep(times, tol=1e-9):
    """get time step from a time series and check for glitches"""
    d_times = np.asarray((times - np.roll(times, 1))[1:])
    timestep = np.mean(d_times)

    assert np.all(np.abs(d_times - timestep) < tol)

    return timestep


def get_frequencies(
    N=None, dt=None, times=None, fs_factor=1, to_cm=False, verbose=False
):
    """compute the frequency domain in THz for signal with fs resolution

    Args:
        N (int): Number of time steps
        dt (float): time step in fs
        times (ndarray): time series in fs (or converted to fs via `fs_factor`)
        fs_factor (float): convert timestep to fs by `dt / fs_factor` (for ps: 1000)
        to_cm (bool): return in inverse cm instead
        verbose (bool): be informative

    Returns:
        frequencies in THz (ndarray)
    """
    if N and dt:
        dt = dt / fs_factor
    elif times is not None:
        N = len(times)
        dt = get_timestep(times) / fs_factor

    # Frequencies in PetaHz
    max_freq = 1.0 / (2.0 * dt)
    w = np.linspace(0.0, max_freq, N // 2)
    # Frequencies in THz
    w *= 1000
    dw = w[1] - w[0]

    if verbose:
        msg = [f".. get frequencies for time series"]
        msg += [f".. timestep:               {np.asarray(dt)} fs"]
        msg += [f"-> maximum frequency:      {np.max(w):.5} THz"]
        msg += [f".. Number of steps:        {N}"]
        msg += [f"-> frequency resolution:   {dw:.5f} THz"]
        _talk(msg)

    if to_cm:
        w *= THz_to_cm

    return w


def get_fft(series):
    """run `np.fft.fft` on time series

    https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.hfft.html

    Args:
        series (np.ndarray [N_t, ...]): time series, first dimension is the time axis

    Returns:
        np.ndarray ([N_t, ...]): Fourier transform of series ([: N_t // 2])
    """

    velocities = np.asarray(series).copy()

    N = series.shape[0]

    velocities = velocities.swapaxes(-1, 0)
    velocities = np.fft.fft(velocities, axis=-1)

    return velocities.swapaxes(-1, 0)[: N // 2]


def zero_pad_array(
    array: xr.DataArray, times: np.ndarray, npad: int = 10000, verbose: bool = True
):
    """zero pad time axis of array

    Args:
        array ([N_t, ...]): the array which is supposed to be zero padded
        times: the time axis
        npad: pad with this many zeros
    Returns:
        DataArray ([N_t, ...]): the zero padded array including the time axis

    """
    data = array.data

    if npad > 0:
        _talk(f".. use zero padding with {npad} zeros", verbose=verbose)
        # use linear_ramp for time, build new DataArray
        # only pad time (assume 1. dimension)
        pad_width = [(0, npad)]
        for _ in array.shape[1:]:
            pad_width.append((0, 0))

        data = np.pad(data, pad_width)

        # pad time using linear_ramp
        t = times
        dt = t[-1] - t[-2]
        times = np.pad(t, (0, npad), mode="linear_ramp", end_values=t[-1] + npad * dt)

    coords = {keys.time: times}
    dims = (keys.time, *array.dims[1:])

    return xr.DataArray(data, dims=dims, coords=coords)


def get_fourier_transformed(array: xr.DataArray, npad: int = 0, verbose: bool = True):
    """Perform Fourier Transformation of Series/DataArray

    Args:
        array ([N_t, ...]): xarray.DataArray with `time` axis in fs
        npad: Use zero padding with this many zeros
        verbose: be verbose
    Return:
        DataArray ([N_t, ...]): FT(series) with `omega` axis in THz
    """
    timer = Timer("Compute FFT", verbose=verbose, prefix=_prefix)

    assert isinstance(array, xr.DataArray), type(array)

    # figure out the time axis
    if hasattr(array, keys.time):  # as it is supposed to be
        times = np.asarray(array[keys.time])
    elif hasattr(array, "index"):  # compatibility with pd.Series
        times = np.asarray(array.index)
    else:  # maybe this used to be a numpy array
        warn(f"time coordinate not found, use `coords=arange`", level=0)
        times = np.arange(len(array))

    array = zero_pad_array(array, times=times, npad=npad, verbose=verbose)

    # perform FFT
    fft = get_fft(array)

    # get frequencies from time axis
    omegas = get_frequencies(times=array[keys.time], verbose=verbose)

    da = xr.DataArray(
        fft,
        dims=(keys.omega, *array.dims[1:]),
        coords={keys.omega: omegas},
        name=keys._join(array.name, keys.fourier_transform),
    )
    result = da

    timer()

    return result


def get_power_spectrum(
    flux: xr.DataArray = None, prefactor: float = 1.0, verbose: bool = True
) -> xr.DataArray:
    """compute power spectrum for given flux

    Args:
        flux (xr.DataArray, optional): flux [Nt, 3]. Defaults to xr.DataArray.
        prefactor (float, optional): prefactor
        verbose (bool, optional): be verbose

    Returns:
        xr.DataArray: heat_flux_power_spectrum
    """
    kw = {"verbose": verbose}
    timer = Timer("Get power spectrum from flux", **kw)

    Jw = get_fourier_transformed(flux.dropna(dims.time), **kw)

    Jspec = abs(Jw)

    timer()

    return Jspec
