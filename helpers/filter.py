import numpy as np
import xarray as xr
from scipy import signal as sl

from vibes import keys
from vibes.helpers import talk, warn


_prefix = "filter"


def _talk(msg, **kw):
    """wrapper for `utils.talk` with prefix"""
    return talk(msg, prefix=_prefix, **kw)


def get_filtered(
    array: xr.DataArray,
    window_fs: float = None,
    window: int = None,
    antisymmetric: bool = False,
    polyorder: int = 1,
    verbose: bool = True,
) -> xr.DataArray:
    """apply Savitzky-Golay filter to array to remove noise

    See: https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter

    Args:
        array [N_t, ...]: the array to be filtered with time axis in the front
        window_fs: the filter window in fs (we assume array has time axis given in fs)
        window: the filter window in time steps
        antisymmetric: use antisymmetric boundary condition to correctly interpolate t=0
        polyorder: order used for filter (default: 1)
        verbose: be verbose

    Returns:
        xr.DataArray [N_t, ...]: array with filter applied to time axis

    """
    # get the window from time axis:
    if window_fs is not None:
        time = array[keys.time]
        window = len(time[time < window_fs])

    if window is None:
        warn("Either `window_fs` or `window` have to be specified.", level=2)

    # make sure window is odd
    window = window // 2 * 2 + 1

    # if antisymmetric, use f(-t) = -f(t) to extend data
    if antisymmetric:
        data = np.concatenate((-array[::-1], array))
    else:
        data = np.asarray(array)

    # prepare array for the filtered data
    data_filtered = np.zeros_like(data)

    # move time axis to back
    data = np.moveaxis(data, 0, -1)
    data_filtered = np.moveaxis(data_filtered, 0, -1)

    # filter the data
    kw = {"window_length": window, "polyorder": polyorder}
    _talk(f"Apply Savitzky-Golay filter with {kw}", verbose=verbose)
    for ij in np.ndindex(data.shape[:-1]):
        data_filtered[ij] = sl.savgol_filter(data[ij], **kw)

    # move time axis back to front
    data = np.moveaxis(data, -1, 0)
    data_filtered = np.moveaxis(data_filtered, -1, 0)

    # create DataArray with filtered data
    new_array = array.copy()
    if antisymmetric:
        new_array.data = data_filtered[len(array) :]
    else:
        new_array.data = data_filtered

    return new_array
