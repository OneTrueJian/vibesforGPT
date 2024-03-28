import numpy as np
import xarray as xr


def xtrace(array: xr.DataArray, axis1: int = -2, axis2: int = -1) -> xr.DataArray:
    """apply `np.trace` to xarray.DataArray

    Args:
        array (xr.DataArray): [description]
        axis1 (int, optional): [description]. Defaults to -2.
        axis2 (int, optional): [description]. Defaults to -1.

    Returns:
        xr.DataArray: DataArray with trace applied
    """
    if axis1 < 0:
        axis1 = len(array.shape) + axis1
    if axis2 < 0:
        axis2 = len(array.shape) + axis2

    data = np.trace(array, axis1=axis1, axis2=axis2)

    # compile new dimensions
    new_dims = list(array.dims)
    new_dims.pop(max(axis1, axis2))
    new_dims.pop(min(axis1, axis2))

    da = xr.DataArray(data, dims=new_dims, coords=array.coords)

    return da
