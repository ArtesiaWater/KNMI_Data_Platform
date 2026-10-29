"""Parsers converting OGC EDR payload responses into in-memory xarray.Dataset instances."""

import io
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


def parse_netcdf_bytes(content: bytes) -> xr.Dataset:
    """Parse raw in-memory NetCDF binary bytes into an xarray.Dataset.

    Parameters
    ----------
    content : bytes
        Raw response bytes returned from an EDR endpoint with f=NetCDF.

    Returns
    -------
    xr.Dataset
        In-memory xarray dataset.
    """
    buffer = io.BytesIO(content)
    # h5netcdf reads directly from in-memory file-like buffer objects
    return xr.open_dataset(buffer, engine="h5netcdf")


def parse_coveragejson(payload: dict[str, Any]) -> xr.Dataset:
    """Parse an OGC CoverageJSON dictionary structure into an xarray.Dataset.

    Parameters
    ----------
    payload : dict[str, Any]
        Parsed JSON response from an EDR query.

    Returns
    -------
    xr.Dataset
        CF-compliant multidimensional dataset.
    """
    domain = payload.get("domain", {})
    axes = domain.get("axes", {})

    coords: dict[str, Any] = {}
    dims: list[str] = []

    # 1. Temporal dimension
    if "t" in axes:
        coords["time"] = pd.to_datetime(axes["t"].get("values", []))
        dims.append("time")

    # 2. Spatial / Station dimensions
    if "composite" in axes:
        coords["station"] = np.array(axes["composite"].get("values", []))
        dims.append("station")
    else:
        for spatial_axis in ["z", "y", "x"]:
            if spatial_axis in axes:
                coords[spatial_axis] = np.array(axes[spatial_axis].get("values", []))
                dims.append(spatial_axis)

    # 3. Data variables
    data_vars: dict[str, Any] = {}
    parameters = payload.get("parameters", {})
    ranges = payload.get("ranges", {})

    target_shape = tuple(len(coords[d]) for d in dims) if dims else (1,)

    for param_id, param_meta in parameters.items():
        if param_id not in ranges:
            continue

        raw_values = ranges[param_id].get("values", [])
        array_data = np.array(raw_values).reshape(target_shape)

        attributes = {
            "units": param_meta.get("unit", {}).get("symbol", ""),
            "long_name": param_meta.get("description", {}).get("en", param_id),
        }
        data_vars[param_id] = (dims, array_data, attributes)

    dataset = xr.Dataset(data_vars=data_vars, coords=coords)

    # 4. CRS metadata
    if referencing := domain.get("referencing"):
        dataset.attrs["crs"] = referencing[0].get("id", "EPSG:4326")

    return dataset
