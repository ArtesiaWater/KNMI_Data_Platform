"""Parsers converting OGC EDR payload responses into in-memory xarray.Dataset instances."""

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from knmidata._exceptions import KNMIError

_AXIS_NAMES = {"t": "time", "composite": "station"}


def _axis_name(key: str) -> str:
    """Map a CoverageJSON domain axis key to a dimension name."""
    return _AXIS_NAMES.get(key, key)


def _to_float_array(values: list[Any]) -> np.ndarray:
    """Convert range values to a float array, mapping nulls to NaN."""
    try:
        # float(None) is not allowed, but numpy maps None to NaN for float dtypes
        return np.array(values, dtype="float64")
    except (TypeError, ValueError):
        return np.array(values, dtype=object)


def parse_coverage(payload: dict[str, Any]) -> xr.Dataset:
    """Parse a single OGC CoverageJSON coverage into an xarray.Dataset.

    Dimension order follows the order of the axes in ``domain.axes``, matching
    the flattened order of the range values.

    Parameters
    ----------
    payload : dict[str, Any]
        A single CoverageJSON coverage object.

    Returns
    -------
    xr.Dataset
        Multidimensional dataset.
    """
    domain = payload.get("domain", {})
    axes = domain.get("axes", {})

    coords: dict[str, Any] = {}
    dims: list[str] = []
    for key, axis in axes.items():
        name = _axis_name(key)
        values = axis.get("values", [])
        coords[name] = pd.to_datetime(values) if key == "t" else np.array(values)
        dims.append(name)

    shape = tuple(len(coords[d]) for d in dims)
    data_vars: dict[str, Any] = {}
    parameters = payload.get("parameters", {})
    ranges = payload.get("ranges", {})

    for param_id, param_meta in parameters.items():
        if param_id not in ranges:
            continue
        raw_values = ranges[param_id].get("values", [])
        array_data = _to_float_array(raw_values)
        expected = int(np.prod(shape)) if shape else 1
        if array_data.size != expected:
            raise KNMIError(
                f"Range '{param_id}' has {array_data.size} values, expected {expected} "
                f"for domain axes {dims}."
            )
        array_data = array_data.reshape(shape) if shape else array_data.reshape(())
        attributes = {
            "units": param_meta.get("unit", {}).get("symbol", ""),
            "long_name": param_meta.get("description", {}).get("en", param_id),
        }
        data_vars[param_id] = (dims, array_data, attributes)

    dataset = xr.Dataset(data_vars=data_vars, coords=coords)

    if referencing := domain.get("referencing"):
        dataset.attrs["crs"] = referencing[0].get("id", "EPSG:4326")

    return dataset


def parse_coveragejson(payload: dict[str, Any]) -> xr.Dataset:
    """Parse an OGC CoverageJSON payload into an xarray.Dataset.

    Handles both single coverages and CoverageCollections (as returned by the
    /cube endpoint on observation collections); coverages in a collection are
    concatenated along the station dimension when present.

    Parameters
    ----------
    payload : dict[str, Any]
        Parsed JSON response from an EDR data query with f=CoverageJSON.

    Returns
    -------
    xr.Dataset
        Multidimensional dataset.
    """
    if payload.get("type") == "CoverageCollection" or "coverages" in payload:
        coverages = [parse_coverage(c) for c in payload.get("coverages", [])]
        if not coverages:
            return xr.Dataset()
        if all("station" in ds.dims for ds in coverages):
            return xr.concat(coverages, dim="station")
        return xr.concat(
            coverages,
            dim=xr.DataArray(
                np.arange(len(coverages)), dims="coverage", name="coverage"
            ),
        )
    return parse_coverage(payload)


def parse_geojson(payload: dict[str, Any]) -> xr.Dataset:
    """Parse a GeoJSON FeatureCollection into an xarray.Dataset.

    Used for EDR responses with f=GeoJSON (e.g. /locations or point queries on
    station collections). The dataset gets a 'feature' dimension, lon/lat/time
    coordinates when present, and one data variable per numeric property.

    Parameters
    ----------
    payload : dict[str, Any]
        Parsed JSON response from an EDR query with f=GeoJSON.

    Returns
    -------
    xr.Dataset
        Dataset with one entry per feature.
    """
    features = payload.get("features", [])
    rows: list[dict[str, Any]] = []
    for feature in features:
        props = dict(feature.get("properties", {}))
        coordinates = (feature.get("geometry") or {}).get("coordinates") or [None, None]
        props.setdefault("lon", coordinates[0] if len(coordinates) > 0 else None)
        props.setdefault("lat", coordinates[1] if len(coordinates) > 1 else None)
        rows.append(props)

    if not rows:
        return xr.Dataset()

    frame = pd.DataFrame(rows)
    coords: dict[str, Any] = {"feature": np.arange(len(frame))}
    if "lon" in frame:
        coords["lon"] = ("feature", frame["lon"].to_numpy())
    if "lat" in frame:
        coords["lat"] = ("feature", frame["lat"].to_numpy())
    if "datetime" in frame:
        coords["time"] = ("feature", pd.to_datetime(frame["datetime"]).to_numpy())

    data_vars: dict[str, Any] = {}
    skip = {"lon", "lat", "datetime", "geometry"}
    for column in frame.columns:
        if column in skip:
            continue
        values = frame[column].to_numpy()
        if pd.api.types.is_numeric_dtype(frame[column]):
            data_vars[column] = (("feature",), _to_float_array(list(values)))
        else:
            data_vars[column] = (("feature",), values.astype(str))

    return xr.Dataset(data_vars=data_vars, coords=coords)
