"""Parsers converting OGC EDR payload responses into in-memory xarray.Dataset instances."""

from logging import getLogger
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from knmidata._exceptions import KNMIError

logger = getLogger(__name__)

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


def parse_coverage(
    payload: dict[str, Any], shared_parameters: dict[str, Any] | None = None
) -> xr.Dataset:
    """Parse a single OGC CoverageJSON coverage into an xarray.Dataset.

    Dimension order follows the order of the axes in ``domain.axes``, matching
    the flattened order of the range values.

    Parameters
    ----------
    payload : dict[str, Any]
        A single CoverageJSON coverage object.
    shared_parameters : dict[str, Any] | None, default=None
        Parameter metadata from the enclosing CoverageCollection, used when the
        coverage itself has no "parameters" member (KNMI puts them at the
        collection level and only "ranges" in each coverage).

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
        coord = pd.to_datetime(values) if key == "t" else np.array(values)
        # keep axis metadata (label, unit) as coordinate attributes
        coord_attrs = {
            k: v for k, v in axis.items() if k not in ("values", "start", "stop") and v
        }
        coords[name] = xr.Variable(dims=[name], data=coord, attrs=coord_attrs)
        dims.append(name)

    shape = tuple(len(coords[d]) for d in dims)
    data_vars: dict[str, Any] = {}
    parameters = payload.get("parameters") or shared_parameters or {}
    ranges = payload.get("ranges", {})
    if not parameters and ranges:
        # no parameter metadata: still expose the ranges as data variables
        parameters = {range_id: {} for range_id in ranges}

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
        unit = param_meta.get("unit", {})
        observed_property = param_meta.get("observedProperty", {})
        measurement_type = param_meta.get("measurementType", {})
        attributes = {
            "units": unit.get("symbol", ""),
            "long_name": param_meta.get("description", {}).get("en", param_id),
        }
        if unit.get("label"):
            attributes["unit_label"] = unit["label"]
        if observed_property.get("label"):
            attributes["observed_property"] = observed_property["label"]
        if measurement_type:
            attributes["measurement_type"] = measurement_type
        data_vars[param_id] = (dims, array_data, attributes)

    dataset = xr.Dataset(data_vars=data_vars, coords=coords)

    # coverage-level metadata
    for key in ("id", "title", "description"):
        if payload.get(key):
            dataset.attrs[f"coverage_{key}"] = payload[key]
    if referencing := domain.get("referencing"):
        dataset.attrs["crs"] = referencing[0].get("id", "EPSG:4326")
        dataset.attrs["referencing"] = referencing

    return dataset


def _coverage_station_label(coverage: dict[str, Any]) -> str | None:
    """Best-effort station label for one coverage of a CoverageCollection.

    KNMI includes the station WSI in CoverageJSON as ``eumetnet:locationId``
    (older responses used ``inspiregloss:Identifier``), usually in the
    coverage's ``domain.referencing`` entries or as the coverage ``id``. Falls
    back to the grid origin for 1x1 station grids, else None.
    """
    if coverage.get("id"):
        return str(coverage["id"])
    domain = coverage.get("domain", {})
    for ref in domain.get("referencing", []) or []:
        if not isinstance(ref, dict):
            continue
        for key in (
            "eumetnet:locationId",
            "inspiregloss:Identifier",
            "locationId",
            "station",
        ):
            if ref.get(key):
                return str(ref[key])
    axes = domain.get("axes", {})
    if "x" in axes and "y" in axes:
        x_values = axes["x"].get("values", [])
        y_values = axes["y"].get("values", [])
        if x_values and y_values:
            return f"{x_values[0]},{y_values[0]}"
    return None


def parse_coveragejson(payload: dict[str, Any]) -> xr.Dataset:
    """Parse an OGC CoverageJSON payload into an xarray.Dataset.

    Handles both single coverages and CoverageCollections (as returned by
    /area, /cube, or multi-location queries on station collections). Coverages
    are concatenated along the ``station`` dimension, labelled with the station
    id (WSI) when the response carries it, else the grid origin, else a
    ``coverage`` index.

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
        # KNMI puts "parameters" at the collection level; each coverage only
        # carries "ranges".
        shared_parameters = payload.get("parameters", {})
        raw_coverages = payload.get("coverages", [])
        coverages = [parse_coverage(c, shared_parameters) for c in raw_coverages]
        if not coverages:
            return xr.Dataset()
        if all("station" in ds.dims for ds in coverages):
            return xr.concat(coverages, dim="station")
        labels = [_coverage_station_label(c) for c in raw_coverages]
        if any(label is not None for label in labels):
            # Per-station 1x1 grid coverages: squeeze the singleton x/y dims
            # (their values differ per station) and re-attach them as lon/lat
            # coordinates along the station dimension.
            lons: list[Any] = []
            lats: list[Any] = []
            squeezed = []
            for raw, ds in zip(raw_coverages, coverages, strict=True):
                axes = raw.get("domain", {}).get("axes", {})
                x_values = axes.get("x", {}).get("values", [])
                y_values = axes.get("y", {}).get("values", [])
                lons.append(x_values[0] if x_values else None)
                lats.append(y_values[0] if y_values else None)
                for d in ("x", "y"):
                    if d in ds.dims and ds.sizes[d] == 1:
                        ds = ds.squeeze(d, drop=True)
                squeezed.append(ds)
            dim = xr.DataArray(
                [label if label is not None else i for i, label in enumerate(labels)],
                dims="station",
                name="station",
            )
            out = xr.concat(squeezed, dim=dim)
            if any(v is not None for v in lons):
                out = out.assign_coords(lon=("station", lons), lat=("station", lats))
            logger.debug(
                "Merged %d coverages along 'station' (labels: %s)",
                len(coverages),
                labels[:5],
            )
            return out
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
    feature_ids: list[Any] = []
    geom_types: list[str] = []
    for feature in features:
        props = dict(feature.get("properties", {}))
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates") or [None, None]
        props.setdefault("lon", coordinates[0] if len(coordinates) > 0 else None)
        props.setdefault("lat", coordinates[1] if len(coordinates) > 1 else None)
        rows.append(props)
        # keep the feature id and geometry type as metadata, not data vars
        feature_ids.append(feature.get("id"))
        geom_types.append(str(geometry.get("type")))

    if not rows:
        return xr.Dataset()

    frame = pd.DataFrame(rows)
    coords: dict[str, Any] = {"feature": np.arange(len(frame))}
    if any(v is not None for v in feature_ids):
        coords["id"] = ("feature", np.array(feature_ids, dtype=object))
    if "label" in frame:
        coords["label"] = ("feature", frame["label"].astype(str).to_numpy())
    if "lon" in frame:
        coords["lon"] = ("feature", frame["lon"].to_numpy())
    if "lat" in frame:
        coords["lat"] = ("feature", frame["lat"].to_numpy())
    if "datetime" in frame:
        coords["time"] = ("feature", pd.to_datetime(frame["datetime"]).to_numpy())

    data_vars: dict[str, Any] = {}
    # structural EDR properties, not measurements
    skip = {
        "lon",
        "lat",
        "datetime",
        "geometry",
        "label",
        "edrqueryendpoint",
        "parameter-name",
    }
    for column in frame.columns:
        if column in skip:
            continue
        values = frame[column].to_numpy()
        if pd.api.types.is_numeric_dtype(frame[column]):
            data_vars[column] = (("feature",), _to_float_array(list(values)))
        else:
            data_vars[column] = (("feature",), values.astype(str))

    dataset = xr.Dataset(data_vars=data_vars, coords=coords)
    if geom_types:
        dataset.attrs["geometry_type"] = geom_types[0]
    if payload.get("bbox"):
        dataset.attrs["bbox"] = payload["bbox"]
    logger.debug(
        "Parsed GeoJSON: %d features, %d data variables", len(rows), len(data_vars)
    )
    return dataset
