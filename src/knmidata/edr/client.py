"""OGC Environmental Data Retrieval (EDR) query client."""

import math
from dataclasses import dataclass
from logging import getLogger
from typing import Any

import httpx
import xarray as xr

from knmidata._exceptions import KNMIError
from knmidata._http import AsyncTransport
from knmidata._time import to_iso8601
from knmidata._typing import (
    BBox,
    Coords,
    OutputFormat,
    QueryParams,
    TimeLike,
    ZLevel,
)
from knmidata.config import EDR_BASE_URL
from knmidata.edr.parsers import parse_coveragejson, parse_geojson

logger = getLogger(__name__)


def _format_interval(
    start_time: TimeLike | None, end_time: TimeLike | None
) -> str | None:
    """Build the EDR `datetime` query parameter (RFC 3339) from start/end times.

    Both given -> 'start/end'; only `start_time` -> 'start/..'; only `end_time`
    -> '../end'; neither -> None. If exactly one bound is given and is a string
    containing '/' or '..', it is passed through unchanged as an interval
    literal. Naive values are assumed UTC.
    """
    if start_time is None and end_time is None:
        return None
    if start_time is None or end_time is None:
        single = start_time if start_time is not None else end_time
        if isinstance(single, str) and ("/" in single or ".." in single):
            return single
    start = to_iso8601(start_time) if start_time is not None else ".."
    end = to_iso8601(end_time) if end_time is not None else ".."
    return f"{start}/{end}"


def _build_query_params(
    start_time: TimeLike | None = None,
    end_time: TimeLike | None = None,
    parameters: list[str] | None = None,
    output_format: OutputFormat = "CoverageJSON",
    crs: str | None = None,
) -> QueryParams:
    """Assemble the shared EDR data-query parameters (datetime, parameter-name, f, crs)."""
    params: QueryParams = {"f": output_format}
    if dt := _format_interval(start_time, end_time):
        params["datetime"] = dt
    if parameters:
        params["parameter-name"] = ",".join(parameters)
    if crs:
        params["crs"] = crs
    return params


@dataclass(slots=True, kw_only=True)
class AsyncEDRClient:
    """Client for querying the KNMI OGC EDR API service.

    Parameters
    ----------
    transport : AsyncTransport
        Shared HTTP transport and concurrency semaphore.
    auth : httpx.Auth
        Authentication strategy providing the personal EDR API key.
    """

    transport: AsyncTransport
    auth: httpx.Auth

    def __repr__(self) -> str:
        """Concise repr showing transport and auth (auth masks any API key)."""
        return (
            f"{type(self).__name__}(transport={self.transport!r}, auth={self.auth!r})"
        )

    async def collections(self) -> dict[str, Any]:
        """List all collections available in the EDR API.

        Returns
        -------
        dict[str, Any]
            Parsed JSON response of the /collections endpoint.
        """
        return await self.transport.get_json(
            f"{EDR_BASE_URL}/collections", auth=self.auth
        )

    async def collection(self, collection_id: str) -> dict[str, Any]:
        """Get metadata for a collection, including available parameters and output formats.

        Parameters
        ----------
        collection_id : str
            EDR collection ID (e.g. 'daily-in-situ-meteorological-observations').

        Returns
        -------
        dict[str, Any]
            Parsed JSON response of the /collections/{id} endpoint.
        """
        return await self.transport.get_json(
            f"{EDR_BASE_URL}/collections/{collection_id}", auth=self.auth
        )

    async def instances(self, collection_id: str) -> dict[str, Any]:
        """List the instances of a collection.

        Parameters
        ----------
        collection_id : str
            EDR collection ID.

        Returns
        -------
        dict[str, Any]
            Parsed JSON response of the /collections/{id}/instances endpoint.
        """
        return await self.transport.get_json(
            f"{EDR_BASE_URL}/collections/{collection_id}/instances", auth=self.auth
        )

    async def locations(self, collection_id: str) -> dict[str, Any]:
        """List the measurement locations of a collection as GeoJSON.

        Parameters
        ----------
        collection_id : str
            EDR collection ID.

        Returns
        -------
        dict[str, Any]
            GeoJSON FeatureCollection.
        """
        return await self.transport.get_json(
            f"{EDR_BASE_URL}/collections/{collection_id}/locations", auth=self.auth
        )

    async def location(self, collection_id: str, location_id: str) -> dict[str, Any]:
        """Get metadata for a single location.

        Parameters
        ----------
        collection_id : str
            EDR collection ID.
        location_id : str
            Location identifier.

        Returns
        -------
        dict[str, Any]
            GeoJSON Feature.
        """
        return await self.transport.get_json(
            f"{EDR_BASE_URL}/collections/{collection_id}/locations/{location_id}",
            auth=self.auth,
        )

    @staticmethod
    def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        """Great-circle distance in km between two (lon, lat) points."""
        r = 6371.0
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dp = math.radians(lat2 - lat1)
        dl = math.radians(lon2 - lon1)
        a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 2 * r * math.asin(math.sqrt(a))

    async def nearest_location(
        self, collection_id: str, coords: Coords
    ) -> dict[str, Any]:
        """Find the measurement location nearest to `coords` (client-side).

        `/position` does server-side nearest-station matching but only within a
        tolerance — it returns HTTP 404 "The query returned no stations" when no
        station is close enough. This helper always returns the closest station:
        it fetches `/locations` and computes haversine distances.

        Parameters
        ----------
        collection_id : str
            EDR collection ID.
        coords : Coords
            Point coordinates (lon, lat).

        Returns
        -------
        dict[str, Any]
            The nearest location: {'id', 'name', 'coordinates', 'distance_km',
            'properties'}.

        Raises
        ------
        KNMIError
            If the collection has no locations.
        """
        locations = await self.locations(collection_id)
        lon, lat = coords
        best: dict[str, Any] | None = None
        for feat in locations.get("features", []):
            geom_coords = (feat.get("geometry") or {}).get("coordinates") or []
            if len(geom_coords) < 2:
                continue
            props = feat.get("properties", {})
            distance = self._haversine_km(lon, lat, geom_coords[0], geom_coords[1])
            if best is None or distance < best["distance_km"]:
                best = {
                    "id": feat.get("id") or props.get("id"),
                    "name": props.get("name") or props.get("label"),
                    "coordinates": [geom_coords[0], geom_coords[1]],
                    "distance_km": distance,
                    "properties": props,
                }
        if best is None:
            logger.error("Collection '%s' has no locations", collection_id)
            raise KNMIError(f"Collection '{collection_id}' has no locations.")
        logger.info(
            "Nearest location to (%.4f, %.4f) in '%s': %s (%s) at %.2f km",
            lon,
            lat,
            collection_id,
            best["id"],
            best["name"],
            best["distance_km"],
        )
        return best

    async def location_data(
        self,
        collection_id: str,
        location_id: str,
        start_time: TimeLike | None = None,
        end_time: TimeLike | None = None,
        parameters: list[str] | None = None,
        output_format: OutputFormat = "CoverageJSON",
    ) -> xr.Dataset:
        """Query data for a single location (weather station).

        Uses /collections/{id}/locations/{location_id}. This is the typical way
        to get a time series for a station collection: note that /position with
        a POINT coordinate can return HTTP 404 ("The query returned no stations")
        for these collections.

        Parameters
        ----------
        collection_id : str
            EDR collection ID.
        location_id : str
            Location identifier (station id, e.g. '06260' for De Bilt; see
            `locations()`).
        start_time : TimeLike | None, default=None
            Start of the time interval (see `position`).
        end_time : TimeLike | None, default=None
            End of the time interval (see `position`).
        parameters : list[str] | None, default=None
            List of parameter names to query.
        output_format : OutputFormat, default="CoverageJSON"
            Response format: 'CoverageJSON' or 'GeoJSON'.

        Returns
        -------
        xr.Dataset
            In-memory xarray dataset.
        """
        if not location_id:
            raise ValueError(
                "location_id must be a non-empty station id; get ids from locations() "
                "(the GeoJSON feature 'id', e.g. '06260' for De Bilt)."
            )
        endpoint = f"{EDR_BASE_URL}/collections/{collection_id}/locations/{location_id}"
        params = _build_query_params(start_time, end_time, parameters, output_format)
        return await self._query(endpoint, params, output_format)

    async def _query(
        self,
        endpoint: str,
        params: QueryParams,
        output_format: OutputFormat,
    ) -> xr.Dataset:
        """Dispatch a data query to an EDR endpoint and route the response to a parser."""
        logger.info("EDR query: GET %s", endpoint)
        logger.debug("EDR query params: %s", params)
        response = await self.transport.request(
            "GET", endpoint, params=params, auth=self.auth
        )
        payload = response.json()
        if output_format == "GeoJSON":
            return parse_geojson(payload)
        return parse_coveragejson(payload)

    async def position(
        self,
        collection: str,
        coords: Coords,
        start_time: TimeLike | None = None,
        end_time: TimeLike | None = None,
        parameters: list[str] | None = None,
        output_format: OutputFormat = "CoverageJSON",
        crs: str | None = None,
    ) -> xr.Dataset:
        """Query station or point time-series via /collections/{id}/position.

        Note: /position does server-side nearest-station matching, but only
        within a tolerance — it returns HTTP 404 ("The query returned no
        stations") when no station is close enough to the point. For a reliable
        client-side nearest lookup use `nearest_location()`, then
        `location_data()` with the returned station id.

        Parameters
        ----------
        collection : str
            EDR collection ID.
        coords : Coords
            Point coordinates (lon, lat).
        start_time : TimeLike | None, default=None
            Start of the queried time interval (ISO-8601 string, pandas
            Timestamp, datetime, or date; naive values are assumed UTC). Sent as
            an RFC 3339 interval: with both start and end as 'start/end', with
            only start_time as 'start/..'.
        end_time : TimeLike | None, default=None
            End of the queried time interval. With only end_time given the
            interval '../end' is sent. For a single instant, pass the same value
            as start_time and end_time.
        parameters : list[str] | None, default=None
            List of parameter names to query (see collection metadata).
        output_format : str, default="CoverageJSON"
            Response format: 'CoverageJSON' or 'GeoJSON'.
        crs : str | None, default=None
            Coordinate reference system of `coords`, e.g. 'EPSG:4326'.

        Returns
        -------
        xr.Dataset
            In-memory xarray dataset.
        """
        endpoint = f"{EDR_BASE_URL}/collections/{collection}/position"
        params: QueryParams = {
            "coords": f"POINT({coords[0]} {coords[1]})",
            **_build_query_params(start_time, end_time, parameters, output_format, crs),
        }
        return await self._query(endpoint, params, output_format)

    async def area(
        self,
        collection: str,
        bbox: BBox,
        start_time: TimeLike | None = None,
        end_time: TimeLike | None = None,
        parameters: list[str] | None = None,
        output_format: OutputFormat = "CoverageJSON",
        crs: str | None = None,
    ) -> xr.Dataset:
        """Query a spatial bounding box via /collections/{id}/area.

        Parameters
        ----------
        collection : str
            EDR collection ID.
        bbox : BBox
            Bounding box as (minx, miny, maxx, maxy).
        start_time : TimeLike | None, default=None
            Start of the queried time interval (see `position`).
        end_time : TimeLike | None, default=None
            End of the queried time interval (see `position`).
        parameters : list[str] | None, default=None
            List of parameter names to query.
        output_format : str, default="CoverageJSON"
            Response format: 'CoverageJSON' or 'GeoJSON'.
        crs : str | None, default=None
            Coordinate reference system of `bbox`.

        Returns
        -------
        xr.Dataset
            In-memory xarray dataset.
        """
        endpoint = f"{EDR_BASE_URL}/collections/{collection}/area"
        minx, miny, maxx, maxy = bbox
        wkt_poly = (
            f"POLYGON(({minx} {miny}, {maxx} {miny}, "
            f"{maxx} {maxy}, {minx} {maxy}, {minx} {miny}))"
        )
        params: QueryParams = {
            "coords": wkt_poly,
            **_build_query_params(start_time, end_time, parameters, output_format, crs),
        }
        return await self._query(endpoint, params, output_format)

    async def cube(
        self,
        collection: str,
        bbox: BBox,
        z: ZLevel,
        start_time: TimeLike | None = None,
        end_time: TimeLike | None = None,
        parameters: list[str] | None = None,
        output_format: OutputFormat = "CoverageJSON",
        crs: str | None = None,
    ) -> xr.Dataset:
        """Query a 3D volume via /collections/{id}/cube.

        Parameters
        ----------
        collection : str
            EDR collection ID.
        bbox : BBox
            Bounding box as (minx, miny, maxx, maxy).
        z : ZLevel
            Vertical level or interval (e.g. '100' or (100, 500)).
        start_time : TimeLike | None, default=None
            Start of the queried time interval (see `position`).
        end_time : TimeLike | None, default=None
            End of the queried time interval (see `position`).
        parameters : list[str] | None, default=None
            List of parameter names to query.
        output_format : str, default="CoverageJSON"
            Response format: 'CoverageJSON' or 'GeoJSON'.
        crs : str | None, default=None
            Coordinate reference system of `bbox` and `z`.

        Returns
        -------
        xr.Dataset
            In-memory xarray dataset.
        """
        endpoint = f"{EDR_BASE_URL}/collections/{collection}/cube"
        minx, miny, maxx, maxy = bbox
        z_str = f"{z[0]}/{z[1]}" if isinstance(z, tuple) else str(z)
        params: QueryParams = {
            "bbox": f"{minx},{miny},{maxx},{maxy}",
            "z": z_str,
            **_build_query_params(start_time, end_time, parameters, output_format, crs),
        }
        return await self._query(endpoint, params, output_format)
