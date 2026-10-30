"""OGC Environmental Data Retrieval (EDR) query client."""

from dataclasses import dataclass
from typing import Any

import httpx
import xarray as xr

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


def _format_interval(
    start_time: TimeLike | None, end_time: TimeLike | None
) -> str | None:
    """Build the EDR `datetime` query parameter (RFC 3339) from start/end times.

    Both given -> 'start/end'; only `start_time` -> 'start/..'; only `end_time`
    -> '../end'; neither -> None. A string containing '/' or '..' is passed
    through unchanged as an interval literal. Naive values are assumed UTC.
    """
    if start_time is None and end_time is None:
        return None
    for value in (start_time, end_time):
        if (
            isinstance(value, str)
            and ("/" in value or ".." in value)
            and (start_time is None or end_time is None)
        ):
            return value
    start = to_iso8601(start_time) if start_time is not None else ".."
    end = to_iso8601(end_time) if end_time is not None else ".."
    return f"{start}/{end}"


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
        params: QueryParams = {}
        if dt := _format_interval(start_time, end_time):
            params["datetime"] = dt
        if parameters:
            params["parameter-name"] = ",".join(parameters)
        return await self._query(endpoint, params, output_format)

    async def _query(
        self,
        endpoint: str,
        params: QueryParams,
        output_format: OutputFormat,
    ) -> xr.Dataset:
        """Dispatch a data query to an EDR endpoint and route the response to a parser."""
        query_params = {**params, "f": output_format}
        response = await self.transport.request(
            "GET", endpoint, params=query_params, auth=self.auth
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

        Note: for station-based collections, /position with a POINT coordinate
        can return HTTP 404 ("The query returned no stations"); use
        `location_data()` with a station id from `locations()` instead.

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
        params: QueryParams = {"coords": f"POINT({coords[0]} {coords[1]})"}
        if dt := _format_interval(start_time, end_time):
            params["datetime"] = dt
        if parameters:
            params["parameter-name"] = ",".join(parameters)
        if crs:
            params["crs"] = crs
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
        params: QueryParams = {"coords": wkt_poly}
        if dt := _format_interval(start_time, end_time):
            params["datetime"] = dt
        if parameters:
            params["parameter-name"] = ",".join(parameters)
        if crs:
            params["crs"] = crs
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
        }
        if dt := _format_interval(start_time, end_time):
            params["datetime"] = dt
        if parameters:
            params["parameter-name"] = ",".join(parameters)
        if crs:
            params["crs"] = crs
        return await self._query(endpoint, params, output_format)
