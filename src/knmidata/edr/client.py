"""OGC Environmental Data Retrieval (EDR) query client."""

from dataclasses import dataclass
from typing import Any, Literal

import httpx
import xarray as xr

from knmidata._constants import EDR_BASE_URL
from knmidata._http import AsyncTransport
from knmidata.edr.parsers import parse_coveragejson, parse_geojson

OutputFormat = Literal["CoverageJSON", "GeoJSON"]


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
            EDR collection ID (e.g. 'hourly-in-situ-meteorological-observations').

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

    async def _query(
        self,
        endpoint: str,
        params: dict[str, Any],
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
        coords: tuple[float, float],
        datetime: str | None = None,
        parameters: list[str] | None = None,
        output_format: OutputFormat = "CoverageJSON",
        crs: str | None = None,
    ) -> xr.Dataset:
        """Query station or point time-series via /collections/{id}/position.

        Parameters
        ----------
        collection : str
            EDR collection ID.
        coords : tuple[float, float]
            Point coordinates (lon, lat).
        datetime : str | None, default=None
            RFC 3339 datetime or interval (e.g. '2026-01-01T00:00:00Z/2026-01-02T00:00:00Z').
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
        params: dict[str, Any] = {"coords": f"POINT({coords[0]} {coords[1]})"}
        if datetime:
            params["datetime"] = datetime
        if parameters:
            params["parameter-name"] = ",".join(parameters)
        if crs:
            params["crs"] = crs
        return await self._query(endpoint, params, output_format)

    async def area(
        self,
        collection: str,
        bbox: tuple[float, float, float, float],
        datetime: str | None = None,
        parameters: list[str] | None = None,
        output_format: OutputFormat = "CoverageJSON",
        crs: str | None = None,
    ) -> xr.Dataset:
        """Query a spatial bounding box via /collections/{id}/area.

        Parameters
        ----------
        collection : str
            EDR collection ID.
        bbox : tuple[float, float, float, float]
            Bounding box as (minx, miny, maxx, maxy).
        datetime : str | None, default=None
            RFC 3339 datetime or interval.
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
        params: dict[str, Any] = {"coords": wkt_poly}
        if datetime:
            params["datetime"] = datetime
        if parameters:
            params["parameter-name"] = ",".join(parameters)
        if crs:
            params["crs"] = crs
        return await self._query(endpoint, params, output_format)

    async def cube(
        self,
        collection: str,
        bbox: tuple[float, float, float, float],
        z: str | tuple[float, float],
        datetime: str | None = None,
        parameters: list[str] | None = None,
        output_format: OutputFormat = "CoverageJSON",
        crs: str | None = None,
    ) -> xr.Dataset:
        """Query a 3D volume via /collections/{id}/cube.

        Parameters
        ----------
        collection : str
            EDR collection ID.
        bbox : tuple[float, float, float, float]
            Bounding box as (minx, miny, maxx, maxy).
        z : str | tuple[float, float]
            Vertical level or interval (e.g. '100' or (100, 500)).
        datetime : str | None, default=None
            RFC 3339 datetime or interval.
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
        params: dict[str, Any] = {
            "bbox": f"{minx},{miny},{maxx},{maxy}",
            "z": z_str,
        }
        if datetime:
            params["datetime"] = datetime
        if parameters:
            params["parameter-name"] = ",".join(parameters)
        if crs:
            params["crs"] = crs
        return await self._query(endpoint, params, output_format)
