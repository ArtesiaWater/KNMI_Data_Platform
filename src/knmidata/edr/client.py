"""OGC Environmental Data Retrieval (EDR) query client."""

from dataclasses import dataclass
from typing import Any

import httpx
import xarray as xr

from knmidata._http import AsyncTransport
from knmidata.edr.parsers import parse_coveragejson, parse_netcdf_bytes

EDR_BASE_URL = "https://api.knmi.nl/edr/v1"


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

    async def _execute_query(
        self, endpoint: str, params: dict[str, Any], output_format: str
    ) -> xr.Dataset:
        """Dispatch query to EDR endpoint and route response to the correct parser."""
        params["f"] = output_format

        async with self.transport.semaphore:
            response = await self.transport.client.get(
                endpoint, params=params, auth=self.auth
            )
            response.raise_for_status()

        if output_format.lower() == "netcdf":
            return parse_netcdf_bytes(response.content)
        return parse_coveragejson(response.json())

    async def position(
        self,
        collection: str,
        coords: tuple[float, float],
        datetime: str | None = None,
        parameters: list[str] | None = None,
        output_format: str = "CoverageJSON",
    ) -> xr.Dataset:
        """Query station or point time-series via /collections/{id}/position.

        Parameters
        ----------
        collection : str
            EDR collection ID.
        coords : tuple[float, float]
            Point coordinates (lon, lat) or (x, y).
        datetime : str | None, default=None
            ISO-8601 interval (e.g. '2026-01-01T00:00:00Z/2026-01-02T00:00:00Z').
        parameters : list[str] | None, default=None
            List of variable/parameter names to query.
        output_format : str, default="CoverageJSON"
            Response format: 'CoverageJSON' or 'NetCDF'.

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

        return await self._execute_query(endpoint, params, output_format)

    async def area(
        self,
        collection: str,
        bbox: tuple[float, float, float, float],
        datetime: str | None = None,
        parameters: list[str] | None = None,
        output_format: str = "CoverageJSON",
    ) -> xr.Dataset:
        """Query spatial polygon/bounding box via /collections/{id}/area.

        Parameters
        ----------
        collection : str
            EDR collection ID.
        bbox : tuple[float, float, float, float]
            Bounding box as (minx, miny, maxx, maxy).
        datetime : str | None, default=None
            ISO-8601 datetime or interval.
        parameters : list[str] | None, default=None
            List of variable/parameter names to query.
        output_format : str, default="CoverageJSON"
            Response format: 'CoverageJSON' or 'NetCDF'.

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

        return await self._execute_query(endpoint, params, output_format)

    async def cube(
        self,
        collection: str,
        bbox: tuple[float, float, float, float],
        z: str | tuple[float, float],
        datetime: str | None = None,
        parameters: list[str] | None = None,
        output_format: str = "CoverageJSON",
    ) -> xr.Dataset:
        """Query 3D volume via /collections/{id}/cube.

        Parameters
        ----------
        collection : str
            EDR collection ID.
        bbox : tuple[float, float, float, float]
            Bounding box as (minx, miny, maxx, maxy).
        z : str | tuple[float, float]
            Vertical height level or interval (e.g. '100' or (100, 500)).
        datetime : str | None, default=None
            ISO-8601 datetime or interval.
        parameters : list[str] | None, default=None
            List of variable/parameter names to query.
        output_format : str, default="CoverageJSON"
            Response format: 'CoverageJSON' or 'NetCDF'.

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

        return await self._execute_query(endpoint, params, output_format)
