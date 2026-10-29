"""KNMI OGC EDR API client components."""

from knmidata.edr.client import AsyncEDRClient
from knmidata.edr.parsers import parse_coveragejson, parse_geojson

__all__ = ["AsyncEDRClient", "parse_coveragejson", "parse_geojson"]
