"""Tests for the EDR client and the CoverageJSON/GeoJSON parsers."""

import httpx
import numpy as np
import pandas as pd
import pytest
import respx

from knmidata._auth import APIKeyAuth
from knmidata._constants import EDR_BASE_URL as EDR
from knmidata._exceptions import KNMIError
from knmidata.edr.client import AsyncEDRClient
from knmidata.edr.parsers import parse_coveragejson, parse_geojson


@pytest.fixture
def edr(transport):
    return AsyncEDRClient(transport=transport, auth=APIKeyAuth(api_key="edr-key"))


def test_parse_coveragejson_axis_order_follows_domain():
    # axes ordered x, y, t -> range values are flattened in that order
    payload = {
        "type": "Coverage",
        "domain": {
            "axes": {
                "x": {"values": [1.0, 2.0]},
                "y": {"values": [10.0]},
                "t": {"values": ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"]},
            },
            "referencing": [{"id": "EPSG:4326"}],
        },
        "parameters": {"TG": {"unit": {"symbol": "C"}, "description": {"en": "temp"}}},
        "ranges": {"TG": {"values": [1.0, 2.0, 3.0, 4.0]}},  # (x=2, y=1, t=2)
    }
    ds = parse_coveragejson(payload)
    assert ds["TG"].dims == ("x", "y", "time")
    assert ds["TG"].shape == (2, 1, 2)
    # values flattened in x-major, then y, then t order
    assert ds["TG"].values[0, 0, 0] == 1.0
    assert ds["TG"].values[1, 0, 1] == 4.0
    assert ds.sizes["time"] == 2
    assert ds.attrs["crs"] == "EPSG:4326"


def test_parse_coveragejson_maps_nulls_to_nan():
    payload = {
        "domain": {
            "axes": {"t": {"values": ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"]}}
        },
        "parameters": {"T": {"unit": {"symbol": "C"}, "description": {"en": "t"}}},
        "ranges": {"T": {"values": [5.0, None]}},
    }
    ds = parse_coveragejson(payload)
    assert np.isnan(ds["T"].values[1])
    assert ds["T"].values[0] == 5.0


def test_parse_coveragejson_rejects_wrong_range_size():
    payload = {
        "domain": {
            "axes": {"t": {"values": ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"]}}
        },
        "parameters": {"T": {"unit": {"symbol": "C"}, "description": {"en": "t"}}},
        "ranges": {"T": {"values": [1.0, 2.0, 3.0]}},
    }
    with pytest.raises(KNMIError, match="expected 2"):
        parse_coveragejson(payload)


def test_parse_coveragejson_collection_concats_along_station():
    def coverage(station, values):
        return {
            "domain": {
                "axes": {
                    "t": {"values": ["2024-01-01T00:00:00Z"]},
                    "composite": {"values": [station]},
                }
            },
            "parameters": {"T": {"unit": {"symbol": "C"}, "description": {"en": "t"}}},
            "ranges": {"T": {"values": values}},
        }

    payload = {
        "type": "CoverageCollection",
        "coverages": [coverage("260", [1.0]), coverage("240", [2.0])],
    }
    ds = parse_coveragejson(payload)
    assert ds["T"].dims == ("time", "station")
    assert ds.sizes["station"] == 2
    assert sorted(ds["T"].values.flatten().tolist()) == [1.0, 2.0]


def test_parse_geojson():
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [4.89, 52.37]},
                "properties": {
                    "station": "260",
                    "datetime": "2024-01-01T00:00:00Z",
                    "T": 5.5,
                },
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [5.12, 52.09]},
                "properties": {
                    "station": "240",
                    "datetime": "2024-01-01T01:00:00Z",
                    "T": 6.5,
                },
            },
        ],
    }
    ds = parse_geojson(payload)
    assert ds.sizes["feature"] == 2
    assert ds["T"].values.tolist() == [5.5, 6.5]
    assert ds["station"].values.tolist() == ["260", "240"]
    assert ds["lon"].values.tolist() == [4.89, 5.12]
    assert ds["time"].dims == ("feature",)
    assert ds["time"].size == 2


async def test_collections_and_metadata_endpoints(edr):
    with respx.mock(assert_all_called=False) as router:
        router.get(f"{EDR}/collections").mock(
            return_value=httpx.Response(200, json={"collections": []})
        )
        router.get(f"{EDR}/collections/c1").mock(
            return_value=httpx.Response(200, json={"id": "c1"})
        )
        router.get(f"{EDR}/collections/c1/locations").mock(
            return_value=httpx.Response(
                200, json={"type": "FeatureCollection", "features": []}
            )
        )
        router.get(f"{EDR}/collections/c1/locations/260").mock(
            return_value=httpx.Response(200, json={"type": "Feature"})
        )
        router.get(f"{EDR}/collections/c1/instances").mock(
            return_value=httpx.Response(200, json={"instances": []})
        )
        assert await edr.collections() == {"collections": []}
        assert await edr.collection("c1") == {"id": "c1"}
        assert (await edr.locations("c1"))["type"] == "FeatureCollection"
        assert await edr.location("c1", "260") == {"type": "Feature"}
        assert await edr.instances("c1") == {"instances": []}


async def test_position_builds_query_and_parses(edr):
    from conftest import coveragejson_payload

    endpoint = f"{EDR}/collections/c1/position"
    with respx.mock(assert_all_called=False) as router:
        route = router.get(endpoint).mock(
            return_value=httpx.Response(200, json=coveragejson_payload())
        )
        ds = await edr.position(
            "c1",
            coords=(4.89, 52.37),
            datetime="2024-01-01T00:00:00Z/2024-01-02T00:00:00Z",
            parameters=["T", "RH"],
        )
        params = route.calls[0].request.url.params
        assert params["coords"] == "POINT(4.89 52.37)"
        assert params["datetime"] == "2024-01-01T00:00:00Z/2024-01-02T00:00:00Z"
        assert params["parameter-name"] == "T,RH"
        assert params["f"] == "CoverageJSON"
        assert route.calls[0].request.headers["Authorization"] == "edr-key"
    assert ds["T"].dims == ("time", "station")
    assert ds["T"].values.tolist() == [[5.0], [6.0]]
    assert ds["T"].attrs["units"] == "C"


async def test_area_sends_polygon(edr):
    endpoint = f"{EDR}/collections/c1/area"
    with respx.mock(assert_all_called=False) as router:
        route = router.get(endpoint).mock(
            return_value=httpx.Response(
                200,
                json={
                    "domain": {"axes": {"t": {"values": ["2024-01-01T00:00:00Z"]}}},
                    "parameters": {
                        "T": {"unit": {"symbol": "C"}, "description": {"en": "t"}}
                    },
                    "ranges": {"T": {"values": [1.0]}},
                },
            )
        )
        ds = await edr.area("c1", bbox=(3.0, 50.5, 7.5, 53.7))
        params = route.calls[0].request.url.params
        assert (
            params["coords"]
            == "POLYGON((3.0 50.5, 7.5 50.5, 7.5 53.7, 3.0 53.7, 3.0 50.5))"
        )
    assert ds["T"].values.tolist() == [1.0]


async def test_cube_sends_bbox_and_z(edr):
    endpoint = f"{EDR}/collections/c1/cube"
    with respx.mock(assert_all_called=False) as router:
        route = router.get(endpoint).mock(
            return_value=httpx.Response(
                200,
                json={
                    "domain": {"axes": {"t": {"values": ["2024-01-01T00:00:00Z"]}}},
                    "parameters": {
                        "T": {"unit": {"symbol": "C"}, "description": {"en": "t"}}
                    },
                    "ranges": {"T": {"values": [1.0]}},
                },
            )
        )
        await edr.cube("c1", bbox=(3.0, 50.5, 7.5, 53.7), z=(100, 500))
        params = route.calls[0].request.url.params
        assert params["bbox"] == "3.0,50.5,7.5,53.7"
        assert params["z"] == "100/500"


async def test_datetime_accepts_timestamp_and_interval(edr):
    endpoint = f"{EDR}/collections/c1/position"
    payload = {
        "domain": {"axes": {"t": {"values": ["2024-01-01T00:00:00Z"]}}},
        "parameters": {"T": {"unit": {"symbol": "C"}, "description": {"en": "t"}}},
        "ranges": {"T": {"values": [1.0]}},
    }
    with respx.mock(assert_all_called=False) as router:
        route = router.get(endpoint).mock(
            return_value=httpx.Response(200, json=payload)
        )
        # single Timestamp, naive -> UTC
        await edr.position(
            "c1", coords=(4.89, 52.37), datetime=pd.Timestamp("2024-01-01 06:00")
        )
        assert route.calls[0].request.url.params["datetime"] == "2024-01-01T06:00:00Z"
        # (start, end) tuple -> interval
        await edr.position(
            "c1",
            coords=(4.89, 52.37),
            datetime=(
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-02T02:00:00+02:00"),
            ),
        )
        assert (
            route.calls[1].request.url.params["datetime"]
            == "2024-01-01T00:00:00Z/2024-01-02T00:00:00Z"
        )
        # interval string passes through unchanged
        await edr.position(
            "c1", coords=(4.89, 52.37), datetime="2024-01-01T00:00:00Z/.."
        )
        assert (
            route.calls[2].request.url.params["datetime"] == "2024-01-01T00:00:00Z/.."
        )


async def test_geojson_output_format(edr):
    endpoint = f"{EDR}/collections/c1/position"
    with respx.mock(assert_all_called=False) as router:
        route = router.get(endpoint).mock(
            return_value=httpx.Response(
                200,
                json={
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "geometry": {"coordinates": [4.89, 52.37]},
                            "properties": {"station": "260", "T": 5.5},
                        }
                    ],
                },
            )
        )
        ds = await edr.position("c1", coords=(4.89, 52.37), output_format="GeoJSON")
        assert route.calls[0].request.url.params["f"] == "GeoJSON"
    assert ds["T"].values.tolist() == [5.5]
