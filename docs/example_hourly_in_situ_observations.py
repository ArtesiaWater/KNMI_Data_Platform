"""Example: fetch the hourly validated in-situ observations dataset via both APIs.

Dataset page:
https://dataplatform.knmi.nl/dataset/access/hourly-in-situ-meteorological-observations-validated-1-0

- Open Data API: dataset 'hourly-in-situ-meteorological-observations-validated',
  version '1.0' (bulk NetCDF files, downloaded and opened as one xarray.Dataset).
- EDR API: collection 'hourly-in-situ-meteorological-observations-validated'
  (spatio-temporal queries parsed directly into an xarray.Dataset).

Authentication:
- Open Data API: pass ``api_key=...`` to the client or set KNMI_API_KEY /
  KNMI_OPEN_DATA_KEY. If omitted, the public anonymous key is used automatically
  (shared rate limit).
- EDR API: requires a registered key (there is no anonymous EDR key): pass
  ``edr_key=...`` or set KNMI_EDR_KEY / KNMI_API_KEY.

Run with:  python docs/example_hourly_in_situ_observations.py
"""

from pathlib import Path

import pandas as pd

import knmidata

DATASET_NAME = "hourly-in-situ-meteorological-observations-validated"
DATASET_VERSION = "1.0"
COLLECTION = "hourly-in-situ-meteorological-observations-validated"

# De Bilt (station 260), as (lon, lat).
STATION_COORDS = (4.89, 52.37)
# Bounding box over the Netherlands: (minx, miny, maxx, maxy).
NL_BBOX = (3.0, 50.5, 7.5, 53.7)

# Time arguments accept str, pd.Timestamp, datetime, or date (naive = UTC).
START = pd.Timestamp("2024-06-01")
END = pd.Timestamp("2024-06-08")

# Downloaded Open Data files are kept here (omit to load into memory only).
CACHE_DIR = Path("data")


def open_data_example(client: knmidata.KNMIClient) -> None:
    """Download a few hourly files via the Open Data API and open them as a dataset."""
    print("=== Open Data API ===")
    ds = client.open_dataset(
        DATASET_NAME,
        DATASET_VERSION,
        start_time=START,
        end_time=END,
        max_files=2,  # remove to download the whole period
        cache_dir=CACHE_DIR,
    )
    print(f"Downloaded files are cached in: {CACHE_DIR.resolve()}")
    print(ds)


def edr_example(client: knmidata.KNMIClient) -> None:
    """Query the same dataset via the EDR API: metadata, locations, position, area."""
    print("\n=== EDR API ===")
    try:
        meta = client.edr.collection(COLLECTION)
    except knmidata.KNMIAuthError as exc:
        print(f"Skipping EDR example: {exc}")
        return

    # Discover available parameters from the collection metadata.
    parameter_ids = list(meta.get("parameters", {}))
    print(f"Collection '{COLLECTION}' has {len(parameter_ids)} parameters, e.g.:")
    print("  " + ", ".join(parameter_ids[:5]))
    parameters = parameter_ids[:2] or None

    locations = client.edr.locations(COLLECTION)
    print(f"Number of measurement locations: {len(locations.get('features', []))}")

    # Time series at the point nearest to De Bilt.
    ts = client.edr.position(
        COLLECTION,
        coords=STATION_COORDS,
        datetime=(START, END),  # (start, end) tuple -> RFC 3339 interval
        parameters=parameters,
    )
    print("\nPosition query at De Bilt (nearest station):")
    print(ts)

    # All stations inside the Netherlands bounding box.
    area = client.edr.area(
        COLLECTION,
        bbox=NL_BBOX,
        datetime=(START, END),
        parameters=parameters,
    )
    print("\nArea query over the Netherlands:")
    print(area)


def main() -> None:
    """Run the Open Data and EDR examples with a synchronous client."""
    with knmidata.KNMIClient() as client:
        open_data_example(client)
        edr_example(client)


if __name__ == "__main__":
    main()
