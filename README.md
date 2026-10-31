# KNMI_Data_Platform

`knmidata` is a lightweight Python client for the [KNMI Data Platform](https://dataplatform.knmi.nl/) that fetches observational and gridded meteorological data directly into standard `xarray.Dataset` objects.

Two APIs are supported:

- **Open Data API** — bulk NetCDF file archives (e.g. radar composites, in-situ observations), downloaded and assembled into one `xarray.Dataset`.
- **EDR API** — OGC EDR spatio-temporal queries (`/collections`, `/locations`, `/position`, `/area`, `/cube`); CoverageJSON/GeoJSON responses are parsed into in-memory `xarray.Dataset` objects.

## Install

```bash
pip install knmidata
```

## Authentication

All endpoints require an API key. Request a free key at [developer.dataplatform.knmi.nl](https://developer.dataplatform.knmi.nl/). KNMI issues a **separate key per API** — an Open Data key does not work for EDR and vice versa:

```bash
export KNMI_OPENDATA_API_KEY="<your open data key>"
export KNMI_EDR_API_KEY="<your edr key>"   # EDR has no anonymous key
```

For Open Data only, you can omit the key entirely and the public anonymous key is used automatically (shared rate limit).

## Usage

Time arguments accept `str`, `pandas.Timestamp`, `datetime`, or `date` (exported as `knmidata.TimeLike`); naive values are assumed UTC and normalized to ISO-8601 UTC.

### Synchronous client

`KNMIClient` runs requests on a background event loop, so it works in plain scripts and in environments that already run a loop (e.g. Jupyter):

```python
import knmidata

with knmidata.KNMIClient() as client:
    # Open Data: get files as one dataset (cached on disk if cache_dir is given)
    ds = client.opendata.get_dataset(
        "daily-in-situ-meteorological-observations",
        "1.0",
        start_time="2024-01-01",
        end_time="2024-01-31",
        cache_dir="./data",  # omit to keep the dataset in memory only
        max_files=10,
    )

    # EDR: time series for one station (De Bilt, WSI 06260)
    ts = client.edr.get_location(
        "daily-in-situ-meteorological-observations",
        "06260",
        start_time="2026-06-01",
        end_time="2026-06-08",
        parameters=["RH", "TG", "EV24"],  # or a single variable: "TG"
    )
    ts["TG"].plot()  # needs matplotlib
```

### One-liners

The same without keeping a client around:

```python
import knmidata

ds = knmidata.get_dataset(
    "10-minute-in-situ-meteorological-observations",
    "1.0",
    start_time="2024-06-01",
    end_time="2024-06-02",
    cache_dir="./data",
)

ts = knmidata.get_timeseries(
    "daily-in-situ-meteorological-observations",
    "06260",
    start_time="2026-06-01",
    end_time="2026-06-08",
    parameters="TG",
)
```

### Asynchronous client

The same API is available asynchronously:

```python
import knmidata

async with knmidata.AsyncKNMIClient() as client:
    ds = await client.opendata.get_dataset(
        "daily-in-situ-meteorological-observations", "1.0", ...
    )
    ts = await client.edr.get_location(
        "daily-in-situ-meteorological-observations", "06260", ...
    )
```

### Lower-level methods

`client.opendata` also exposes `list_files`, `get_download_url`, `download_files` (API → disk), `open_dataset` (disk → `xr.Dataset`), and `get_dataset` (either). `client.edr` exposes metadata (`collections`, `collection`, `instances`, `locations`) and the queries `get_location`, `get_position`, `get_area`, `get_cube`, plus `nearest_location` to find a station by coordinates.

See [docs/example_daily_in_situ_observations.ipynb](docs/example_daily_in_situ_observations.ipynb) for a full walkthrough comparing both APIs.

## Documentation

The documentation (example walkthrough and API reference) is built with Sphinx and published to GitHub Pages at [artesiawater.github.io/KNMI_Data_Platform](https://artesiawater.github.io/KNMI_Data_Platform/).

## Development

```bash
uv sync --extra dev
uv run pytest            # tests with coverage
uv run ruff check        # lint
uv run ty check src      # type check
```
