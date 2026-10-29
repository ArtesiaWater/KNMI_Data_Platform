# KNMI_Data_Platform

`knmidata` is a lightweight Python client for the [KNMI Data Platform](https://dataplatform.knmi.nl/) that fetches observational and gridded meteorological data directly into standard `xarray.Dataset` objects.

## Install

```bash
pip install knmidata
```

## Authentication

All API endpoints require an API key in the `Authorization` header. Request a free registered key at [developer.dataplatform.knmi.nl](https://developer.dataplatform.knmi.nl/), then either pass it explicitly or set an environment variable:

```bash
export KNMI_API_KEY="<your registered key>"        # used by both APIs
export KNMI_OPEN_DATA_KEY="<open data key>"         # Open Data API only
export KNMI_EDR_KEY="<edr key>"                     # EDR API only (no anonymous key exists)
```

Note that Open Data and EDR keys are not interchangeable. For the Open Data API only, you can omit the key entirely and the public anonymous key (published on the docs page) is used automatically, subject to a shared rate limit.

### Keeping keys out of shared notebooks/scripts

Never hardcode a key in a notebook cell. Instead, in order of preference:

1. a gitignored `.env` file loaded with `python-dotenv` (`pip install knmidata[dotenv]`, see `docs/.env.example`): `from dotenv import load_dotenv; load_dotenv()`;
2. shell environment variables (`export KNMI_API_KEY=...`);
3. a key file: `KNMI_API_KEY_FILE=/path/to/key.txt` (chmod 600) — `knmidata` reads the file for you (`KNMI_OPEN_DATA_KEY_FILE` / `KNMI_EDR_KEY_FILE` also exist).

`knmidata` never prints or logs the key, and strip notebook outputs before sharing (pre-commit runs `nbstripout`).

## Supported APIs

- **Open Data API** (`https://api.dataplatform.knmi.nl/open-data/v1`): bulk file archives (e.g. radar precipitation composites, in-situ observations). Handles cursor-based file pagination (with optional ISO-8601 time filtering), AWS S3 presigned URL retrieval, concurrent streaming downloads with atomic writes, and assembly into `xarray.Dataset` via `xr.open_mfdataset`. Only NetCDF/HDF5 files are supported (GRIB files, e.g. HARMONIE model output, are not).
- **EDR API** (`https://api.dataplatform.knmi.nl/edr/v1`): OGC Environmental Data Retrieval for spatio-temporal queries — `/collections`, `/locations`, `/instances`, `/position`, `/area`, `/cube`. Responses in CoverageJSON or GeoJSON are parsed into in-memory `xarray.Dataset` objects.

## Usage

### Open Data: download files and open as a dataset

```python
import knmidata

# one-liner (uses KNMI_API_KEY or the anonymous key)
ds = knmidata.open_dataset(
    "10-minute-in-situ-meteorological-observations",
    "1.0",
    start_time="2024-06-01T00:00:00Z",  # str, pd.Timestamp, datetime, or date
    end_time=pd.Timestamp("2024-06-02"),  # naive values are assumed UTC
    file_pattern=r"KMDS__OPER_P___10M_OBS_L2_2024060[12].*\.nc",
    cache_dir="./data",  # keep downloaded files here; omit to load into memory only
)
```

Time arguments accept `str`, `pandas.Timestamp`, `datetime`, or `date` (exported as `knmidata.TimeLike`) and are normalized to UTC ISO-8601 before being sent to the API.

### Async client

```python
import knmidata

async with knmidata.AsyncKNMIClient() as client:
    files = await client.catalog.list_files(
        "10-minute-in-situ-meteorological-observations",
        "1.0",
        start_time="2024-06-01T00:00:00Z",
        end_time="2024-06-02T00:00:00Z",
    )
    ds = await client.open_dataset(
        "10-minute-in-situ-meteorological-observations",
        "1.0",
        start_time="2024-06-01T00:00:00Z",
        end_time="2024-06-02T00:00:00Z",
    )
```

### EDR: spatio-temporal queries

```python
import knmidata

with knmidata.KNMIClient() as client:
    print(client.edr.collections())  # available collections
    meta = client.edr.collection("hourly-in-situ-meteorological-observations")
    print(client.edr.locations("hourly-in-situ-meteorological-observations"))

    ds = client.edr.position(
        "hourly-in-situ-meteorological-observations",
        coords=(4.89, 52.37),  # De Bilt: lon, lat
        datetime="2024-06-01T00:00:00Z/2024-06-07T00:00:00Z",
        parameters=["T", "RH"],
    )
```

### Sync client

`KNMIClient` runs all requests on a dedicated background event loop, so it can be called repeatedly and from environments that already run an event loop (e.g. Jupyter notebooks):

```python
import knmidata

with knmidata.KNMIClient() as client:
    ds = client.open_dataset(
        "daily-in-situ-meteorological-observations",
        "1.0",
        start_time="2024-01-01T00:00:00Z",
        max_files=10,
    )
    ts = client.edr.area(
        "Tg1",
        bbox=(3.0, 50.5, 7.5, 53.7),
        datetime="2024-06-01/2024-06-30",
        parameters=["TG"],
    )
```

## Examples

See [docs/example_hourly_in_situ_observations.py](docs/example_hourly_in_situ_observations.py) — a notebook-style (`# %%` cells) walkthrough that fetches the [hourly-in-situ-meteorological-observations-validated](https://dataplatform.knmi.nl/dataset/access/hourly-in-situ-meteorological-observations-validated-1-0) dataset via both the Open Data API and the EDR API, explaining how KNMI data is organized and when to use which API:

```bash
python docs/example_hourly_in_situ_observations.py   # as a script
jupytext --to notebook docs/example_hourly_in_situ_observations.py  # as a notebook
```

## Development

```bash
uv sync --extra dev
uv run pytest            # tests with coverage
uv run ruff check        # lint
uv run ty check src      # type check
```
