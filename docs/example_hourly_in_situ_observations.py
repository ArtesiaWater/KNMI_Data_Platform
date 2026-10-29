"""Notebook-style example: KNMI hourly in-situ observations via Open Data and EDR."""

# %% [markdown]
# # Hourly weather observations from KNMI: Open Data API vs EDR API
#
# This notebook-style script shows how to get data from the
# [KNMI Data Platform](https://dataplatform.knmi.nl/) with the `knmidata`
# Python package, using the dataset
# [**hourly-in-situ-meteorological-observations-validated**](https://dataplatform.knmi.nl/dataset/access/hourly-in-situ-meteorological-observations-validated-1-0)
# (hourly observations from the Dutch automatic weather stations, validated).
#
# We fetch the same dataset through **both** APIs and compare:
#
# 1. **Open Data API** — download whole files (NetCDF) and open them as one
#    `xarray.Dataset`.
# 2. **EDR API** — run spatio-temporal *queries* (position / area) and parse
#    the response directly into an `xarray.Dataset`.
#
# The file uses the `# %%` "percent" cell format: in VS Code / Jupyter /
# Spyder it renders as a notebook, and it also runs as a plain script:
#
# ```bash
# python docs/example_hourly_in_situ_observations.py
# # or convert to a real notebook:
# jupytext --to notebook docs/example_hourly_in_situ_observations.py
# ```

# %% [markdown]
# ## How KNMI data is organized
#
# - The **KNMI Data Platform** (`dataplatform.knmi.nl`) is the data catalog.
#   Everything in it is a **dataset** with a unique *name* and *version*
#   (e.g. `hourly-in-situ-meteorological-observations-validated`, version
#   `1.0`). The version matters: the API paths contain both.
# - Datasets come in two flavours:
#   - **in-situ observations**: point measurements at automatic weather
#     stations (AWS), e.g. temperature, wind, precipitation per station;
#   - **gridded products**: radar composites, HARMONIE model output, satellite
#     imagery — regularly spaced grids.
# - Each dataset is exposed through one or two APIs (see the "Access" tab on
#   the dataset page):
#   - the **Open Data API** (`/open-data/v1`): file-based — you list files and
#     download them (via temporary presigned S3 URLs);
#   - the **EDR API** (`/edr/v1`): an OGC standard for spatio-temporal
#     *queries* — you ask for a point, area, or cube and get back a structured
#     payload (CoverageJSON), no files involved.
# - Times are **UTC** everywhere. KNMI file names embed the observation time
#   (e.g. `..._202406011200.nc` = 2024-06-01 12:00 UTC).
# - "validated" in the dataset name means the observations have passed KNMI's
#   quality control (near-real-time datasets are not validated yet).

# %% [markdown]
# ## The two APIs at a glance
#
# | | Open Data API | EDR API |
# |---|---|---|
# | Base URL | `api.dataplatform.knmi.nl/open-data/v1` | `api.dataplatform.knmi.nl/edr/v1` |
# | Model | list *files* in a dataset, get a presigned download URL per file | query a *collection* by location/time/parameters |
# | Typical use | bulk download of whole files / long history | targeted extracts: one station, one day, a few parameters |
# | You get | NetCDF/HDF5 files on disk → `xarray` | CoverageJSON/GeoJSON → parsed to `xarray.Dataset` in memory |
# | Granularity | whole files (all stations/variables in the file) | exactly the queried subset |
# | Auth | anonymous key possible (shared, rate-limited) | registered key required (dedicated quota) |
#
# Rule of thumb: **use EDR when you know what you want** (a station, a bbox, a
# time range, a few variables); **use Open Data when you want everything**
# (whole files, full archives, model/radar grids).

# %% [markdown]
# ## Authentication
#
# Every request needs an API key in the `Authorization` header (raw key, no
# `Bearer` prefix). Request a free registered key at
# [developer.dataplatform.knmi.nl](https://developer.dataplatform.knmi.nl/).
#
# `knmidata` resolves keys in this order — explicit argument first, then
# environment variables:
#
# | API | Argument | Environment variables |
# |---|---|---|
# | Open Data | `api_key=` | `KNMI_OPEN_DATA_KEY` (+`_FILE`), then `KNMI_API_KEY` (+`_FILE`), then the public anonymous key |
# | EDR | `edr_key=` | `KNMI_EDR_KEY` (+`_FILE`), then `KNMI_API_KEY` (+`_FILE`) |
#
# Note: **Open Data and EDR keys are not interchangeable**, and the EDR API
# has no anonymous key. The anonymous Open Data key is published on the docs
# page and scraped automatically by `knmidata`, but it shares one rate limit
# across all anonymous users.
#
# ### Keeping the key out of shared notebooks
#
# **Never paste the key into a notebook cell** — anyone who can view the
# notebook (or its saved outputs, git history, screenshots) can see it. Safe
# patterns, in order of preference:
#
# 1. **`.env` file (gitignored) + python-dotenv.** Copy `docs/.env.example`
#    to `.env`, fill in your keys, `chmod 600 .env`. `.env` is in
#    `.gitignore`, so it is never committed. In the notebook, only load it:
#    `load_dotenv()` — the key itself never appears in the notebook.
#    Install with `pip install knmidata[dotenv]`.
# 2. **Shell environment variables**: `export KNMI_API_KEY=...` /
#    `export KNMI_EDR_KEY=...` in your profile or before launching Jupyter.
# 3. **Key file**: set `KNMI_API_KEY_FILE=/path/to/key.txt` (chmod 600)
#    instead of the key itself; `knmidata` reads the file for you.
#
# Additional hygiene: `knmidata` never prints or logs the key (check with
# `bool(os.environ.get("KNMI_API_KEY"))`, not by printing it); when sharing an
# executed `.ipynb`, strip outputs first (the repo's pre-commit already runs
# `nbstripout`); if a key ever does leak, revoke/rotate it in the
# [developer portal](https://developer.dataplatform.knmi.nl/).

# %%
from pathlib import Path

import pandas as pd

import knmidata

# The dataset from the KNMI Data Platform page linked above.
DATASET_NAME = "hourly-in-situ-meteorological-observations-validated"
DATASET_VERSION = "1.0"
COLLECTION = "hourly-in-situ-meteorological-observations-validated"  # EDR collection id

# De Bilt (KNMI's own station, WMO/WSI 260), as (lon, lat).
STATION_COORDS = (4.89, 52.37)
# Bounding box over the Netherlands: (minx, miny, maxx, maxy).
NL_BBOX = (3.0, 50.5, 7.5, 53.7)

# Time arguments accept str, pd.Timestamp, datetime, or date; naive = UTC.
START = pd.Timestamp("2024-06-01")
END = pd.Timestamp("2024-06-08")

# Downloaded Open Data files are kept here (omit cache_dir to load into
# memory only and delete the files afterwards).
CACHE_DIR = Path("data")

print(f"knmidata {knmidata.__version__}")

# %% [markdown]
# ## Setup: create a client
#
# `KNMIClient` is the synchronous client (runs everything on a background
# event loop, so it also works inside Jupyter). For async code use
# `AsyncKNMIClient` instead. Without keys, Open Data falls back to the
# anonymous key; the EDR part below checks whether a key is available and
# skips gracefully otherwise.

# %%
import os

# Load keys from the gitignored .env file (see docs/.env.example).
# The key itself never appears in this notebook. On GitHub Actions, skip the
# .env file: store the key as a repository *secret* and expose it to the
# workflow as an env var (see .github/workflows/notebooks.yaml) — GitHub
# masks secret values in logs.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # alternative: export KNMI_API_KEY / KNMI_EDR_KEY in your shell

# Presence check only — never print the key itself.
print("KNMI_API_KEY set:", bool(os.environ.get("KNMI_API_KEY")))
print("KNMI_EDR_KEY set:", bool(os.environ.get("KNMI_EDR_KEY")))

try:
    client = knmidata.KNMIClient()  # keys are resolved from the environment
    client_ok = True
except knmidata.KNMIAuthError as exc:
    client = None
    client_ok = False
    print(f"Could not create a client (no usable key?): {exc}")

# %% [markdown]
# ## 1. Open Data API — list and download whole files
#
# The Open Data API is **file-based**:
#
# 1. `GET /datasets/{name}/versions/{version}/files` lists the files, with
#    cursor pagination (`maxKeys` up to 1000 + `nextPageToken`).
# 2. Time filtering uses `begin`/`end`, but note: they are compared against
#    the `orderBy` attribute. `knmidata` therefore sends
#    `orderBy=lastModified` automatically when you pass `start_time`/`end_time`
#    (with the default `orderBy=filename`, `begin`/`end` would be compared as
#    *filename strings* instead of timestamps).
# 3. `GET .../files/{filename}/url` returns a short-lived **presigned S3 URL**.
# 4. The file is streamed to disk (atomic `.part` → rename, size-checked), and
#    `open_dataset` assembles the files into one `xarray.Dataset`
#    (`combine_by_coords` over the time dimension).

# %%
if client_ok:
    try:
        files = client.catalog.list_files(
            DATASET_NAME,
            DATASET_VERSION,
            start_time=START,
            end_time=END,
        )
    except knmidata.KNMIHTTPError as exc:
        # e.g. HTTP 429 when the shared anonymous key is rate-limited
        print(f"Listing files failed: {exc}")
        files = []
    print(f"{len(files)} files between {START} and {END}, first 5:")
    for f in files[:5]:
        print("  " + f)

# %% [markdown]
# Now download just the first two files of that period and open them as a
# single dataset. Each hourly file contains all stations for that hour, so
# even `max_files=2` already gives you a 2-hour, all-stations slice.
#
# `start_time`/`end_time` are normalized to UTC ISO-8601 (`...Z`) before being
# sent — passing `pd.Timestamp` objects is fine.

# %%
ds = None
if client_ok:
    try:
        ds = client.open_dataset(
            DATASET_NAME,
            DATASET_VERSION,
            start_time=START,
            end_time=END,
            max_files=2,  # remove this to download the whole week
            cache_dir=CACHE_DIR,
        )
    except knmidata.KNMIHTTPError as exc:
        # e.g. HTTP 429 when the shared anonymous key is rate-limited:
        # register for your own key (see above) and pass api_key="...".
        print(f"Open Data request failed: {exc}")

if ds is not None:
    print(f"\nDownloaded files are cached in: {CACHE_DIR.resolve()}")
    print("\nCombined dataset (2 hourly files):")
    print(ds)
    print("\nVariables in the dataset:", list(ds.data_vars))

# %% [markdown]
# ## 2. EDR API — query instead of download
#
# The EDR API (OGC Environmental Data Retrieval standard) exposes datasets as
# **collections**:
#
# - `GET /collections` / `GET /collections/{id}` — what's available: which
#   parameters (variables), which output formats, the spatial/temporal extent.
# - `GET /collections/{id}/locations` — the measurement locations (GeoJSON).
# - Data queries: `/position` (point, WKT `POINT(lon lat)`), `/area` (polygon
#   from a bbox), `/cube` (3D: bbox + `z`), with query parameters:
#   - `datetime` — RFC 3339 instant or interval (`start/end`, open intervals
#     with `..`); `knmidata` accepts `pd.Timestamp` or a `(start, end)` tuple;
#   - `parameter-name` — comma-separated variables (see collection metadata);
#   - `f` — output format: `CoverageJSON` (parsed to `xr.Dataset`) or
#     `GeoJSON`.
#
# Instead of files, you get a structured JSON payload describing exactly the
# queried cube of data, which `knmidata` parses into an `xarray.Dataset`
# in memory. That makes EDR ideal for "give me temperature at De Bilt for last
# week" without downloading gigabytes of files.

# %%
try:
    collections = client.edr.collections()
    edr_available = True
except knmidata.KNMIAuthError as exc:
    edr_available = False
    print(f"EDR API needs a registered key — skipping the EDR section.\n{exc}")

if edr_available:
    print(
        f"Available EDR collections: {[c['id'] for c in collections.get('collections', [])]}"
    )

    # Collection metadata: description, extent, and the available parameters.
    meta = client.edr.collection(COLLECTION)
    parameter_ids = list(meta.get("parameters", {}))
    print(
        f"\nCollection '{COLLECTION}': {len(parameter_ids)} parameters, e.g. "
        + ", ".join(parameter_ids[:5])
    )
    parameters = parameter_ids[:2] or None  # query just the first two variables

# %% [markdown]
# ### Locations: where are the stations?
#
# `/locations` returns a GeoJSON FeatureCollection — one feature per weather
# station, with its coordinates and metadata (station id, name, height).

# %%
if edr_available:
    locations = client.edr.locations(COLLECTION)
    features = locations.get("features", [])
    print(f"{len(features)} measurement locations; first 3:")
    for feat in features[:3]:
        props = feat.get("properties", {})
        print(
            f"  {props.get('id')}: {props.get('name')} at {feat['geometry']['coordinates']}"
        )

# %% [markdown]
# ### Position query: time series at De Bilt
#
# `coords=POINT(lon lat)` asks for the station nearest to that point.
# `datetime=(START, END)` becomes an RFC 3339 interval. The response is
# CoverageJSON with a `time` axis and a `station` axis — parsed into a
# dataset where each queried parameter is a data variable with units.

# %%
if edr_available:
    ts = client.edr.position(
        COLLECTION,
        coords=STATION_COORDS,
        datetime=(START, END),
        parameters=parameters,
    )
    print("Time series at the station nearest to De Bilt:")
    print(ts)

# %% [markdown]
# ### Area query: all stations in a bounding box
#
# `area` takes a bbox and returns every station inside it (a CoverageCollection
# of per-station coverages, merged by `knmidata` along the `station` dim).

# %%
if edr_available:
    area = client.edr.area(
        COLLECTION,
        bbox=NL_BBOX,
        datetime=(START, END),
        parameters=parameters,
    )
    print("All stations within the NL bounding box:")
    print(area)

# %% [markdown]
# ## Open Data vs EDR — what did we just see?
#
# - **Same dataset, two access patterns.** Open Data gave us whole hourly
#   *files* (all stations, all variables in the file) assembled along `time`;
#   EDR gave us exactly the queried subset (2 parameters, 1 station or all
#   stations in a bbox, 1 week) as an in-memory dataset.
# - **Data volume.** Open Data moves whole files (MBs-GBs each); EDR moves
#   only the queried slice (KBs-MBs). EDR is much cheaper for targeted
#   questions, but cannot give you data that isn't exposed as a collection.
# - **Formats.** Open Data files are NetCDF/HDF5 (this package opens them with
#   xarray; GRIB model output would need an extra backend). EDR speaks
#   CoverageJSON/GeoJSON, parsed here into xarray.
# - **Auth/quota.** Open Data works anonymously within a shared rate limit;
#   EDR needs your registered key with a dedicated quota.
# - Both paths end in the same place: an `xarray.Dataset` you can analyze,
#   plot, or write to disk.

# %%
if client_ok:
    client.close()
print("\nDone. Client closed.")
