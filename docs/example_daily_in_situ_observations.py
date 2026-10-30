"""Notebook-style example: KNMI daily in-situ observations via Open Data and EDR."""

# %% [markdown]
# # Daily weather observations from KNMI: Open Data API vs EDR API
#
# This notebook-style script shows how to get data from the
# [KNMI Data Platform](https://dataplatform.knmi.nl/) with the `knmidata`
# Python package, using the dataset
# [**daily-in-situ-meteorological-observations**](https://dataplatform.knmi.nl/dataset/access/daily-in-situ-meteorological-observations-1-0)
# (daily observations from the Dutch automatic weather stations).
#
# We fetch the same dataset through **both** APIs and compare:
#
# 1. **Open Data API** — download whole files (NetCDF) and open them as one
#    `xarray.Dataset`.
# 2. **EDR API** — run spatio-temporal *queries* (location / area), parse the
#    response into an `xarray.Dataset`, and plot the time series of
#    precipitation (RH), temperature (TG) and evaporation (EV24) at De Bilt.
#
# The file uses the `# %%` "percent" cell format: in VS Code / Jupyter /
# Spyder it renders as a notebook, and it also runs as a plain script:
#
# ```bash
# python docs/example_daily_in_situ_observations.py
# # or convert to a real notebook:
# jupytext --to notebook docs/example_daily_in_situ_observations.py
# ```

# %% [markdown]
# ## How KNMI data is organized
#
# - The **KNMI Data Platform** (`dataplatform.knmi.nl`) is the data catalog.
#   Everything in it is a **dataset** with a unique *name* and *version*
#   (e.g. `daily-in-situ-meteorological-observations`, version `1.0`). The
#   version matters: the API paths contain both.
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
#   (e.g. `..._20260601.nc` = 2026-06-01).
# - This dataset holds **daily** values per station. The variable codes are the
#   classic KNMI "etmaalwaarden" codes: **TG** = daily mean temperature (°C),
#   **RH** = daily precipitation sum (mm), **EV24** = Makkink reference
#   evaporation (mm), plus TN/TX (min/max temperature), FG (mean wind), etc.

# %% [markdown]
# ## The two APIs at a glance
#
# | | Open Data API | EDR API |
# |---|---|---|
# | Base URL | `api.dataplatform.knmi.nl/open-data/v1` | `api.dataplatform.knmi.nl/edr/v1` |
# | Model | list *files* in a dataset, get a presigned download URL per file | query a *collection* by location/time/parameters |
# | Typical use | bulk download of whole files / long history | targeted extracts: one station, one period, a few parameters |
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
# | Open Data | `api_key=` | `KNMI_OPENDATA_API_KEY` (+`_FILE`), then the public anonymous key |
# | EDR | `edr_key=` | `KNMI_EDR_API_KEY` (+`_FILE`) |
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
# 1. **`.env` file (gitignored) + python-dotenv.** Create a `.env` file (see the README Authentication section)
#    to `.env`, fill in your keys, `chmod 600 .env`. `.env` is in
#    `.gitignore`, so it is never committed. In the notebook, only load it:
#    `load_dotenv()` — the key itself never appears in the notebook.
#    Install with `pip install knmidata[dotenv]`.
# 2. **Shell environment variables**: `export KNMI_OPENDATA_API_KEY=...` /
#    `export KNMI_EDR_API_KEY=...` in your profile or before launching Jupyter.
# 3. **Key file**: set `KNMI_OPENDATA_API_KEY_FILE=/path/to/key.txt` (chmod 600)
#    instead of the key itself; `knmidata` reads the file for you.
# 4. **GitHub Actions**: store the key as a repository *secret* and pass it
#    to the workflow step as an env var (see `.github/workflows/notebooks.yaml`,
#    which executes this notebook in CI) — GitHub masks secret values in logs.
#
# Additional hygiene: `knmidata` never prints or logs the key (check with
# `bool(os.environ.get("KNMI_OPENDATA_API_KEY"))`, not by printing it); when sharing an
# executed `.ipynb`, strip outputs first (the repo's pre-commit already runs
# `nbstripout`); if a key ever does leak, revoke/rotate it in the
# [developer portal](https://developer.dataplatform.knmi.nl/).

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import knmidata

# The dataset from the KNMI Data Platform page linked above.
DATASET_NAME = "daily-in-situ-meteorological-observations"
DATASET_VERSION = "1.0"
COLLECTION = "daily-in-situ-meteorological-observations"  # EDR collection id

# The variables we plot below (classic KNMI daily "etmaalwaarden" codes):
# RH = daily precipitation sum (mm), TG = daily mean temperature (degC),
# EV24 = Makkink reference evaporation (mm).
PARAMETERS = ["RH", "TG", "EV24"]

# De Bilt (KNMI's own station, WSI 0-20000-0-06260), as (lon, lat) — the
# station's grid point in the EDR responses.
STATION_COORDS = (5.18, 52.10)
# Bounding box over the Netherlands: (minx, miny, maxx, maxy).
NL_BBOX = (3.0, 50.5, 7.5, 53.7)

# Time arguments accept str, pd.Timestamp, datetime, or date; naive = UTC.
# One week of daily values (change to e.g. a month for a longer time series).
START = pd.Timestamp("2026-06-01")
END = pd.Timestamp("2026-06-08")

# Downloaded Open Data files are kept here (omit cache_dir to load into
# memory only and delete the files afterwards).
CACHE_DIR = Path("data")

knmidata.show_versions()
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

# Load keys from the gitignored .env file (see the README Authentication section).
# The key itself never appears in this notebook. On GitHub Actions, skip the
# .env file: store the key as a repository *secret* and expose it to the
# workflow as an env var (see .github/workflows/notebooks.yaml) — GitHub
# masks secret values in logs.
try:
    from dotenv import load_dotenv

    load_dotenv(".env")
except ImportError:
    pass  # alternative: export KNMI_OPENDATA_API_KEY / KNMI_EDR_API_KEY in your shell

# Presence check only — never print the key itself.
print("KNMI_OPENDATA_API_KEY set:", bool(os.environ.get("KNMI_OPENDATA_API_KEY")))
print("KNMI_EDR_API_KEY set:", bool(os.environ.get("KNMI_EDR_API_KEY")))

try:
    client = knmidata.KNMIClient(
        api_key=os.environ.get("KNMI_OPENDATA_API_KEY") or None,
        edr_key=os.environ.get("KNMI_EDR_API_KEY") or None,
    )  # keys are resolved from the environment
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

files = client.opendata.list_files(
    DATASET_NAME,
    DATASET_VERSION,
    start_time=START,
    end_time=END,
)
print(f"{len(files)} files between {START} and {END}: {files}")


# %% [markdown]
# Now download just the first two files of that period and open them as a
# single dataset. Each **daily** file contains all stations for that day, so
# even `max_files=2` already gives you a 2-day, all-stations slice.
#
# `start_time`/`end_time` are normalized to UTC ISO-8601 (`...Z`) before being
# sent — passing `pd.Timestamp` objects is fine.

# %%
ds = client.opendata.open_dataset(
    DATASET_NAME,
    DATASET_VERSION,
    start_time=START,
    end_time=END,
    cache_dir=CACHE_DIR,
)

print(f"\nDownloaded files are cached in: {CACHE_DIR.resolve()}")
print("\nCombined dataset:")
print(ds)
print("\nVariables in the dataset:", list(ds.data_vars))

# %% [markdown]
# ## 2. EDR API — query instead of download
#
# The EDR API (OGC Environmental Data Retrieval standard) exposes datasets as
# **collections**:
#
# - `GET /collections` / `GET /collections/{id}` — what's available: which
#   parameters (variables, under the `parameter-names` key), which output
#   formats, the spatial/temporal extent.
# - `GET /collections/{id}/locations` — the measurement locations (GeoJSON).
# - Data queries, with `start_time`/`end_time` (RFC 3339 interval),
#   `parameter-name` and `f` (CoverageJSON/GeoJSON) query parameters.
#
# Instead of files, you get a structured JSON payload describing exactly the
# queried cube of data, which `knmidata` parses into an `xarray.Dataset`
# in memory. For station collections there are **three ways** to pick the
# station(s) — shown below:
#
# | Method | Call | When to use |
# |---|---|---|
# | by station id | `client.edr.location_data(COLLECTION, location_id, ...)` | you know the station id (WSI) |
# | nearest to a point | `client.edr.nearest_location(COLLECTION, coords)` → id, then `location_data`; or `client.edr.position(..., coords=...)` (server-side, tolerance-limited) | you have coordinates |
# | area / bbox | `client.edr.area(COLLECTION, bbox=...)` | you want all stations in a region |

# %%

collections = client.edr.collections()

print(
    f"Available EDR collections: {[c['id'] for c in collections.get('collections', [])]}"
)

# Collection metadata: description, extent, and the available parameters.
# (The wire key is "parameter-names"; fall back to other spellings.)
meta = client.edr.collection(COLLECTION)
params_meta = (
    meta.get("parameter-names")
    or meta.get("parameter_names")
    or meta.get("parameters")
    or {}
)
print(
    f"\nCollection '{COLLECTION}': {len(params_meta)} parameters, e.g. "
    + ", ".join(list(params_meta)[:8])
)
# Check that the variables we want are actually in this collection.
missing = [p for p in PARAMETERS if p not in params_meta]
print(f"Requested {PARAMETERS}; missing from collection: {missing or 'none'}")


# %% [markdown]
# ### Method 1: by station id — `location_data`
#
# `/collections/{id}/locations/{location_id}` with `start_time`/`end_time` and
# `parameter-name` returns one station's time series (a CoverageCollection with
# a single coverage). We ask for **RH** (precipitation sum), **TG** (mean
# temperature) and **EV24** (Makkink evaporation).

# %%
de_bilt_id = "06260"  # fallback: WMO id of De Bilt
ts = client.edr.location_data(
    COLLECTION,
    de_bilt_id,
    start_time=START,
    end_time=END,
    parameters=PARAMETERS,
)
print(f"Daily time series at station {de_bilt_id}:")
print(ts)

# %% [markdown]
# ### Plot the EDR time series
#
# The parsed dataset is a plain `xarray.Dataset`, so plotting is one line per
# variable (needs `matplotlib`, included in the `notebooks` extra). If the
# Open Data download above succeeded, its values for the same station are
# overlaid for comparison.

# %%
fig, axes = plt.subplots(
    len(PARAMETERS),
    1,
    figsize=(9, 2.5 * len(PARAMETERS)),
    sharex=True,
    layout="tight",
)
for ax, var in zip(axes, PARAMETERS, strict=True):
    # drop singleton dims (x, y, station, coverage) so a 1D time series remains
    da = ts[var].squeeze(drop=True)
    for d in list(da.dims):
        if d != "time" and da.sizes[d] > 1:
            da = da.isel({d: 0})
    da.plot(ax=ax, marker="o", label="EDR")
    das = ds.sel(station=de_bilt_id)[var]  # WSI, e.g. '0-20000-0-06260'
    das.squeeze(drop=True).plot(ax=ax, marker="x", color="C1", label="Open Data")
    ax.legend()
    ax.set_title(
        f"{var} — {da.attrs.get('long_name', var)} ({da.attrs.get('units', '?')})"
    )
    ax.grid(alpha=0.3)
fig.suptitle(f"Daily values at De Bilt, {START.date()} to {END.date()}")
plt.show()

# %% [markdown]
# ### Method 2: nearest station to coordinates
#
# `/locations` returns a GeoJSON FeatureCollection — one feature per station,
# with coordinates and metadata (the station id is the feature `id`, e.g. the
# WSI `0-20000-0-06260` for De Bilt). `nearest_location()` picks the closest
# station client-side (haversine over `/locations`) and always answers.
# `position(..., coords=...)` is the server-side alternative: it matches the
# nearest station too, but only within a tolerance and 404s ("no stations")
# when nothing is close — shown below for comparison.

# %%
locations = client.edr.locations(COLLECTION)
features = locations.get("features", [])
print(f"{len(features)} measurement locations; first 3:")
for feat in features[:3]:
    props = feat.get("properties", {})
    name = props.get("name") or props.get("label")
    print(f"  id={feat.get('id')}: {name} at {feat['geometry']['coordinates']}")

nearest = client.edr.nearest_location(COLLECTION, coords=STATION_COORDS)
de_bilt_id = str(nearest["id"])
print(
    f"Nearest to {STATION_COORDS}: {nearest['name']} (id={de_bilt_id}, "
    f"{nearest['distance_km']:.2f} km)"
)

ts_position = client.edr.position(
    COLLECTION,
    coords=STATION_COORDS,
    start_time=START,
    end_time=END,
    parameters=PARAMETERS,
)
print("\n/position query (server-side nearest):")
print(ts_position)
# %% [markdown]
# ### Method 3: area — all stations in a bounding box
#
# `area` takes a bbox and returns every station inside it (a CoverageCollection
# of per-station coverages). `knmidata` merges them along a `station` dimension
# whose coordinate values are the station ids (WSI, e.g. '0-20000-0-06260') with
# `lon`/`lat` coordinates, so you can always tell which entry belongs to which
# station: `area["TG"].sel(station="0-20000-0-06260")`.

# %%

area = client.edr.area(
    COLLECTION,
    bbox=NL_BBOX,
    start_time=START,
    end_time=END,
    parameters=PARAMETERS,
)
print("All stations within the NL bounding box:")
print(area)

# %% [markdown]
# ## Open Data vs EDR — what did we just see?
#
# - **Same dataset, two access patterns.** Open Data gave us whole daily
#   *files* (all stations, all variables in the file) assembled along `time`;
#   EDR gave us exactly the queried subset (3 variables, 1 station or all
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
#   plot (as above), or write to disk.

# %%
client.close()
print("Client closed.")
