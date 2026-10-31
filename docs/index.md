# Documentation

`knmidata` is a lightweight Python client for the [KNMI Data Platform](https://dataplatform.knmi.nl/) that fetches observational and gridded meteorological data directly into standard `xarray.Dataset` objects.

Two APIs are supported:

- **Open Data API** — bulk NetCDF file archives (e.g. radar composites, in-situ observations), downloaded and assembled into one `xarray.Dataset`.
- **EDR API** — OGC EDR spatio-temporal queries (`/collections`, `/locations`, `/position`, `/area`, `/cube`); CoverageJSON/GeoJSON responses are parsed into in-memory `xarray.Dataset` objects.

## Authentication

All endpoints require an API key. Request a free key at [developer.dataplatform.knmi.nl](https://developer.dataplatform.knmi.nl/). KNMI issues a **separate key per API** — an Open Data key does not work for EDR and vice versa:

```bash
export KNMI_OPENDATA_API_KEY="<your open data key>"
export KNMI_EDR_API_KEY="<your edr key>"   # EDR has no anonymous key
```

For Open Data only, you can omit the key entirely and the public anonymous key is used automatically (shared rate limit).

## Contents

```{toctree}
:maxdepth: 2

example_daily_in_situ_observations
_api/modules.rst
```
