# KNMI_Data_Platform

`knmidata` is a lightweight Python client designed to fetch observational and gridded meteorological data from the [KNMI Data Platform](https://dataplatform.knmi.nl/) directly into standard `xarray.Dataset` objects.

## Supported APIs
- **Open Data API (/open-data/v1)**: Accesses KNMI's bulk file archives (e.g., radar precipitation composites, Harmonie model runs, satellite imagery). It handles dataset version resolution, cursor-based file pagination, AWS S3 presigned URL retrieval, and file streaming.
- **EDR API (/edr/v1)**: Implements OGC Environmental Data Retrieval for spatio-temporal filtering (/position, /area, /cube, /locations). It fetches in-situ station observations and spatial crops, parsing NetCDF streams or CoverageJSON directly into xarray.Dataset objects in memory without intermediate disk writes.
