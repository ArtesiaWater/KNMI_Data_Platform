"""Helpers for Open Data files: download metadata and xarray dataset assembly."""

from dataclasses import dataclass
from typing import Any, cast

import xarray as xr

from knmidata._exceptions import KNMIError
from knmidata._typing import CombineMethod, FilePaths


@dataclass(slots=True, kw_only=True)
class FileDownloadInfo:
    """Metadata and presigned URL for a single dataset file.

    Parameters
    ----------
    content_type : str
        MIME type of the file.
    size : int
        File size in bytes, as reported by the API.
    last_modified : str
        ISO-8601 timestamp of the last modification.
    url : str
        Temporary presigned download URL.
    """

    content_type: str
    size: int
    last_modified: str
    url: str

    def __repr__(self) -> str:
        """Repr that masks the presigned URL (it is a temporary credential)."""
        return (
            f"{type(self).__name__}(content_type={self.content_type!r}, "
            f"size={self.size}, last_modified={self.last_modified!r}, url='***')"
        )


def files_to_dataset(
    file_paths: FilePaths,
    engine: str = "h5netcdf",
    combine: CombineMethod = "by_coords",
    concat_dim: str | None = None,
    **open_dataset_kwargs: Any,
) -> xr.Dataset:
    """Assemble local NetCDF/HDF5 files into an xarray.Dataset.

    Implemented with `xr.open_dataset` + `xr.combine_by_coords` so that no dask
    installation is required.

    Parameters
    ----------
    file_paths : FilePaths
        Paths to the downloaded NetCDF files.
    engine : str, default="h5netcdf"
        Xarray backend engine.
    combine : CombineMethod, default="by_coords"
        How to combine the per-file datasets: 'by_coords' (combine_by_coords),
        'nested' (concat along `concat_dim`), or 'no_conflicts' (merge).
    concat_dim : str | None, default=None
        Dimension to concatenate along when combine='nested'.
    **open_dataset_kwargs : Any
        Additional options forwarded to `xr.open_dataset`.

    Returns
    -------
    xr.Dataset
        Combined multidimensional dataset.

    Raises
    ------
    KNMIError
        If `file_paths` is empty, or combine='nested' without `concat_dim`.
    """
    paths = [str(p) for p in file_paths]
    if not paths:
        raise KNMIError("files_to_dataset() requires at least one file path.")
    datasets = [xr.open_dataset(p, engine=engine, **open_dataset_kwargs) for p in paths]
    if combine == "by_coords":
        return cast("xr.Dataset", xr.combine_by_coords(datasets))
    if combine == "no_conflicts":
        return xr.merge(datasets, compat="no_conflicts")
    if concat_dim is None:
        raise KNMIError("combine='nested' requires concat_dim to be set.")
    return xr.concat(datasets, dim=concat_dim)
