"""Version information for knmidata."""

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from platform import python_version

try:
    __version__ = version("knmidata")
except PackageNotFoundError:
    # Not installed (e.g. running from a bare source checkout): read the
    # single source of truth in pyproject.toml.
    with (Path(__file__).resolve().parents[2] / "pyproject.toml").open("rb") as fh:
        __version__ = tomllib.load(fh)["project"]["version"]


def get_versions() -> dict[str, str]:
    """Collect versions of knmidata and its runtime dependencies."""
    return {
        "python": python_version(),
        "knmidata": __version__,
        "httpx": version("httpx"),
        "xarray": version("xarray"),
        "h5netcdf": version("h5netcdf"),
    }


def show_versions() -> None:
    """Print versions of knmidata and its runtime dependencies."""
    print("Versions:")
    for key, value in get_versions().items():
        print(f"  {key}: {value}")
