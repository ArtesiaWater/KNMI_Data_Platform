"""Version information for knmidata."""

from importlib.metadata import version
from platform import python_version

__version__ = version("knmidata")


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
