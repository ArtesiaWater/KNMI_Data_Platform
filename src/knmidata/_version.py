"""Version information for knmidata."""

from importlib.metadata import PackageNotFoundError, version
from platform import python_version

try:
    __version__ = version("knmidata")
except PackageNotFoundError:  # running from a source checkout without installation
    __version__ = "0.0.1"


def _dist_version(dist: str) -> str:
    """Return the installed version of a distribution, or a placeholder."""
    try:
        return version(dist)
    except PackageNotFoundError:
        return "not installed"


def get_versions() -> dict[str, str]:
    """Collect versions of knmidata and its runtime dependencies."""
    return {
        "python": python_version(),
        "knmidata": __version__,
        "httpx": _dist_version("httpx"),
        "xarray": _dist_version("xarray"),
        "h5netcdf": _dist_version("h5netcdf"),
    }


def show_versions() -> None:
    """Print versions of knmidata and its runtime dependencies."""
    print("Versions:")
    for key, value in get_versions().items():
        print(f"  {key}: {value}")
