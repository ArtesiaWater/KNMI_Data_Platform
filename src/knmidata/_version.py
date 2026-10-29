from importlib.metadata import version

from platform import python_version


__version__ = "0.0.1"

def get_versions() -> dict[str, str]:
    return {
        "python": python_version(),
        "knmidata": __version__,
        "httpx": version("httpx"),
        "xarray": version("xarray"),
        "h5netcdf": version("h5netcdf"),
    }


def show_versions() -> None:
    versions = get_versions()
    print("Versions:")
    for key, value in versions.items():
        print(f"  {key}: {value}")