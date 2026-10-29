"""knmidata: ."""

# ruff: noqa: F401
import logging

from ._version import __version__, show_versions

logging.getLogger(__name__).addHandler(logging.NullHandler())


__all__ = [
    "show_versions",
]