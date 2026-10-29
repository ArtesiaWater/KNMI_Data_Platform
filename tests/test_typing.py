"""Tests that the shared type aliases are defined in _typing and exported."""

import knmidata
from knmidata import _typing


def test_aliases_defined_in_typing_module():
    for name in (
        "TimeLike",
        "EDRDatetime",
        "PathLike",
        "FilePaths",
        "Coords",
        "BBox",
        "ZLevel",
        "OutputFormat",
        "CombineMethod",
        "QueryParams",
        "AuthFlow",
        "RunCoroutine",
        "T",
    ):
        assert hasattr(_typing, name), name


def test_aliases_exported_at_package_level():
    for name in (
        "TimeLike",
        "EDRDatetime",
        "PathLike",
        "FilePaths",
        "Coords",
        "BBox",
        "ZLevel",
        "OutputFormat",
        "CombineMethod",
    ):
        assert name in knmidata.__all__
        assert getattr(knmidata, name) is getattr(_typing, name)


def test_time_module_reexports_typelike():
    from knmidata._time import TimeLike

    assert TimeLike is _typing.TimeLike
