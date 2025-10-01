"""Module for reporting the version of dbscan1d."""

from contextlib import suppress
from importlib.metadata import PackageNotFoundError, version

__version__ = "0.0.0"
with suppress(PackageNotFoundError):
    __version__ = version("dbscan1d")
