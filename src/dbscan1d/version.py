"""Module for reporting the version of dbscan1d."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dbscan1d")
# package is not installed
except PackageNotFoundError:  # NOQA
    __version__ = "0.0.0"  # NOQA
