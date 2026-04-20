from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("himga")
except PackageNotFoundError:
    __version__ = "unknown"
