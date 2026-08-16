"""Standalone Tkinter ML Research Studio — no chart server required."""

try:
    from __version__ import __version__, __app_name__
except ImportError:
    __version__ = "1.0.0"
    __app_name__ = "AruMLStudio"

__all__ = ["__version__", "__app_name__"]
