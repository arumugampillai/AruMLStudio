"""Schema migration framework for Feature Intelligence Core."""

from __future__ import annotations

from .runner import MigrationRunner, MigrationInfo

__all__ = ["MigrationInfo", "MigrationRunner"]
