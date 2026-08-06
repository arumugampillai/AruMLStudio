"""Centralized logging for Feature Intelligence Core."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import LoggingConfig

_CONFIGURED = False
_LOGGER_NAME = "feature_intelligence"


def setup_logging(config: LoggingConfig | None = None) -> logging.Logger:
    """Configure the FIC root logger (idempotent for the process)."""
    global _CONFIGURED
    logger = logging.getLogger(_LOGGER_NAME)

    if config is None:
        from .config import LoggingConfig as _LoggingConfig

        config = _LoggingConfig()

    level = getattr(logging, config.level.upper(), logging.INFO)
    logger.setLevel(level)

    if _CONFIGURED and logger.handlers:
        return logger

    formatter = logging.Formatter(config.format)
    stream = logging.StreamHandler()
    stream.setLevel(level)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if config.to_file and config.file_path is not None:
        config.file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(config.file_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    _CONFIGURED = True
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger under ``feature_intelligence``."""
    if name is None or name == _LOGGER_NAME:
        return logging.getLogger(_LOGGER_NAME)
    if name.startswith(f"{_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")
