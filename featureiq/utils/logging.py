"""Logging configuration for FeatureIQ."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the FeatureIQ namespace.

    Args:
        name: Module name to create logger for.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(f"featureiq.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )
        logger.addHandler(handler)
    return logger
