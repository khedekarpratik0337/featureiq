"""Tests for featureiq.utils.logging module."""

from __future__ import annotations

import logging

from featureiq.utils.logging import get_logger


class TestGetLogger:
    """Tests for get_logger utility."""

    def test_returns_logger_with_correct_name(self) -> None:
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "featureiq.test_module"

    def test_logger_has_handler(self) -> None:
        logger = get_logger("handler_check")
        assert len(logger.handlers) >= 1
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)

    def test_calling_twice_does_not_duplicate_handlers(self) -> None:
        logger1 = get_logger("dedup_test")
        n_handlers = len(logger1.handlers)
        logger2 = get_logger("dedup_test")
        assert logger1 is logger2
        assert len(logger2.handlers) == n_handlers
