"""Configure application logging for the RAG pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from src.core.config import Config


APPLICATION_LOGGER_NAME = "src"
DEFAULT_LOG_FILE_NAME = "rag_pipeline.log"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
MANAGED_HANDLER_ATTRIBUTE = "_rag_pipeline_handler"


def setup_logging(
    config: Config,
    log_file_name: str = DEFAULT_LOG_FILE_NAME,
) -> logging.Logger:
    """Configure console and file logging for project modules.

    Args:
        config: Loaded project configuration.
        log_file_name: Name of the log file created in the configured log directory.

    Returns:
        The configured application logger.
    """
    # Read logging settings
    log_level_name = str(config.get("logging", "level", default="INFO")).upper()
    log_level = _get_log_level(log_level_name)
    log_directory = config.log_dir

    # Prepare output location
    log_directory.mkdir(parents=True, exist_ok=True)

    # Configure application logger
    application_logger = logging.getLogger(APPLICATION_LOGGER_NAME)
    application_logger.setLevel(log_level)
    application_logger.propagate = False
    _remove_managed_handlers(application_logger)

    formatter = logging.Formatter(LOG_FORMAT)

    # Console output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    _mark_as_managed(console_handler)
    application_logger.addHandler(console_handler)

    # Persistent file output
    file_handler = logging.FileHandler(log_directory / log_file_name, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    _mark_as_managed(file_handler)
    application_logger.addHandler(file_handler)

    application_logger.info("Logging configured with level %s", log_level_name)
    return application_logger


def _get_log_level(log_level_name: str) -> int:
    """Return a logging level for a configured level name."""
    log_level = getattr(logging, log_level_name, None)
    if not isinstance(log_level, int):
        raise ValueError(f"Unsupported logging level: {log_level_name}")
    return log_level


def _remove_managed_handlers(application_logger: logging.Logger) -> None:
    """Remove handlers previously created by this module."""
    for handler in application_logger.handlers[:]:
        if not getattr(handler, MANAGED_HANDLER_ATTRIBUTE, False):
            continue

        application_logger.removeHandler(handler)
        handler.close()


def _mark_as_managed(handler: logging.Handler) -> None:
    """Mark a handler so repeated setup does not duplicate log output."""
    setattr(handler, MANAGED_HANDLER_ATTRIBUTE, True)
