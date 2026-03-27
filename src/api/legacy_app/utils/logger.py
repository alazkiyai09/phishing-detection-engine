"""
Structured logging configuration for the application.
"""
import logging
import sys
import json
from typing import Any, Dict
from datetime import datetime
import logging.handlers


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def setup_logger(
    name: str,
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: str = None
) -> logging.Logger:
    """
    Setup structured logger for the application.

    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type (json or text)
        log_file: Optional log file path

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Clear existing handlers
    logger.handlers.clear()

    # Choose formatter
    if log_format.lower() == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to avoid duplicate logs
    logger.propagate = False

    return logger


class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter for adding extra context to log messages.
    """

    def process(self, msg, kwargs):
        """
        Process log message with extra context.
        """
        # Add extra fields to kwargs
        if "extra" not in kwargs:
            kwargs["extra"] = {}

        kwargs["extra"].update(self.extra)

        return msg, kwargs


def get_logger(name: str, **context) -> logging.Logger:
    """
    Get a logger with optional context.

    Args:
        name: Logger name
        **context: Additional context fields

    Returns:
        Logger with context adapter
    """
    logger = logging.getLogger(name)

    if context:
        return LoggerAdapter(logger, context)

    return logger
