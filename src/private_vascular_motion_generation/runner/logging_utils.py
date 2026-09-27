"""Logging configuration for the study runners.

Ref: Sec. 4.9 (per-cell diagnostics are recorded rather than averaged across arms).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

LOGGER_NAME = "caps"
DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


@dataclass(frozen=True)
class LoggingConfig:
    level: int = logging.INFO
    log_file: Path | None = None


def configure_logging(config: LoggingConfig | None = None) -> logging.Logger:
    settings = config if config is not None else LoggingConfig()
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(settings.level)
    logger.handlers.clear()
    formatter = logging.Formatter(DEFAULT_FORMAT)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if settings.log_file is not None:
        settings.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(settings.log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def get_logger(suffix: str) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{suffix}")


def log_metrics(logger: logging.Logger, metrics: dict[str, float], prefix: str = "") -> None:
    rendered = " ".join(f"{key}={value:.6g}" for key, value in sorted(metrics.items()))
    logger.info("%s%s", f"{prefix} " if prefix else "", rendered)


def summarise_round(round_index: int, total: int, rho: float, loss: float) -> str:
    return f"round {round_index + 1}/{total} loss={loss:.6f} rho={rho:.4f}"
