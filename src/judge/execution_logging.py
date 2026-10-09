"""Single-line execution diagnostics, kept off the SSH JSON stdout channel."""

import json
import logging
import os
import re
import sys

LOGGER_NAME = "judge.execution"
logger = logging.getLogger(LOGGER_NAME)
OUTPUT_LIMIT = 8192


def configure_logging() -> None:
    """Configure only our logger; leave Uvicorn and SDK logging alone."""
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def safe_text(value: object) -> str:
    """Redact configured credentials before bounding captured command output."""
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    if not isinstance(value, str):
        return ""
    for name, secret in os.environ.items():
        if secret and (
            any(part in name for part in ("TOKEN", "SECRET", "PASSWORD", "KEY"))
            or name == "JUDGE_SLACK_WEBHOOK_URL"
        ):
            value = value.replace(secret, "[REDACTED]")
    value = re.sub(r"\b(?:hf_|ghp_|github_pat_)[A-Za-z0-9_]+", "[REDACTED]", value)
    encoded = value.encode()
    if len(encoded) > OUTPUT_LIMIT:
        return encoded[:OUTPUT_LIMIT].decode(errors="ignore") + " [truncated]"
    return value


def log_event(event: str, *, level: int = logging.INFO, **fields: object) -> None:
    if not logger.isEnabledFor(level):
        return
    details = " ".join(
        f"{name}={json.dumps(safe_text(value) if isinstance(value, (str, bytes)) else value, default=str)}"
        for name, value in fields.items()
    )
    logger.log(level, "%s %s", event, details)
