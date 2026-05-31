"""
Structured logging configuration for Store Intelligence API.
"""

import json
import logging
import sys
import time
import uuid

from datetime import datetime
from typing import Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings


class JsonFormatter(logging.Formatter):
    """
    Format log records as JSON.
    """

    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "service": "store-intelligence-api",
            "message": record.getMessage(),
        }

        if hasattr(record, "trace_id"):
            log_record["trace_id"] = record.trace_id

        if hasattr(record, "store_id"):
            log_record["store_id"] = record.store_id

        if hasattr(record, "endpoint"):
            log_record["endpoint"] = record.endpoint

        if hasattr(record, "latency_ms"):
            log_record["latency_ms"] = record.latency_ms

        if hasattr(record, "status_code"):
            log_record["status_code"] = record.status_code

        return json.dumps(log_record)


def setup_logging():
    """
    Configure application logging.
    """

    root_logger = logging.getLogger()
    root_logger.setLevel(
        getattr(
            logging,
            settings.log_level.upper(),
            logging.INFO,
        )
    )

    root_logger.handlers.clear()

    formatter = JsonFormatter()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        "store_intelligence.log"
    )
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


class StructuredLogger:
    """
    Wrapper around Python logger
    supporting trace IDs.
    """

    def __init__(
        self,
        name: str,
        trace_id: Optional[str] = None,
    ):
        self.logger = logging.getLogger(name)
        self.trace_id = trace_id

    def info(self, message: str, **kwargs):
        self.logger.info(
            message,
            extra={
                "trace_id": self.trace_id,
                **kwargs,
            },
        )

    def warning(self, message: str, **kwargs):
        self.logger.warning(
            message,
            extra={
                "trace_id": self.trace_id,
                **kwargs,
            },
        )

    def error(self, message: str, **kwargs):
        self.logger.error(
            message,
            extra={
                "trace_id": self.trace_id,
                **kwargs,
            },
        )

    def debug(self, message: str, **kwargs):
        self.logger.debug(
            message,
            extra={
                "trace_id": self.trace_id,
                **kwargs,
            },
        )


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Request/response logging middleware.
    """

    async def dispatch(
        self,
        request: Request,
        call_next,
    ):
        trace_id = str(uuid.uuid4())[:8]

        request.state.trace_id = trace_id

        start_time = time.perf_counter()

        logger = StructuredLogger(
            __name__,
            trace_id=trace_id,
        )

        logger.info(
            "Request started",
            endpoint=request.url.path,
        )

        try:
            response = await call_next(request)

            latency_ms = round(
                (
                    time.perf_counter()
                    - start_time
                )
                * 1000,
                2,
            )

            logger.info(
                "Request completed",
                endpoint=request.url.path,
                latency_ms=latency_ms,
                status_code=response.status_code,
            )

            response.headers[
                "X-Trace-ID"
            ] = trace_id

            return response

        except Exception as exc:
            latency_ms = round(
                (
                    time.perf_counter()
                    - start_time
                )
                * 1000,
                2,
            )

            logger.error(
                f"Request failed: {str(exc)}",
                endpoint=request.url.path,
                latency_ms=latency_ms,
                status_code=500,
            )

            raise