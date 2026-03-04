"""
ingestion/error_handler.py
-----------------------------
Production-grade error handling for the Universal Data Intake Layer.

Design rules:
  1. NEVER crash silently — all exceptions are caught, categorised, and logged.
  2. Every ingestion gets a correlation_id (UUID) for tracing.
  3. Errors are categorised into 7 typed exceptions with human-readable detail.
  4. All failures are written to the DIPEX audit log.
  5. safe_execute() wraps any reader call — returns (result, errors) tuple.
"""

from __future__ import annotations

import logging
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Generator, List, Optional, Tuple, TypeVar

from ingestion.issf import IngestionError

logger = logging.getLogger("dipex.ingestion.error_handler")

T = TypeVar("T")

# ── Typed Exception Hierarchy ─────────────────────────────────────────────────

class IntakeError(Exception):
    """Base class for all UDIL exceptions."""
    error_type = "INTAKE_ERROR"

    def __init__(self, message: str, correlation_id: Optional[str] = None,
                 column: Optional[str] = None, row_index: Optional[int] = None) -> None:
        super().__init__(message)
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.column         = column
        self.row_index      = row_index
        self.timestamp      = datetime.now(timezone.utc).isoformat()

    def to_ingestion_error(self) -> IngestionError:
        return IngestionError(
            error_type=self.error_type,
            message=str(self),
            row_index=self.row_index,
            column=self.column,
            severity="ERROR",
            correlation_id=self.correlation_id,
        )

    def human_readable(self) -> str:
        return (
            f"[{self.error_type}] {str(self)}\n"
            f"  Correlation ID : {self.correlation_id}\n"
            f"  Timestamp      : {self.timestamp}"
        )


class SchemaError(IntakeError):
    """Column missing, type change, or breaking schema drift detected."""
    error_type = "SCHEMA_ERROR"


class DataFormatError(IntakeError):
    """File format invalid, malformed rows, unsupported delimiter, etc."""
    error_type = "DATA_FORMAT_ERROR"


class EncodingError(IntakeError):
    """Could not decode file bytes — unknown or corrupt encoding."""
    error_type = "ENCODING_ERROR"


class APITimeoutError(IntakeError):
    """API request timed out after max retries."""
    error_type = "API_TIMEOUT_ERROR"


class APIResponseError(IntakeError):
    """API returned unexpected status code or malformed JSON."""
    error_type = "API_RESPONSE_ERROR"


class DBConnectionError(IntakeError):
    """Database connection failed or credentials rejected."""
    error_type = "DB_CONNECTION_ERROR"


class StreamLagError(IntakeError):
    """Consumer lag exceeded safe threshold — backpressure detected."""
    error_type = "STREAM_LAG_ERROR"


class PartialDataError(IntakeError):
    """Data received partially — truncated file or incomplete API response."""
    error_type = "PARTIAL_DATA_ERROR"


class QualityGateError(IntakeError):
    """Quality thresholds violated — ingestion blocked by quality gate."""
    error_type = "QUALITY_GATE_ERROR"


# ── Error Classifier ──────────────────────────────────────────────────────────

_ERROR_MAP = {
    "ConnectionRefusedError":    DBConnectionError,
    "OperationalError":          DBConnectionError,
    "InterfaceError":            DBConnectionError,
    "UnicodeDecodeError":        EncodingError,
    "UnicodeError":              EncodingError,
    "TimeoutError":              APITimeoutError,
    "ReadTimeout":               APITimeoutError,
    "ConnectTimeout":            APITimeoutError,
    "JSONDecodeError":           DataFormatError,
    "ParserError":               DataFormatError,
    "EmptyDataError":            DataFormatError,
    "SchemaInferenceError":      SchemaError,
}


def classify_exception(exc: Exception, correlation_id: str) -> IntakeError:
    """Wrap any Python exception in the appropriate typed IntakeError."""
    exc_type = type(exc).__name__
    cls = _ERROR_MAP.get(exc_type, IntakeError)
    wrapped = cls(
        message=f"{exc_type}: {str(exc)}",
        correlation_id=correlation_id,
    )
    return wrapped


# ── Safe Executor ─────────────────────────────────────────────────────────────

class SafeExecutor:
    """
    Context manager and utility class for safe ingestion execution.

    Usage::

        ex = SafeExecutor(dataset_id="sales", source_type="file")
        result, errors = ex.run(file_reader.read, "path/to/data.csv")
        if errors:
            for e in errors:
                print(e.human_readable())
    """

    def __init__(self, dataset_id: str = "", source_type: str = "file") -> None:
        self.dataset_id    = dataset_id
        self.source_type   = source_type
        self.correlation_id = str(uuid.uuid4())
        self._errors: List[IntakeError] = []

    @property
    def errors(self) -> List[IntakeError]:
        return list(self._errors)

    @property
    def has_errors(self) -> bool:
        return bool(self._errors)

    def run(
        self,
        fn: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Optional[T], List[IntakeError]]:
        """
        Execute `fn(*args, **kwargs)` safely.
        Returns (result, errors). Never raises.
        """
        try:
            result = fn(*args, **kwargs)
            return result, []
        except IntakeError as exc:
            exc.correlation_id = self.correlation_id
            self._errors.append(exc)
            logger.error(
                "[%s] Ingestion error (dataset=%s, source=%s): %s",
                self.correlation_id, self.dataset_id, self.source_type,
                exc.human_readable(),
            )
            self._audit_log(exc)
            return None, [exc]
        except Exception as exc:  # noqa: BLE001
            wrapped = classify_exception(exc, self.correlation_id)
            self._errors.append(wrapped)
            logger.error(
                "[%s] Unclassified exception in %s ingestion (dataset=%s):\n%s",
                self.correlation_id, self.source_type, self.dataset_id,
                traceback.format_exc(),
            )
            self._audit_log(wrapped)
            return None, [wrapped]

    @contextmanager
    def guard(self, operation: str = "") -> Generator[None, None, None]:
        """Context manager for wrapping code blocks (non-fatal guard)."""
        try:
            yield
        except IntakeError as exc:
            exc.correlation_id = self.correlation_id
            self._errors.append(exc)
            logger.warning("[%s] Guarded error in '%s': %s", self.correlation_id, operation, exc)
        except Exception as exc:  # noqa: BLE001
            wrapped = classify_exception(exc, self.correlation_id)
            self._errors.append(wrapped)
            logger.warning(
                "[%s] Guarded unclassified error in '%s': %s — %s",
                self.correlation_id, operation, type(exc).__name__, exc,
            )

    def _audit_log(self, exc: IntakeError) -> None:
        """Append structured error entry to the DIPEX audit log."""
        import json, os
        os.makedirs("audit", exist_ok=True)
        entry = {
            "event":          "INGESTION_ERROR",
            "correlation_id": self.correlation_id,
            "dataset_id":     self.dataset_id,
            "source_type":    self.source_type,
            "error_type":     exc.error_type,
            "message":        str(exc),
            "timestamp":      exc.timestamp,
        }
        with open("audit/audit.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


# ── Ingestion Error Aggregator ────────────────────────────────────────────────

class ErrorAggregator:
    """
    Collects soft errors (warnings, malformed rows) during ingestion
    without stopping the pipeline. Fatal errors still raise.
    """

    def __init__(self) -> None:
        self._records: List[IngestionError] = []

    def add(
        self,
        error_type: str,
        message: str,
        severity: str = "WARN",
        row_index: Optional[int] = None,
        column: Optional[str] = None,
    ) -> None:
        self._records.append(IngestionError(
            error_type=error_type, message=message,
            severity=severity, row_index=row_index, column=column,
        ))
        if severity == "ERROR":
            logger.error("Ingestion error [%s]: %s", error_type, message)
        else:
            logger.warning("Ingestion warning [%s]: %s", error_type, message)

    @property
    def records(self) -> List[IngestionError]:
        return list(self._records)

    @property
    def has_errors(self) -> bool:
        return any(r.severity == "ERROR" for r in self._records)

    @property
    def error_count(self) -> int:
        return len(self._records)
