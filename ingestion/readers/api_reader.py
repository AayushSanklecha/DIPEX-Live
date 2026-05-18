"""
ingestion/readers/api_reader.py
---------------------------------
Universal API-based data reader.

Supported sources
-----------------
REST endpoints (GET / POST / PUT)
GraphQL endpoints
Paginated APIs (page/offset, cursor, Link-header)
OAuth2 (client_credentials grant)
API key (header / query param)
Bearer token
Webhook payload parsing

Design contracts
----------------
- Exponential backoff with jitter on 429 / 5xx (configurable max_retries)
- Rate-limit awareness: reads Retry-After header
- Timeout on every request
- API response JSON schema validation (jsonschema, optional)
- Partial response handling: logs and continues
- Correlation ID on every request
- All errors → typed IntakeError — never crash
"""

from __future__ import annotations

import json
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import pandas as pd

from ingestion.error_handler import (
    APIResponseError, APITimeoutError, DataFormatError,
    ErrorAggregator, PartialDataError,
)

logger = logging.getLogger("dipex.ingestion.readers.api")

# ── Config Structures ─────────────────────────────────────────────────────────

@dataclass
class AuthConfig:
    """Unified auth config for any API auth strategy."""
    strategy: str = "none"           # none | api_key | bearer | oauth2
    api_key: Optional[str] = None
    api_key_header: str = "X-API-Key"
    bearer_token: Optional[str] = None
    oauth2_token_url: Optional[str] = None
    oauth2_client_id: Optional[str] = None
    oauth2_client_secret: Optional[str] = None
    oauth2_scope: Optional[str] = None


@dataclass
class PaginationConfig:
    """Pagination strategy for paginated APIs."""
    strategy: str = "none"           # none | page | offset | cursor | link
    page_param: str = "page"
    page_size_param: str = "page_size"
    page_size: int = 100
    max_pages: int = 1000
    cursor_field: str = "next_cursor"
    data_path: str = ""              # dot-separated path to records list in response


@dataclass
class APISourceConfig:
    """Full configuration for an API data source."""
    url: str
    method: str = "GET"              # GET | POST | PUT
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    body: Optional[Dict[str, Any]] = None
    auth: AuthConfig = field(default_factory=AuthConfig)
    pagination: PaginationConfig = field(default_factory=PaginationConfig)
    timeout_s: float = 30.0
    max_retries: int = 3
    backoff_base: float = 2.0
    data_path: str = ""              # dot-path to records in response
    schema_validator: Optional[Dict] = None   # JSON Schema dict
    is_graphql: bool = False
    graphql_query: Optional[str] = None
    graphql_variables: Optional[Dict] = None


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class APIReadResult:
    data: pd.DataFrame
    row_count: int
    pages_fetched: int
    total_requests: int
    errors: List = field(default_factory=list)
    response_schema_valid: bool = True
    read_time_ms: float = 0.0


# ── Auth Provider ─────────────────────────────────────────────────────────────

class AuthProvider:
    """Provides auth headers / params for different strategies."""

    def __init__(self, auth: AuthConfig) -> None:
        self.auth = auth
        self._oauth2_token: Optional[str] = None
        self._oauth2_expiry: float = 0.0

    def get_headers(self) -> Dict[str, str]:
        a = self.auth
        if a.strategy == "api_key":
            return {a.api_key_header: a.api_key or ""}
        if a.strategy == "bearer":
            return {"Authorization": f"Bearer {a.bearer_token or ''}"}
        if a.strategy == "oauth2":
            token = self._get_oauth2_token()
            return {"Authorization": f"Bearer {token}"}
        return {}

    def _get_oauth2_token(self) -> str:
        """Fetch/refresh OAuth2 client_credentials token."""
        if time.time() < self._oauth2_expiry - 30 and self._oauth2_token:
            return self._oauth2_token
        try:
            import requests
            resp = requests.post(
                self.auth.oauth2_token_url or "",
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.auth.oauth2_client_id,
                    "client_secret": self.auth.oauth2_client_secret,
                    "scope": self.auth.oauth2_scope or "",
                },
                timeout=10,
            )
            resp.raise_for_status()
            token_data = resp.json()
            self._oauth2_token = token_data.get("access_token", "")
            self._oauth2_expiry = time.time() + token_data.get("expires_in", 3600)
            return self._oauth2_token or ""
        except Exception as exc:  # noqa: BLE001
            raise APIResponseError(f"OAuth2 token fetch failed: {exc}") from exc


# ── API Reader ────────────────────────────────────────────────────────────────

class APIReader:
    """
    Universal API reader with retry, pagination, auth, and schema validation.

    Usage::

        cfg = APISourceConfig(url="https://api.example.com/data", ...)
        reader = APIReader()
        result = reader.read(cfg)
        df = result.data
    """

    def read(self, config: APISourceConfig) -> APIReadResult:
        t0 = time.perf_counter()
        errors = ErrorAggregator()
        pages_fetched = 0
        total_requests = 0

        try:
            import uuid as _uuid
            from ingestion.chunked_writer import ChunkedParquetWriter
            run_id = _uuid.uuid4().hex[:12]
            writer = ChunkedParquetWriter(
                base_dir="data/tmp",
                dataset_id="api_stream",
                run_id=run_id,
            )
        except Exception as exc:
            logger.warning("Could not initialize ChunkedParquetWriter (fallback to memory): %s", exc)
            writer = None
        records_buf = []

        # ── [RL] Adaptive backoff selection ──────────────────────────────────
        try:
            from ingestion.adaptive_rate_limiter import get_rl_agent as _get_rl
            _rl_agent = _get_rl()
            config.backoff_base = _rl_agent.get_api_backoff(config.url)
            logger.debug("[RL] API backoff_base=%.1f for %s", config.backoff_base, config.url)
        except Exception:  # noqa: BLE001
            _rl_agent = None

        try:
            import requests
        except ImportError:
            raise DataFormatError("requests library not installed — run: pip install requests")

        auth_provider = AuthProvider(config.auth)
        session = requests.Session()

        def _request(url: str, params: Dict, body: Any) -> requests.Response:
            nonlocal total_requests
            total_requests += 1
            headers = {**config.headers, **auth_provider.get_headers()}
            headers["X-Correlation-ID"] = str(uuid.uuid4())
            headers["User-Agent"] = "DIPEX/3.0 DataIntake"

            for attempt in range(config.max_retries + 1):
                try:
                    if config.method.upper() == "POST" or config.is_graphql:
                        resp = session.post(url, json=body, headers=headers,
                                            params=params, timeout=config.timeout_s)
                    else:
                        resp = session.request(
                            config.method.upper(), url,
                            headers=headers, params=params, timeout=config.timeout_s,
                        )

                    # Rate limit
                    if resp.status_code == 429:
                        retry_after = float(resp.headers.get("Retry-After", 2 ** attempt))
                        logger.warning("Rate limited. Sleeping %.1fs (attempt %d)", retry_after, attempt + 1)
                        time.sleep(retry_after)
                        continue

                    # Server errors — retry with backoff
                    if resp.status_code >= 500:
                        wait = config.backoff_base ** attempt + random.uniform(0, 1)
                        logger.warning("HTTP %d — retrying in %.1fs (attempt %d)", resp.status_code, wait, attempt + 1)
                        time.sleep(wait)
                        continue

                    return resp

                except Exception as exc:  # noqa: BLE001 (requests.Timeout, ConnectionError etc.)
                    if "Timeout" in type(exc).__name__:
                        wait = config.backoff_base ** attempt + random.uniform(0, 1)
                        logger.warning("Timeout (attempt %d) — retrying in %.1fs", attempt + 1, wait)
                        time.sleep(wait)
                    else:
                        raise APITimeoutError(f"Request failed: {exc}") from exc

            raise APITimeoutError(
                f"API {config.url} failed after {config.max_retries + 1} attempts."
            )

        # Build request body for GraphQL
        request_body = config.body
        if config.is_graphql:
            request_body = {
                "query": config.graphql_query or "",
                "variables": config.graphql_variables or {},
            }

        params = dict(config.params)
        if config.pagination.strategy == "page":
            params[config.pagination.page_size_param] = config.pagination.page_size

        cursor = None
        page = 1

        while True:
            if config.pagination.strategy == "page":
                params[config.pagination.page_param] = page
            elif config.pagination.strategy == "offset":
                params[config.pagination.page_param] = (page - 1) * config.pagination.page_size
                params[config.pagination.page_size_param] = config.pagination.page_size
            elif config.pagination.strategy == "cursor" and cursor:
                params[config.pagination.cursor_field] = cursor

            try:
                resp = _request(config.url, params, request_body)
            except (APITimeoutError, APIResponseError) as exc:
                errors.add(exc.error_type, str(exc), severity="ERROR")
                break

            if not resp.ok:
                errors.add(
                    "API_RESPONSE_ERROR",
                    f"HTTP {resp.status_code}: {resp.text[:200]}",
                    severity="WARN",   # WARN not ERROR — try next page
                )
                # Skip this page, allow pagination to continue unless fatal
                if resp.status_code in (401, 403, 404):
                    break  # Auth/not-found errors are unrecoverable
                page += 1
                continue

            # ── Content-type aware parsing ─────────────────────────────
            content_type = resp.headers.get("Content-Type", "").lower()
            payload = None

            if "application/json" in content_type or "text/json" in content_type:
                try:
                    payload = resp.json()
                except Exception:  # noqa: BLE001
                    errors.add("DATA_FORMAT_ERROR",
                               f"Content-Type is JSON but parse failed: {resp.text[:200]}",
                               severity="WARN")
                    page += 1
                    continue

            elif "xml" in content_type:
                # XML API response — convert to DataFrame rows
                try:
                    import io as _io
                    xml_df = pd.read_xml(_io.StringIO(resp.text))
                    extracted = xml_df.to_dict(orient="records")
                    if writer:
                        records_buf.extend(extracted)
                    else:
                        records_buf.extend(extracted)
                    pages_fetched += 1
                    errors.add("QUALITY_WARN", "XML API response — parsed via pd.read_xml", severity="WARN")
                    break  # XML APIs typically return all data in one response
                except Exception as xml_exc:  # noqa: BLE001
                    errors.add("DATA_FORMAT_ERROR", f"XML parse failed: {xml_exc}", severity="WARN")
                    page += 1
                    continue

            elif "text/csv" in content_type:
                # CSV API response
                try:
                    import io as _io
                    csv_df = pd.read_csv(_io.StringIO(resp.text), encoding_errors="replace")
                    extracted = csv_df.to_dict(orient="records")
                    records_buf.extend(extracted)
                    pages_fetched += 1
                    errors.add("QUALITY_WARN", "CSV API response — parsed via pd.read_csv", severity="WARN")
                    break
                except Exception as csv_exc:  # noqa: BLE001
                    errors.add("DATA_FORMAT_ERROR", f"CSV parse failed: {csv_exc}", severity="WARN")
                    page += 1
                    continue

            else:
                # Try JSON first regardless of content-type (many APIs lie)
                try:
                    payload = resp.json()
                except Exception:  # noqa: BLE001
                    # Last resort: treat text as raw content row
                    raw_text = resp.text.strip()
                    if raw_text:
                        records_buf.append({"_raw_response": raw_text[:1000], "_page": page})
                        errors.add("QUALITY_WARN",
                                   f"Non-parseable response on page {page} — stored as _raw_response",
                                   severity="WARN")
                    page += 1
                    continue

            if payload is None:
                page += 1
                continue

            # Validate schema
            if config.schema_validator:
                try:
                    import jsonschema
                    jsonschema.validate(payload, config.schema_validator)
                except Exception as exc:  # noqa: BLE001
                    errors.add("SCHEMA_ERROR", f"API response schema invalid: {exc}", severity="WARN")

            # Extract records with auto-flattening of deeply nested payloads
            records = self._extract_records(payload, config.data_path)
            if records is None:
                errors.add("PARTIAL_DATA_ERROR",
                           f"Could not extract records from page {page} — skipping",
                           severity="WARN")
                page += 1
                continue  # Skip page, don't abort

            extracted = records if isinstance(records, list) else [records]

            # If records are dicts with nested structure, flatten up to 3 levels
            flat_extracted = []
            for rec in extracted:
                if isinstance(rec, dict):
                    try:
                        norm = pd.json_normalize([rec], max_level=3)
                        flat_extracted.extend(norm.to_dict(orient="records"))
                    except Exception:  # noqa: BLE001
                        flat_extracted.append(rec)
                else:
                    flat_extracted.append({"_value": rec})

            if writer:
                records_buf.extend(flat_extracted)
                if len(records_buf) >= config.pagination.page_size * 5:
                    try:
                        writer.write_chunk(pd.json_normalize(records_buf, max_level=3))
                    except Exception:
                        writer.write_chunk(pd.DataFrame(records_buf))
                    records_buf.clear()
            else:
                records_buf.extend(flat_extracted)

            pages_fetched += 1

            # Pagination control
            if config.pagination.strategy == "none":
                break
            if config.pagination.strategy in ("page", "offset"):
                # Empty page = end of data
                if len(extracted) == 0 or pages_fetched >= config.pagination.max_pages:
                    break
                page += 1
            elif config.pagination.strategy == "cursor":
                cursor = self._get_nested(payload, config.pagination.cursor_field)
                if not cursor:
                    break
            elif config.pagination.strategy == "link":
                link = resp.headers.get("Link", "")
                next_url = self._parse_link_header(link)
                if not next_url:
                    break
                config = APISourceConfig(**{**config.__dict__, "url": next_url,
                                            "pagination": PaginationConfig(strategy="none")})
            else:
                break

        # Flatten to DataFrame
        if writer:
            if records_buf:
                try:
                    writer.write_chunk(pd.json_normalize(records_buf))
                except Exception:
                    writer.write_chunk(pd.DataFrame(records_buf))
            df = writer.merge_to_single()
            writer.cleanup()
        else:
            if records_buf:
                try:
                    df = pd.json_normalize(records_buf)
                except Exception:
                    df = pd.DataFrame(records_buf)
            else:
                df = pd.DataFrame()

        elapsed = (time.perf_counter() - t0) * 1000
        session.close()

        # ── [RL] Record outcome for next-run learning ───────────────────────
        if _rl_agent is not None:
            try:
                _rl_agent.record_api_outcome(
                    config.url, config.backoff_base,
                    success=not df.empty,
                    latency_ms=elapsed,
                    has_errors=bool(errors.records),
                )
            except Exception:  # noqa: BLE001
                pass

        return APIReadResult(
            data=df,
            row_count=len(df),
            pages_fetched=pages_fetched,
            total_requests=total_requests,
            errors=errors.records,
            read_time_ms=round(elapsed, 2),
        )

    # ── Webhook payload ───────────────────────────────────────────────────────

    def parse_webhook(self, payload_bytes: bytes, data_path: str = "") -> pd.DataFrame:
        """Parse a raw webhook payload bytes into a DataFrame."""
        try:
            data = json.loads(payload_bytes.decode("utf-8", errors="replace"))
            records = self._extract_records(data, data_path)
            if isinstance(records, list):
                return pd.json_normalize(records)
            return pd.DataFrame([records])
        except Exception as exc:  # noqa: BLE001
            raise DataFormatError(f"Webhook payload parse failed: {exc}") from exc

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_records(payload: Any, data_path: str) -> Any:
        """Traverse dot-separated data_path to find records list. Auto-detects common keys if empty."""
        if not data_path:
            # Auto-unwrap common patterns if payload is a dict containing a list
            if isinstance(payload, dict):
                common_keys = ["results", "data", "items", "records", "users", "list"]
                for key in common_keys:
                    if key in payload and isinstance(payload[key], list):
                        return payload[key]
                # Fallback: if there's exactly one key and its value is a list, use it
                if len(payload) == 1:
                    val = next(iter(payload.values()))
                    if isinstance(val, list):
                        return val
            return payload
        for key in data_path.split("."):
            if isinstance(payload, dict):
                payload = payload.get(key)
            else:
                return None
            if payload is None:
                return None
        return payload

    @staticmethod
    def _get_nested(obj: Any, path: str) -> Any:
        """Get a nested value by dot-separated path."""
        for key in path.split("."):
            if isinstance(obj, dict):
                obj = obj.get(key)
            else:
                return None
        return obj

    @staticmethod
    def _parse_link_header(link: str) -> Optional[str]:
        """Parse RFC 5988 Link header for rel=next URL."""
        for part in link.split(","):
            url_part, *rels = part.strip().split(";")
            for rel in rels:
                if 'rel="next"' in rel or "rel=next" in rel:
                    return url_part.strip().strip("<>")
        return None
