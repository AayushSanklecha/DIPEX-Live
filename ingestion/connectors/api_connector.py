"""
ingestion/connectors/api_connector.py
----------------------------------------
Production REST / GraphQL API connector.

Features:
- OAuth2 Client Credentials + API key auth
- Rate limit awareness with configurable backoff
- Paginated response stitching (offset, cursor, page-number)
- Exponential backoff retry on 429 / 5xx
- GraphQL query support
- Response schema flattening (nested JSON)
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd

from .base_connector import BaseConnector, ConnectorError

logger = logging.getLogger("dipex.connectors.api")

_DEFAULT_TIMEOUT: int = 30
_DEFAULT_MAX_RETRIES: int = 4
_RETRYABLE_STATUS: tuple = (429, 500, 502, 503, 504)


class APIConnector(BaseConnector):
    """
    Universal REST / GraphQL connector.

    Config keys:
        base_url           : API base URL (required)
        endpoint           : API endpoint path (required for extract)
        auth_type          : "none" | "api_key" | "bearer" | "oauth2_cc" | "basic"
        api_key            : API key value (env: API_KEY)
        api_key_header     : Header name for API key (default: X-API-Key)
        bearer_token       : Bearer token (env: API_BEARER)
        oauth2_token_url   : OAuth2 token endpoint URL
        oauth2_client_id   : Client ID (env: OAUTH2_CLIENT_ID)
        oauth2_client_secret : Client secret (env: OAUTH2_CLIENT_SECRET)
        oauth2_scope       : OAuth2 scope (optional)
        username           : Basic auth username (env: API_USER)
        password           : Basic auth password (env: API_PASS)
        headers            : Additional static headers dict
        params             : Static query params dict
        pagination_type    : "none" | "offset" | "cursor" | "page"
        page_size          : Items per page (default: 100)
        max_pages          : Max pages to fetch (default: 100)
        data_path          : JSON path to data list (e.g. "data.items")
        cursor_path        : JSON path to next cursor
        graphql            : Bool — True to use GraphQL mode
        graphql_query      : GraphQL query string
        timeout            : Request timeout seconds (default: 30)
        max_retries        : Max retry attempts on failure (default: 4)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._session = None
        self._oauth2_token: Optional[str] = None
        self._oauth2_expiry: float = 0.0

    def _get_session(self):
        if self._session is None:
            try:
                import requests
                self._session = requests.Session()
                self._configure_auth(self._session)
            except ImportError as exc:
                raise ConnectorError("requests is required: pip install requests") from exc
        return self._session

    def _configure_auth(self, session) -> None:
        """Configure session authentication headers."""
        auth_type = self.config.get("auth_type", "none")

        if auth_type == "api_key":
            key = os.environ.get("API_KEY", self.config.get("api_key", ""))
            header = self.config.get("api_key_header", "X-API-Key")
            session.headers[header] = key

        elif auth_type == "bearer":
            token = os.environ.get("API_BEARER", self.config.get("bearer_token", ""))
            session.headers["Authorization"] = f"Bearer {token}"

        elif auth_type == "basic":
            from requests.auth import HTTPBasicAuth
            user = os.environ.get("API_USER", self.config.get("username", ""))
            pwd = os.environ.get("API_PASS", self.config.get("password", ""))
            session.auth = HTTPBasicAuth(user, pwd)

        elif auth_type == "oauth2_cc":
            self._refresh_oauth2_token(session)

        # Additional static headers
        for k, v in (self.config.get("headers") or {}).items():
            session.headers[k] = v

    def _refresh_oauth2_token(self, session=None) -> None:
        """Fetch OAuth2 token using client credentials grant."""
        if time.time() < self._oauth2_expiry - 60:
            return  # Token still valid

        import requests

        s = session or self._session
        token_url = self.config.get("oauth2_token_url")
        if not token_url:
            raise ConnectorError("APIConnector: oauth2_token_url required for oauth2_cc auth")

        client_id = os.environ.get("OAUTH2_CLIENT_ID", self.config.get("oauth2_client_id", ""))
        client_secret = os.environ.get("OAUTH2_CLIENT_SECRET", self.config.get("oauth2_client_secret", ""))
        scope = self.config.get("oauth2_scope", "")

        resp = requests.post(
            token_url,
            data={"grant_type": "client_credentials", "client_id": client_id,
                  "client_secret": client_secret, "scope": scope},
            timeout=_DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        self._oauth2_token = data["access_token"]
        expires_in = int(data.get("expires_in", 3600))
        self._oauth2_expiry = time.time() + expires_in

        if s is not None:
            s.headers["Authorization"] = f"Bearer {self._oauth2_token}"

    def test_connection(self) -> bool:
        try:
            session = self._get_session()
            base_url = self.config.get("base_url", "")
            resp = session.get(base_url, timeout=10)
            result = resp.status_code < 500
            logger.info("APIConnector: connection test %s (status=%d)", "PASSED" if result else "FAILED", resp.status_code)
            return result
        except Exception as exc:
            logger.error("APIConnector: connection test FAILED — %s", exc)
            return False

    def get_schema(self) -> Dict[str, Any]:
        """Infer schema by fetching one page of data."""
        try:
            df = self._fetch_one_page()
            if df.empty:
                return {"columns": [], "dtypes": {}, "estimated_row_count": -1}
            return {
                "endpoint": self.config.get("endpoint", ""),
                "columns": list(df.columns),
                "dtypes": {col: str(df[col].dtype) for col in df.columns},
                "estimated_row_count": -1,
                "description": f"Schema inferred from API response sample ({len(df)} rows)",
            }
        except Exception as exc:
            return {"error": str(exc), "columns": [], "dtypes": {}, "estimated_row_count": -1}

    def extract(self, query: Optional[str] = None, **kwargs: Any) -> pd.DataFrame:
        """
        Extract all pages from the API endpoint.
        `query` overrides the configured `endpoint` path.
        """
        if self.config.get("graphql"):
            return self._extract_graphql()

        endpoint = query or self.config.get("endpoint", "")
        pagination_type = self.config.get("pagination_type", "none")

        if pagination_type == "none":
            return self._fetch_one_page(endpoint)
        elif pagination_type == "offset":
            return self._fetch_offset_paginated(endpoint)
        elif pagination_type == "page":
            return self._fetch_page_paginated(endpoint)
        elif pagination_type == "cursor":
            return self._fetch_cursor_paginated(endpoint)
        else:
            return self._fetch_one_page(endpoint)

    def stream(self, chunk_size: int = 100, **kwargs: Any) -> Iterator[pd.DataFrame]:
        """Yield one DataFrame per API page."""
        pagination_type = self.config.get("pagination_type", "none")
        endpoint = self.config.get("endpoint", "")

        if pagination_type in ("none", "none"):
            yield self._fetch_one_page(endpoint)
            return

        for page_df in self._iter_pages(endpoint):
            yield page_df

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _request_with_retry(self, method: str, url: str, **kwargs) -> Any:
        """HTTP request with exponential backoff retry."""
        session = self._get_session()
        max_retries = self.config.get("max_retries", _DEFAULT_MAX_RETRIES)
        timeout = self.config.get("timeout", _DEFAULT_TIMEOUT)

        for attempt in range(1, max_retries + 1):
            try:
                resp = session.request(method, url, timeout=timeout, **kwargs)
                if resp.status_code == 401 and self.config.get("auth_type") == "oauth2_cc":
                    self._refresh_oauth2_token()
                    continue
                if resp.status_code in _RETRYABLE_STATUS and attempt < max_retries:
                    delay = 2 ** (attempt - 1)
                    if resp.status_code == 429:
                        retry_after = int(resp.headers.get("Retry-After", delay))
                        delay = max(delay, retry_after)
                    logger.warning("APIConnector: status %d, retry %d in %ds", resp.status_code, attempt, delay)
                    time.sleep(delay)
                    continue
                resp.raise_for_status()
                return resp
            except Exception as exc:
                if attempt == max_retries:
                    raise ConnectorError(f"APIConnector: request failed after {attempt} attempts — {exc}") from exc
                time.sleep(2 ** (attempt - 1))
        raise ConnectorError("APIConnector: max retries exhausted")

    def _fetch_one_page(self, endpoint: Optional[str] = None) -> pd.DataFrame:
        """Fetch a single page and flatten response."""
        base_url = self.config.get("base_url", "").rstrip("/")
        ep = (endpoint or self.config.get("endpoint", "")).lstrip("/")
        url = f"{base_url}/{ep}" if ep else base_url
        params = dict(self.config.get("params") or {})
        resp = self._request_with_retry("GET", url, params=params)
        return self._parse_response(resp)

    def _fetch_offset_paginated(self, endpoint: str) -> pd.DataFrame:
        """Fetch all pages using offset/limit pagination."""
        base_url = self.config.get("base_url", "").rstrip("/")
        ep = endpoint.lstrip("/")
        url = f"{base_url}/{ep}"
        page_size = self.config.get("page_size", 100)
        max_pages = self.config.get("max_pages", 100)
        frames = []
        offset = 0

        for _ in range(max_pages):
            params = {**dict(self.config.get("params") or {}),
                      "limit": page_size, "offset": offset}
            resp = self._request_with_retry("GET", url, params=params)
            df = self._parse_response(resp)
            if df.empty:
                break
            frames.append(df)
            if len(df) < page_size:
                break
            offset += page_size

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _fetch_page_paginated(self, endpoint: str) -> pd.DataFrame:
        """Fetch all pages using page number pagination."""
        frames = list(self._iter_pages(endpoint))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _iter_pages(self, endpoint: str) -> Iterator[pd.DataFrame]:
        base_url = self.config.get("base_url", "").rstrip("/")
        ep = endpoint.lstrip("/")
        url = f"{base_url}/{ep}"
        page_size = self.config.get("page_size", 100)
        max_pages = self.config.get("max_pages", 100)
        page = 1

        for _ in range(max_pages):
            params = {**dict(self.config.get("params") or {}),
                      "per_page": page_size, "page": page}
            resp = self._request_with_retry("GET", url, params=params)
            df = self._parse_response(resp)
            if df.empty:
                return
            yield df
            if len(df) < page_size:
                return
            page += 1

    def _fetch_cursor_paginated(self, endpoint: str) -> pd.DataFrame:
        """Fetch all pages using cursor-based pagination."""
        base_url = self.config.get("base_url", "").rstrip("/")
        ep = endpoint.lstrip("/")
        url = f"{base_url}/{ep}"
        cursor_path = self.config.get("cursor_path", "next_cursor")
        page_size = self.config.get("page_size", 100)
        max_pages = self.config.get("max_pages", 100)
        frames = []
        cursor = None

        for _ in range(max_pages):
            params = {**dict(self.config.get("params") or {}), "limit": page_size}
            if cursor:
                params["cursor"] = cursor
            resp = self._request_with_retry("GET", url, params=params)
            df, raw = self._parse_response(resp, return_raw=True)
            if df.empty:
                break
            frames.append(df)
            cursor = self._extract_path(raw, cursor_path)
            if not cursor:
                break

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _extract_graphql(self) -> pd.DataFrame:
        """Execute a GraphQL query."""
        gql_query = self.config.get("graphql_query", "")
        if not gql_query:
            raise ConnectorError("APIConnector: graphql_query must be set in config for GraphQL mode")
        base_url = self.config.get("base_url", "")
        payload = {"query": gql_query, "variables": self.config.get("graphql_variables", {})}
        resp = self._request_with_retry("POST", base_url, json=payload)
        return self._parse_response(resp)

    def _parse_response(self, resp, return_raw: bool = False):
        """Parse HTTP response into DataFrame."""
        try:
            data = resp.json()
        except Exception:
            data = {"raw": resp.text}

        data_path = self.config.get("data_path")
        if data_path:
            data = self._extract_path(data, data_path) or data

        if isinstance(data, list):
            df = pd.json_normalize(data)
        elif isinstance(data, dict):
            for key in ("data", "items", "results", "records", "rows"):
                if key in data and isinstance(data[key], list):
                    df = pd.json_normalize(data[key])
                    if return_raw:
                        return df, data
                    return df
            df = pd.json_normalize([data])
        else:
            df = pd.DataFrame()

        if return_raw:
            return df, data
        return df

    @staticmethod
    def _extract_path(data: Any, path: str) -> Any:
        """Extract nested value using dot-notation path."""
        parts = path.split(".")
        for part in parts:
            if isinstance(data, dict) and part in data:
                data = data[part]
            else:
                return None
        return data

    def close(self) -> None:
        if self._session:
            self._session.close()
            self._session = None
