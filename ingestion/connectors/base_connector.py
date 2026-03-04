"""
ingestion/connectors/base_connector.py
----------------------------------------
Abstract Base Class for all DIPEX data connectors.
Any new connector must implement this interface.
The pipeline is source-agnostic — it only interacts with BaseConnector.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional

import pandas as pd

logger = logging.getLogger("dipex.connectors.base")


class BaseConnector(ABC):
    """
    Abstract interface for all DIPEX data source connectors.

    Contract:
    - `extract()` always returns a clean pandas DataFrame
    - `test_connection()` must be called before `extract()`
    - `get_schema()` returns column metadata without loading all data
    - `stream()` is optional; yields DataFrame chunks for large sources
    - All credentials must come from the config dict (never hardcoded)
    - Failed connections must raise `ConnectorError`, not crash silently
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._logger = logging.getLogger(f"dipex.connectors.{self.__class__.__name__}")

    @abstractmethod
    def extract(self, query: Optional[str] = None, **kwargs: Any) -> pd.DataFrame:
        """
        Extract data and return as a DataFrame.

        Args:
            query : SQL query, API endpoint path, collection filter, etc.
            **kwargs : Source-specific options

        Returns:
            pd.DataFrame — always. Empty DF if no data.

        Raises:
            ConnectorError if connection or extraction fails.
        """

    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test connectivity to the data source.

        Returns:
            True if connection is healthy, False otherwise.
        """

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Return metadata about the data source schema without
        loading the full dataset.

        Returns:
            Dict with keys: columns (list), dtypes (dict),
                            estimated_row_count (int), description (str)
        """

    def stream(self, chunk_size: int = 10_000, **kwargs: Any) -> Iterator[pd.DataFrame]:
        """
        Optional streaming interface for large data sources.
        Override in connectors that support chunked reads.
        Default: extract once and yield as a single chunk.
        """
        df = self.extract(**kwargs)
        if df.empty:
            return
        for start in range(0, len(df), chunk_size):
            yield df.iloc[start: start + chunk_size].copy()

    def close(self) -> None:
        """Release any resources (connections, file handles, etc.)."""

    def __enter__(self) -> "BaseConnector":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config_keys={list(self.config.keys())})"


class ConnectorError(RuntimeError):
    """Raised when a connector cannot connect or extract data."""
