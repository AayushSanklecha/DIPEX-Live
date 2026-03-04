"""
ingestion/batch_loader.py
-------------------------
Handles loading data from static batch sources: CSV, Excel, JSON, and SQL.
"""

import csv
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


class BatchLoader:
    """Handles loading data from static batch sources (CSV, Excel, JSON, SQL)."""

    @staticmethod
    def load_csv(
        path: str,
        delimiter: Optional[str] = None,
        encoding: str = "utf-8",
    ) -> pd.DataFrame:
        """
        Loads a CSV file.

        Uses `csv.Sniffer` for automatic delimiter detection when `delimiter`
        is not provided, which is more accurate than manual heuristics.
        """
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        if delimiter is None:
            with open(resolved, "r", encoding=encoding, newline="") as f:
                sample = f.read(4096)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                    delimiter = dialect.delimiter
                except csv.Error:
                    delimiter = ","  # Safe fallback
            logger.debug("Auto-detected delimiter %r for %s", delimiter, path)

        df = pd.read_csv(resolved, sep=delimiter, encoding=encoding)
        logger.info("Loaded CSV: %s  rows=%d cols=%d", path, len(df), len(df.columns))
        return df

    @staticmethod
    def load_excel(path: str, sheet_name: int = 0) -> pd.DataFrame:
        """Loads an Excel file."""
        df = pd.read_excel(path, sheet_name=sheet_name)
        logger.info("Loaded Excel: %s  rows=%d cols=%d", path, len(df), len(df.columns))
        return df

    @staticmethod
    def load_json(path: str, orient: str = "records") -> pd.DataFrame:
        """Loads a JSON file."""
        df = pd.read_json(path, orient=orient)
        logger.info("Loaded JSON: %s  rows=%d cols=%d", path, len(df), len(df.columns))
        return df

    @staticmethod
    def load_sql(connection_string: str, query: str) -> pd.DataFrame:
        """
        Loads data from a SQL database.

        The SQLAlchemy engine is properly disposed after use to prevent
        connection pool leaks.
        """
        engine = create_engine(connection_string)
        try:
            df = pd.read_sql(text(query), engine)
            logger.info("Loaded SQL query — rows=%d cols=%d", len(df), len(df.columns))
            return df
        finally:
            engine.dispose()

    @classmethod
    def load(cls, source: str, source_type: str = "csv", **kwargs) -> pd.DataFrame:
        """Generic load dispatcher."""
        loaders = {
            "csv": cls.load_csv,
            "excel": cls.load_excel,
            "json": cls.load_json,
        }
        if source_type == "sql":
            return cls.load_sql(source, kwargs.pop("query", ""))
        if source_type not in loaders:
            raise ValueError(
                f"Unsupported source_type: '{source_type}'. "
                f"Choose from: {list(loaders.keys()) + ['sql']}"
            )
        return loaders[source_type](source, **kwargs)
