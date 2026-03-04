"""
ingestion/connectors/parquet_connector.py
-------------------------------------------
Production Parquet file connector for DIPEX.

Parquet is the industry-standard columnar file format used in:
  - Data Lakes (S3, GCS, Azure Blob)
  - Apache Spark / Flink pipelines
  - DuckDB (which can also read Parquet directly)
  - DIPEX intermediate pipeline output

Features:
  - Read single file or glob of files (e.g. 'data/sales/*.parquet')
  - Schema inspection via PyArrow schema
  - Predicate pushdown (column + row filters)
  - Chunked streaming via PyArrow RecordBatch reader
  - Write DataFrame → Parquet (snappy or zstd compression)
  - Partition-aware reads (Hive partition format: key=value/...)
  - No server required — pure file I/O via pyarrow

Credentials: None required (file path based)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd

from .base_connector import BaseConnector, ConnectorError

logger = logging.getLogger("dipex.connectors.parquet")

_DEFAULT_CHUNK   = 100_000   # rows per streaming chunk
_DEFAULT_CODEC   = "snappy"  # compression: snappy | zstd | gzip | none


class ParquetConnector(BaseConnector):
    """
    Parquet columnar file connector using PyArrow.

    Config keys:
        path         : Path to .parquet file or glob (e.g. 'data/*.parquet')
        columns      : List of column names to read (default: all)
        filters      : PyArrow filters list for predicate pushdown
                       e.g. [("year", "=", 2024), ("region", "in", ["EU", "US"])]
        chunk_size   : Rows per streaming batch (default: 100_000)
        compression  : Write compression codec — snappy | zstd | gzip | none
        partition_cols: Columns to partition by on write (Hive format)
        row_group_size: Row group size for write (default: PyArrow default)

    Usage:
        conn = ParquetConnector({"path": "data/sales/*.parquet",
                                  "columns": ["id", "amount", "date"]})
        df = conn.extract()
        conn.write(df, "output/result.parquet", compression="zstd")
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._chunk_size: int = int(config.get("chunk_size", _DEFAULT_CHUNK))

    def _get_path(self) -> str:
        path = os.environ.get("PARQUET_PATH", self.config.get("path", ""))
        if not path:
            raise ConnectorError("ParquetConnector: 'path' is required in config")
        return path

    # ------------------------------------------------------------------
    # BaseConnector interface
    # ------------------------------------------------------------------

    def test_connection(self) -> bool:
        """Verify the target path/glob resolves to at least one readable file."""
        try:
            import pyarrow.parquet as pq  # type: ignore
            path = self._get_path()
            if "*" in path or "?" in path:
                import glob
                files = glob.glob(path, recursive=True)
                if not files:
                    logger.error("ParquetConnector: glob '%s' matched 0 files", path)
                    return False
                pq.read_schema(files[0])
            else:
                pq.read_schema(path)
            logger.info("ParquetConnector: path OK — %s", path)
            return True
        except Exception as exc:
            logger.error("ParquetConnector: test_connection FAILED — %s", exc)
            return False

    def get_schema(self) -> Dict[str, Any]:
        """Return PyArrow schema as a DIPEX schema dict."""
        try:
            import pyarrow.parquet as pq
            path = self._get_path()
            if "*" in path or "?" in path:
                import glob
                files = glob.glob(path, recursive=True)
                schema = pq.read_schema(files[0]) if files else None
                file_count = len(files)
            else:
                schema = pq.read_schema(path)
                file_count = 1

            if schema is None:
                return {"error": "no files matched", "columns": []}

            columns = [f.name for f in schema]
            dtypes  = {f.name: str(f.type) for f in schema}
            
            # Estimate row count from metadata if available
            estimated_rows = -1
            try:
                meta = pq.read_metadata(path if file_count == 1 else files[0])
                estimated_rows = meta.num_rows
            except Exception:
                pass

            return {
                "path":          path,
                "file_count":    file_count,
                "columns":       columns,
                "dtypes":        dtypes,
                "estimated_row_count": estimated_rows,
                "description":   f"Parquet schema for: {path}",
            }
        except Exception as exc:
            return {"error": str(exc), "columns": [], "dtypes": {}}

    def extract(self, query: Optional[str] = None, **kwargs: Any) -> pd.DataFrame:
        """
        Read Parquet file(s) into a DataFrame.
        query is unused (Parquet uses filters config key for predicate pushdown).
        """
        try:
            import pyarrow.parquet as pq
            path    = self._get_path()
            columns = kwargs.get("columns") or self.config.get("columns")
            filters = kwargs.get("filters") or self.config.get("filters")

            # pq.read_table handles both single files and datasets (glob via directory)
            if "*" in path or "?" in path or os.path.isdir(path):
                import pyarrow.dataset as ds  # type: ignore
                dataset = ds.dataset(path, format="parquet")
                table   = dataset.to_table(columns=columns, filter=_build_filter(filters))
            else:
                table = pq.read_table(path, columns=columns, filters=filters)

            df = table.to_pandas()
            logger.info("ParquetConnector: read %d rows × %d cols from %s",
                        len(df), len(df.columns), path)
            return df

        except ImportError as exc:
            raise ConnectorError("pyarrow is required: pip install pyarrow") from exc
        except Exception as exc:
            raise ConnectorError(f"ParquetConnector: extract failed — {exc}") from exc

    def stream(self, chunk_size: Optional[int] = None, **kwargs: Any) -> Iterator[pd.DataFrame]:
        """
        Yield DataFrame chunks using PyArrow's RecordBatch reader.
        Highly memory-efficient for large Parquet files.
        """
        size    = chunk_size or self._chunk_size
        path    = self._get_path()
        columns = kwargs.get("columns") or self.config.get("columns")
        filters = kwargs.get("filters") or self.config.get("filters")
        try:
            import pyarrow.parquet as pq
            if "*" in path or "?" in path or os.path.isdir(path):
                import pyarrow.dataset as ds
                dataset  = ds.dataset(path, format="parquet")
                scanner  = dataset.scanner(columns=columns, filter=_build_filter(filters),
                                           batch_size=size)
                for batch in scanner.to_batches():
                    yield batch.to_pandas()
            else:
                pf      = pq.ParquetFile(path)
                for batch in pf.iter_batches(batch_size=size, columns=columns):
                    yield batch.to_pandas()
        except ImportError as exc:
            raise ConnectorError("pyarrow is required: pip install pyarrow") from exc
        except Exception as exc:
            raise ConnectorError(f"ParquetConnector: stream failed — {exc}") from exc

    def close(self) -> None:
        pass   # No persistent connection for file-based connector

    # ------------------------------------------------------------------
    # Write support
    # ------------------------------------------------------------------

    def write(
        self,
        df: pd.DataFrame,
        path: str,
        compression: Optional[str] = None,
        partition_cols: Optional[List[str]] = None,
        row_group_size: Optional[int] = None,
    ) -> str:
        """
        Write a DataFrame to Parquet format.

        Args:
            df             : DataFrame to write
            path           : Output file path or directory (if partitioned)
            compression    : snappy (default) | zstd | gzip | none
            partition_cols : Columns to partition by (Hive-style directory layout)
            row_group_size : PyArrow row group size

        Returns:
            Resolved output path as string.
        """
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq

            codec = compression or self.config.get("compression", _DEFAULT_CODEC)
            table = pa.Table.from_pandas(df, preserve_index=False)

            Path(path).parent.mkdir(parents=True, exist_ok=True)

            write_kwargs: Dict[str, Any] = {"compression": codec}
            if row_group_size:
                write_kwargs["row_group_size"] = row_group_size

            if partition_cols:
                pq.write_to_dataset(
                    table, root_path=path,
                    partition_cols=partition_cols,
                    **write_kwargs,
                )
            else:
                pq.write_table(table, path, **write_kwargs)

            logger.info(
                "ParquetConnector: wrote %d rows → %s (codec=%s)",
                len(df), path, codec,
            )
            return path

        except ImportError as exc:
            raise ConnectorError("pyarrow is required: pip install pyarrow") from exc
        except Exception as exc:
            raise ConnectorError(f"ParquetConnector: write failed — {exc}") from exc


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _build_filter(filters: Any) -> Any:
    """
    Convert filter list to PyArrow dataset expression if needed.
    Supports both pq.read_table filter tuples and ds.Expression.
    """
    if filters is None:
        return None
    # Already a PyArrow expression
    try:
        import pyarrow.compute as pc  # type: ignore
        if hasattr(filters, "_call"):   # pyarrow Expression
            return filters
    except ImportError:
        pass
    # List of tuples — returned as-is (valid for pq.read_table)
    return filters
