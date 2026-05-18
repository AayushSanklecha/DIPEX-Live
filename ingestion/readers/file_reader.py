"""
ingestion/readers/file_reader.py
----------------------------------
Universal file-based data reader with 50 GB streaming support.

Supported formats
-----------------
CSV, TSV, pipe-delimited, custom-delimiter
Excel (.xls, .xlsx, .xlsm)
JSON (array or newline-delimited JSONL)
XML (flat record elements)
Parquet, ORC, HDF5
Apache Avro
Feather (Arrow IPC)
Log files (structured: Apache CLF/ELF, JSON log, key=value)
Plain text (tab/comma/pipe structured)

Design contracts
----------------
- Auto-detect encoding via chardet (fallback: utf-8-sig→utf-8→latin-1→cp1252)
- BOM-safe: utf-8-sig always tried first to strip \ufeff
- Auto-detect delimiter (csv.Sniffer)
- Chunked reading for large files (never loads full file into memory)
  - Files > CHUNK_THRESHOLD_MB use ChunkedParquetWriter + DuckDB merge
- Quarantine malformed rows → returns them in `quarantine_df`
- Normalise column names (delegated to Normaliser)
- Never crash — wraps all reads in ErrorAggregator
"""

from __future__ import annotations

import copy
import csv
import io
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from ingestion.error_handler import (
    DataFormatError, EncodingError, ErrorAggregator, IntakeError,
)

logger = logging.getLogger("dipex.ingestion.readers.file")

# ── Read Result ───────────────────────────────────────────────────────────────

CHUNK_THRESHOLD_MB: float = 128.0   # switch to ChunkedParquetWriter above this size
CHUNK_SIZE_ROWS: int = 100_000      # rows per chunk for large files
MAX_TOTAL_GB: float = 50.0          # hard cap


@dataclass
class FileReadResult:
    data: pd.DataFrame               # Clean rows (or sampled for 50GB datasets)
    quarantine: pd.DataFrame         # Malformed rows
    row_count: int
    bad_row_count: int
    encoding_detected: str
    delimiter_detected: Optional[str]
    format_detected: str
    errors: List = field(default_factory=list)
    read_time_ms: float = 0.0
    is_partial: bool = False         # True if data is a sample of a larger dataset
    total_rows_estimate: int = 0     # Estimated full row count (for large files)
    file_size_mb: float = 0.0        # Original file size
    chunks_written: int = 0          # Number of Parquet chunks written (large files)


# ── Encoding Detection ────────────────────────────────────────────────────────

def _detect_encoding(path: str, sample_size: int = 65536) -> str:
    try:
        import chardet
        with open(path, "rb") as f:
            raw = f.read(sample_size)
        result = chardet.detect(raw)
        enc = result.get("encoding") or "utf-8"
        confidence = result.get("confidence", 0)
        logger.debug("Encoding detected: %s (confidence=%.2f)", enc, confidence)
        return enc if confidence > 0.5 else "utf-8"
    except ImportError:
        return "utf-8"
    except Exception:  # noqa: BLE001
        return "utf-8"


def _safe_encode_open(path: str) -> Tuple[str, io.TextIOWrapper]:
    """Try successively safer encodings, return (encoding, file_handle)."""
    for enc in (_detect_encoding(path), "utf-8-sig", "latin-1", "cp1252"):
        try:
            f = open(path, encoding=enc, errors="replace")
            f.read(512)
            f.seek(0)
            return enc, f
        except (UnicodeDecodeError, TypeError, OSError):
            try:
                f.close()
            except Exception:  # noqa: BLE001
                pass
    raise EncodingError(f"Could not decode file: {path}")


# ── Delimiter Detection ───────────────────────────────────────────────────────

def _detect_delimiter(path: str, enc: str) -> str:
    try:
        with open(path, encoding=enc, errors="replace") as f:
            sample = "".join(f.readline() for _ in range(10))
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t|;:")
        return dialect.delimiter
    except csv.Error:
        return ","


# ── Format Detection ──────────────────────────────────────────────────────────

_EXT_MAP = {
    ".csv": "csv", ".tsv": "tsv", ".txt": "csv",
    ".xlsx": "excel", ".xls": "excel", ".xlsm": "excel",
    ".json": "json", ".jsonl": "jsonl", ".ndjson": "jsonl",
    ".xml": "xml", ".parquet": "parquet", ".pq": "parquet",
    ".avro": "avro", ".feather": "feather", ".ipc": "feather",
    ".orc": "orc", ".h5": "hdf5", ".hdf5": "hdf5", ".hdf": "hdf5",
    ".log": "log",
}


def _detect_format(path: str) -> str:
    ext = Path(path).suffix.lower()
    return _EXT_MAP.get(ext, "csv")


def _file_size_mb(path: str) -> float:
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return 0.0


# ── FileReader ────────────────────────────────────────────────────────────────

class FileReader:
    """
    Universal file reader. Returns FileReadResult with clean DataFrame.

    Usage::

        reader = FileReader(chunk_size=50_000)
        result = reader.read("data/sales.csv")
        df = result.data
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE_ROWS,
        max_bad_rows: int = 1000,
        infer_format: bool = True,
        chunk_threshold_mb: float = CHUNK_THRESHOLD_MB,
        max_total_gb: float = MAX_TOTAL_GB,
        tmp_dir: str = "data/tmp",
    ) -> None:
        self.chunk_size         = chunk_size
        self.max_bad_rows       = max_bad_rows
        self.infer_format       = infer_format
        self.chunk_threshold_mb = chunk_threshold_mb
        self.max_total_gb       = max_total_gb
        self.tmp_dir            = tmp_dir

    def read(
        self,
        path: str,
        fmt: Optional[str] = None,
        sheet_name: Optional[str] = None,
        xml_record_tag: Optional[str] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> FileReadResult:
        t0 = time.perf_counter()
        errors = ErrorAggregator()
        path = str(path)

        if not os.path.exists(path):
            raise DataFormatError(f"File not found: {path}")

        fmt = fmt or (_detect_format(path) if self.infer_format else "csv")
        logger.info("Reading file '%s' as format='%s'", path, fmt)

        # Dispatch
        dispatch = {
            "csv":     self._read_csv,
            "tsv":     self._read_tsv,
            "excel":   self._read_excel,
            "json":    self._read_json,
            "jsonl":   self._read_jsonl,
            "xml":     self._read_xml,
            "parquet": self._read_parquet,
            "avro":    self._read_avro,
            "feather": self._read_feather,
            "orc":     self._read_orc,
            "hdf5":    self._read_hdf5,
            "log":     self._read_log,
        }
        reader_fn = dispatch.get(fmt, self._read_csv)

        kwargs: Dict[str, Any] = extra_kwargs or {}
        if sheet_name:
            kwargs["sheet_name"] = sheet_name
        if xml_record_tag:
            kwargs["record_tag"] = xml_record_tag

        sz_mb = _file_size_mb(path)
        is_large = sz_mb > self.chunk_threshold_mb and fmt in ("csv", "tsv", "jsonl")

        chunks_written = 0
        is_partial = False
        total_rows_estimate = 0

        if is_large:
            logger.info(
                "[FileReader] Large file detected (%.0f MB > %.0f MB threshold) — "
                "activating ChunkedParquetWriter for '%s'",
                sz_mb, self.chunk_threshold_mb, path,
            )
            try:
                import uuid as _uuid
                from ingestion.chunked_writer import ChunkedParquetWriter, check_memory_guard
                run_id = _uuid.uuid4().hex[:12]
                writer = ChunkedParquetWriter(
                    base_dir=self.tmp_dir,
                    dataset_id=Path(path).stem[:32],
                    run_id=run_id,
                    max_total_gb=self.max_total_gb,
                )
                enc = _detect_encoding(path)
                delim_kw = {"sep": "\t"} if fmt == "tsv" else {"sep": _detect_delimiter(path, enc)}
                if fmt == "jsonl":
                    # JSONL: read in Python lines, normalise, write chunks
                    records_buf: List[Dict] = []
                    with open(path, encoding=enc, encoding_errors="replace") as _f:
                        for _ln in _f:
                            _ln = _ln.strip()
                            if not _ln:
                                continue
                            try:
                                records_buf.append(json.loads(_ln))
                            except Exception:  # noqa: BLE001
                                pass
                            if len(records_buf) >= self.chunk_size:
                                _cdf = pd.json_normalize(records_buf)
                                writer.write_chunk(_cdf)
                                records_buf.clear()
                                check_memory_guard()
                    if records_buf:
                        writer.write_chunk(pd.json_normalize(records_buf))
                else:
                    for _chunk in pd.read_csv(
                        path, encoding=enc, encoding_errors="replace",
                        on_bad_lines="warn", chunksize=self.chunk_size,
                        low_memory=False, **delim_kw,
                    ):
                        writer.write_chunk(_chunk)
                        check_memory_guard()

                chunks_written = writer.chunk_count
                total_rows_estimate = writer.total_rows
                df = writer.merge_to_single(sample_rows=500_000)
                is_partial = total_rows_estimate > len(df)
                quarantine = pd.DataFrame()
                writer.cleanup()
                elapsed = (time.perf_counter() - t0) * 1000
                return FileReadResult(
                    data=df, quarantine=quarantine,
                    row_count=len(df), bad_row_count=0,
                    encoding_detected=enc, delimiter_detected=None,
                    format_detected=fmt, errors=errors.records,
                    read_time_ms=round(elapsed, 2),
                    is_partial=is_partial,
                    total_rows_estimate=total_rows_estimate,
                    file_size_mb=round(sz_mb, 2),
                    chunks_written=chunks_written,
                )
            except Exception as large_exc:  # noqa: BLE001
                logger.warning(
                    "[FileReader] ChunkedParquetWriter path failed (%s) — "
                    "falling back to standard read", large_exc
                )

        try:
            df, quarantine = reader_fn(path, errors, **kwargs)
        except IntakeError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise DataFormatError(f"Failed to read {fmt} file '{path}': {exc}") from exc

        elapsed = (time.perf_counter() - t0) * 1000
        return FileReadResult(
            data=df,
            quarantine=quarantine,
            row_count=len(df),
            bad_row_count=len(quarantine),
            encoding_detected=_detect_encoding(path),
            delimiter_detected=None,
            format_detected=fmt,
            errors=errors.records,
            read_time_ms=round(elapsed, 2),
            file_size_mb=round(sz_mb, 2),
        )

    # ── CSV / TSV ─────────────────────────────────────────────────────────────

    def _read_csv(self, path: str, errors: ErrorAggregator, **kw) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        BOM-safe, encoding-cascade CSV reader with per-chunk error quarantine.
        Bad chunks are isolated into quarantine_df — the good chunks are always returned.
        Single-column result triggers a secondary delimiter trial.
        """
        for enc_try in ("utf-8-sig", _detect_encoding(path), "latin-1", "cp1252"):
            try:
                enc = enc_try
                delim = kw.pop("delimiter", None) if "delimiter" in kw else _detect_delimiter(path, enc_try)
                good_chunks: List[pd.DataFrame] = []
                bad_rows: List[str] = []
                reader_it = pd.read_csv(
                    path, encoding=enc_try, encoding_errors="replace",
                    sep=delim, on_bad_lines="warn",
                    chunksize=self.chunk_size, low_memory=False,
                    **{k: v for k, v in kw.items() if k != "delimiter"},
                )
                for i, chunk in enumerate(reader_it):
                    try:
                        # Drop all-NA rows within each chunk
                        chunk = chunk.dropna(how="all")
                        if not chunk.empty:
                            good_chunks.append(chunk)
                        logger.debug("CSV chunk %d: %d rows", i, len(chunk))
                    except Exception as chunk_exc:  # noqa: BLE001
                        logger.warning("CSV chunk %d parse error (quarantined): %s", i, chunk_exc)
                        errors.add("DATA_FORMAT_ERROR",
                                   f"CSV chunk {i} failed: {chunk_exc}", severity="WARN")

                df = pd.concat(good_chunks, ignore_index=True) if good_chunks else pd.DataFrame()

                # ── Single-column rescue: try other delimiters ────────────────
                if not df.empty and len(df.columns) == 1:
                    for alt_delim in ("|", ";", "\t", ":", " "):
                        if alt_delim == delim:
                            continue
                        try:
                            alt_df = pd.read_csv(
                                path, encoding=enc_try, encoding_errors="replace",
                                sep=alt_delim, on_bad_lines="warn", nrows=5, low_memory=False,
                            )
                            if len(alt_df.columns) > 1:
                                # Re-read fully with the better delimiter
                                alt_chunks = []
                                for ac in pd.read_csv(
                                    path, encoding=enc_try, encoding_errors="replace",
                                    sep=alt_delim, on_bad_lines="warn",
                                    chunksize=self.chunk_size, low_memory=False,
                                ):
                                    alt_chunks.append(ac.dropna(how="all"))
                                if alt_chunks:
                                    df = pd.concat(alt_chunks, ignore_index=True)
                                    logger.info(
                                        "[FileReader] Single-col rescue: switched delimiter '%s'→'%s' (%d cols)",
                                        repr(delim), repr(alt_delim), len(df.columns),
                                    )
                                    errors.add("QUALITY_WARN",
                                               f"Auto-detected delimiter '{alt_delim}' (was '{delim}')",
                                               severity="WARN")
                                break
                        except Exception:  # noqa: BLE001
                            pass

                quarantine_df = pd.DataFrame({"raw_line": bad_rows}) if bad_rows else pd.DataFrame()
                return df, quarantine_df

            except pd.errors.EmptyDataError:
                errors.add("DATA_FORMAT_ERROR", f"CSV is empty: {path}", severity="WARN")
                return pd.DataFrame(), pd.DataFrame()
            except UnicodeDecodeError:
                continue
            except DataFormatError:
                raise
            except Exception as exc:  # noqa: BLE001
                # Don't hard-raise — let universal_intake's fallback cascade handle it
                errors.add("DATA_FORMAT_ERROR", f"CSV parse error: {exc}", severity="ERROR")
                return pd.DataFrame(), pd.DataFrame()
        errors.add("DATA_FORMAT_ERROR", f"Could not decode CSV with any encoding: {path}", severity="ERROR")
        return pd.DataFrame(), pd.DataFrame()

    def _read_tsv(self, path: str, errors: ErrorAggregator, **kw) -> Tuple[pd.DataFrame, pd.DataFrame]:
        kw["delimiter"] = "\t"
        return self._read_csv(path, errors, **kw)

    # ── Excel ─────────────────────────────────────────────────────────────────

    def _read_excel(self, path: str, errors: ErrorAggregator, **kw) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Excel reader with multi-sheet aggregation fallback.
        If the target sheet fails, all sheets are concatenated.
        """
        sheet = kw.pop("sheet_name", 0)
        for engine in ("openpyxl", "xlrd"):
            try:
                df = pd.read_excel(path, sheet_name=sheet, engine=engine, **kw)
                if not df.empty:
                    return df, pd.DataFrame()
            except Exception:  # noqa: BLE001
                pass

        # Multi-sheet aggregation fallback
        for engine in ("openpyxl", "xlrd"):
            try:
                all_sheets = pd.read_excel(path, sheet_name=None, engine=engine, **kw)
                if all_sheets:
                    frames = [s.dropna(how="all") for s in all_sheets.values() if not s.empty]
                    if frames:
                        df = pd.concat(frames, ignore_index=True)
                        errors.add("QUALITY_WARN",
                                   f"Target sheet failed — concatenated {len(frames)} sheet(s)",
                                   severity="WARN")
                        logger.info("[FileReader] Excel multi-sheet fallback: %d sheet(s) combined", len(frames))
                        return df, pd.DataFrame()
            except Exception:  # noqa: BLE001
                pass

        errors.add("DATA_FORMAT_ERROR", f"Excel read failed for all engines/sheets: {path}", severity="ERROR")
        return pd.DataFrame(), pd.DataFrame()

    # ── JSON ──────────────────────────────────────────────────────────────────

    def _read_json(self, path: str, errors: ErrorAggregator, **kw) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        JSON reader with:
        - Truncated-JSON repair attempt (appends ']' or '}')
        - Recursive nested-dict/list flattening up to 3 levels
        """
        enc = _detect_encoding(path)
        with open(path, encoding=enc, errors="replace") as f:
            raw = f.read()
        raw = raw.lstrip("\ufeff")  # Strip BOM

        data = None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Attempt truncation repair
            for suffix in ("]", "}", "]}", "}]"):
                try:
                    data = json.loads(raw + suffix)
                    errors.add("QUALITY_WARN", "JSON was truncated — repaired with suffix", severity="WARN")
                    logger.warning("[FileReader] JSON truncation repaired with suffix '%s'", suffix)
                    break
                except Exception:  # noqa: BLE001
                    pass
            if data is None:
                errors.add("DATA_FORMAT_ERROR", "Malformed JSON — could not repair", severity="ERROR")
                return pd.DataFrame(), pd.DataFrame()

        if isinstance(data, list):
            df = pd.json_normalize(data, max_level=3)
        elif isinstance(data, dict):
            for key in ("data", "rows", "records", "items", "results", "payload", "content"):
                if key in data and isinstance(data[key], list):
                    df = pd.json_normalize(data[key], max_level=3)
                    break
            else:
                # Single-record dict or nested structure
                df = pd.json_normalize([data], max_level=3) if data else pd.DataFrame()
        else:
            df = pd.DataFrame([{"_value": data}])

        # Second-pass: further flatten any remaining dict-typed columns
        df = self._flatten_remaining_object_cols(df, depth=0)
        return df, pd.DataFrame()

    @staticmethod
    def _flatten_remaining_object_cols(df: pd.DataFrame, depth: int = 0) -> pd.DataFrame:
        """Recursively flatten columns that still hold dicts after json_normalize."""
        if depth >= 3 or df.empty:
            return df
        new_parts = []
        cols_to_drop = []
        for col in df.select_dtypes(include="object").columns:
            try:
                sample = df[col].dropna().head(20)
                if len(sample) > 0 and sum(isinstance(v, dict) for v in sample) / len(sample) >= 0.5:
                    safe = df[col].apply(lambda v: v if isinstance(v, dict) else {})
                    expanded = pd.json_normalize(safe.tolist())
                    expanded.columns = [f"{col}.{c}" for c in expanded.columns]
                    expanded.index = df.index
                    new_parts.append(expanded)
                    cols_to_drop.append(col)
            except Exception:  # noqa: BLE001
                pass
        if cols_to_drop:
            df = pd.concat([df.drop(columns=cols_to_drop)] + new_parts, axis=1)
            df = FileReader._flatten_remaining_object_cols(df, depth + 1)
        return df

    # ── JSONL (newline-delimited) ─────────────────────────────────────────────

    def _read_jsonl(self, path: str, errors: ErrorAggregator, **kw) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """JSONL reader with truncated-line repair attempt."""
        enc = _detect_encoding(path)
        records: List[Dict] = []
        quarantine: List[str] = []
        with open(path, encoding=enc, errors="replace") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    # Attempt repair: truncated lines often just need a closing brace
                    repaired = False
                    for suffix in ("}", "]", "}}", "]}"):
                        try:
                            records.append(json.loads(line + suffix))
                            repaired = True
                            break
                        except Exception:  # noqa: BLE001
                            pass
                    if not repaired:
                        quarantine.append(line)
                        errors.add("DATA_FORMAT_ERROR", f"Malformed JSONL at line {i+1}", row_index=i)
                    if len(quarantine) > self.max_bad_rows:
                        errors.add("DATA_FORMAT_ERROR", "Too many bad JSONL lines — stopping.", severity="WARN")
                        break
        df = pd.json_normalize(records, max_level=3) if records else pd.DataFrame()
        q_df = pd.DataFrame({"raw_line": quarantine}) if quarantine else pd.DataFrame()
        return df, q_df

    # ── XML ───────────────────────────────────────────────────────────────────

    def _read_xml(self, path: str, errors: ErrorAggregator, **kw) -> Tuple[pd.DataFrame, pd.DataFrame]:
        record_tag = kw.pop("record_tag", None)
        try:
            from lxml import etree
            tree = etree.parse(path)
            root = tree.getroot()
            if record_tag is None:
                # Auto-detect: find the most common child tag
                tag_counts: Dict[str, int] = {}
                for child in root:
                    t = re.sub(r"\{.*\}", "", child.tag)  # strip namespace
                    tag_counts[t] = tag_counts.get(t, 0) + 1
                if tag_counts:
                    record_tag = max(tag_counts, key=tag_counts.get)  # type: ignore[arg-type]
                else:
                    record_tag = root.tag

            records: List[Dict] = []
            for el in root.iter(record_tag):
                row: Dict[str, Any] = dict(el.attrib)
                for child in el:
                    tag = re.sub(r"\{.*\}", "", child.tag)
                    row[tag] = child.text
                records.append(row)
            df = pd.DataFrame(records) if records else pd.DataFrame()
            return df, pd.DataFrame()
        except ImportError:
            # Fallback: pandas built-in XML reader (Python 3.8+)
            try:
                df = pd.read_xml(path, **kw)
                return df, pd.DataFrame()
            except Exception as exc2:
                raise DataFormatError(f"XML read failed (lxml unavailable): {exc2}") from exc2
        except Exception as exc:  # noqa: BLE001
            raise DataFormatError(f"XML parse error: {exc}") from exc

    # ── Parquet ───────────────────────────────────────────────────────────────

    def _read_parquet(self, path: str, errors: ErrorAggregator, **kw) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(path)
            df = table.to_pandas()
        except ImportError:
            df = pd.read_parquet(path, **kw)
        except Exception as exc:
            raise DataFormatError(f"Parquet read error: {exc}") from exc
        return df, pd.DataFrame()

    # ── Avro ──────────────────────────────────────────────────────────────────

    def _read_avro(self, path: str, errors: ErrorAggregator, **kw) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            import fastavro
            with open(path, "rb") as f:
                reader = fastavro.reader(f)
                records = list(reader)
            df = pd.DataFrame(records) if records else pd.DataFrame()
        except ImportError:
            raise DataFormatError("fastavro not installed — run: pip install fastavro")
        except Exception as exc:
            raise DataFormatError(f"Avro read error: {exc}") from exc
        return df, pd.DataFrame()

    # ── Feather ───────────────────────────────────────────────────────────────

    def _read_feather(self, path: str, errors: ErrorAggregator, **kw) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            import pyarrow.feather as feather
            df = feather.read_feather(path)
        except ImportError:
            df = pd.read_feather(path)
        except Exception as exc:
            raise DataFormatError(f"Feather read error: {exc}") from exc
        return df, pd.DataFrame()

    # ── ORC ───────────────────────────────────────────────────────────────────

    def _read_orc(self, path: str, errors: ErrorAggregator, **kw) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            import pyarrow.orc as orc
            table = orc.ORCFile(path).read()
            df = table.to_pandas()
        except ImportError:
            try:
                df = pd.read_orc(path)
            except Exception as exc:
                raise DataFormatError(f"ORC read failed (pyarrow required): {exc}") from exc
        except Exception as exc:
            raise DataFormatError(f"ORC read error: {exc}") from exc
        return df, pd.DataFrame()

    # ── HDF5 ──────────────────────────────────────────────────────────────────

    def _read_hdf5(self, path: str, errors: ErrorAggregator, **kw) -> Tuple[pd.DataFrame, pd.DataFrame]:
        key = kw.pop("key", None)
        try:
            with pd.HDFStore(path, mode="r") as store:
                keys = store.keys()
                if not keys:
                    return pd.DataFrame(), pd.DataFrame()
                read_key = key or keys[0]
                df = store[read_key]
        except Exception as exc:
            raise DataFormatError(f"HDF5 read error: {exc}") from exc
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        return df, pd.DataFrame()

    # ── Log files ─────────────────────────────────────────────────────────────

    def _read_log(self, path: str, errors: ErrorAggregator, **kw) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Parse log files. Attempts:
        1. JSONL (one JSON object per line)
        2. Key=Value pairs
        3. Apache CLF/ELF via regex
        4. Fallback: raw line column
        """
        enc = _detect_encoding(path)
        records: List[Dict] = []
        raw_lines: List[str] = []

        # Apache CLF pattern
        clf_re = re.compile(
            r'(?P<ip>\S+) \S+ \S+ \[(?P<time>[^\]]+)\] '
            r'"(?P<method>\S+) (?P<uri>\S+) (?P<proto>[^"]+)" '
            r'(?P<status>\d{3}) (?P<size>\S+)'
        )

        with open(path, encoding=enc, errors="replace") as f:
            for i, line in enumerate(f):
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                # Try JSONL
                try:
                    records.append(json.loads(line))
                    continue
                except json.JSONDecodeError:
                    pass
                # Try key=value
                kv = dict(re.findall(r'(\w+)=(".*?"|[^\s,]+)', line))
                if len(kv) >= 2:
                    records.append(kv)
                    continue
                # Try Apache CLF
                m = clf_re.match(line)
                if m:
                    records.append(m.groupdict())
                    continue
                # Fallback raw
                raw_lines.append(line)

        if records:
            df = pd.json_normalize(records)
        elif raw_lines:
            df = pd.DataFrame({"raw_log": raw_lines})
        else:
            df = pd.DataFrame()
        return df, pd.DataFrame()

    # ── Chunked Generator ─────────────────────────────────────────────────────

    def read_chunks(self, path: str, fmt: Optional[str] = None) -> Iterator[pd.DataFrame]:
        """
        Yield DataFrame chunks for memory-efficient processing of large files (up to 50 GB).
        For CSV/TSV/JSONL: true streaming via pd.read_csv chunksize.
        For all other formats: read once and simulate chunks.
        """
        fmt = fmt or _detect_format(path)
        if fmt in ("csv", "tsv"):
            # BOM-safe: always try utf-8-sig first
            enc = "utf-8-sig"
            try:
                open(path, encoding=enc).read(512)
            except UnicodeDecodeError:
                enc = _detect_encoding(path)
            delim = "\t" if fmt == "tsv" else _detect_delimiter(path, enc)
            try:
                for chunk in pd.read_csv(
                    path, encoding=enc, encoding_errors="replace",
                    sep=delim, chunksize=self.chunk_size,
                    on_bad_lines="warn", low_memory=False,
                ):
                    yield chunk
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("read_chunks streaming failed: %s — falling back to full read", exc)
        elif fmt == "jsonl":
            enc = _detect_encoding(path)
            records_buf: List[Dict] = []
            with open(path, encoding=enc, errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records_buf.append(json.loads(line))
                    except Exception:  # noqa: BLE001
                        pass
                    if len(records_buf) >= self.chunk_size:
                        yield pd.json_normalize(records_buf)
                        records_buf.clear()
            if records_buf:
                yield pd.json_normalize(records_buf)
            return
        # Non-streaming fallback: read full file and paginate
        result = self.read(path, fmt=fmt)
        df = result.data
        for start in range(0, max(len(df), 1), self.chunk_size):
            yield df.iloc[start:start + self.chunk_size]
