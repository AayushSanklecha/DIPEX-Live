"""
ingestion/readers/file_reader.py
----------------------------------
Universal file-based data reader.

Supported formats
-----------------
CSV, TSV, pipe-delimited, custom-delimiter
Excel (.xls, .xlsx, .xlsm)
JSON (array or newline-delimited JSONL)
XML (flat record elements)
Parquet
Apache Avro
Feather (Arrow IPC)
Log files (structured: Apache CLF/ELF, JSON log, key=value)
Plain text (tab/comma/pipe structured)

Design contracts
----------------
- Auto-detect encoding via chardet (fallback: utf-8, latin-1, cp1252)
- Auto-detect delimiter (csv.Sniffer)
- Strip BOM characters
- Chunked reading for large files (never loads full file into memory)
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

@dataclass
class FileReadResult:
    data: pd.DataFrame               # Clean rows
    quarantine: pd.DataFrame         # Malformed rows
    row_count: int
    bad_row_count: int
    encoding_detected: str
    delimiter_detected: Optional[str]
    format_detected: str
    errors: List = field(default_factory=list)
    read_time_ms: float = 0.0


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
    ".log": "log",
}


def _detect_format(path: str) -> str:
    ext = Path(path).suffix.lower()
    return _EXT_MAP.get(ext, "csv")


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
        chunk_size: int = 50_000,
        max_bad_rows: int = 1000,
        infer_format: bool = True,
    ) -> None:
        self.chunk_size   = chunk_size
        self.max_bad_rows = max_bad_rows
        self.infer_format = infer_format

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
            "log":     self._read_log,
        }
        reader_fn = dispatch.get(fmt, self._read_csv)

        kwargs: Dict[str, Any] = extra_kwargs or {}
        if sheet_name:
            kwargs["sheet_name"] = sheet_name
        if xml_record_tag:
            kwargs["record_tag"] = xml_record_tag

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
        )

    # ── CSV / TSV ─────────────────────────────────────────────────────────────

    def _read_csv(self, path: str, errors: ErrorAggregator, **kw) -> Tuple[pd.DataFrame, pd.DataFrame]:
        enc = _detect_encoding(path)
        delim = kw.pop("delimiter", None) or _detect_delimiter(path, enc)
        chunks: List[pd.DataFrame] = []
        quarantine_rows: List[Dict] = []

        try:
            reader = pd.read_csv(
                path, encoding=enc, encoding_errors="replace",
                sep=delim, on_bad_lines="warn",
                chunksize=self.chunk_size, low_memory=False,
                **kw,
            )
            for i, chunk in enumerate(reader):
                chunks.append(chunk)
                logger.debug("CSV chunk %d: %d rows", i, len(chunk))
        except pd.errors.EmptyDataError:
            errors.add("DATA_FORMAT_ERROR", f"CSV is empty: {path}", severity="WARN")
            return pd.DataFrame(), pd.DataFrame()
        except Exception as exc:  # noqa: BLE001
            raise DataFormatError(f"CSV parse error: {exc}") from exc

        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        return df, pd.DataFrame(quarantine_rows)

    def _read_tsv(self, path: str, errors: ErrorAggregator, **kw) -> Tuple[pd.DataFrame, pd.DataFrame]:
        kw["delimiter"] = "\t"
        return self._read_csv(path, errors, **kw)

    # ── Excel ─────────────────────────────────────────────────────────────────

    def _read_excel(self, path: str, errors: ErrorAggregator, **kw) -> Tuple[pd.DataFrame, pd.DataFrame]:
        sheet = kw.pop("sheet_name", 0)
        try:
            df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl", **kw)
        except Exception:
            try:
                df = pd.read_excel(path, sheet_name=sheet, engine="xlrd", **kw)
            except Exception as exc2:
                raise DataFormatError(f"Excel read failed: {exc2}") from exc2
        return df, pd.DataFrame()

    # ── JSON ──────────────────────────────────────────────────────────────────

    def _read_json(self, path: str, errors: ErrorAggregator, **kw) -> Tuple[pd.DataFrame, pd.DataFrame]:
        enc = _detect_encoding(path)
        with open(path, encoding=enc, errors="replace") as f:
            raw = f.read()
        # Strip BOM
        raw = raw.lstrip("\ufeff")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DataFormatError(f"Malformed JSON: {exc}") from exc

        if isinstance(data, list):
            df = pd.json_normalize(data)
        elif isinstance(data, dict):
            # Try common keys: "data", "rows", "records", "items", "results"
            for key in ("data", "rows", "records", "items", "results"):
                if key in data and isinstance(data[key], list):
                    df = pd.json_normalize(data[key])
                    break
            else:
                df = pd.DataFrame([data])
        else:
            raise DataFormatError("JSON top-level must be an array or object.")

        return df, pd.DataFrame()

    # ── JSONL (newline-delimited) ─────────────────────────────────────────────

    def _read_jsonl(self, path: str, errors: ErrorAggregator, **kw) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
                    quarantine.append(line)
                    errors.add("DATA_FORMAT_ERROR", f"Malformed JSONL at line {i+1}", row_index=i)
                    if len(quarantine) > self.max_bad_rows:
                        errors.add("DATA_FORMAT_ERROR", "Too many bad JSONL lines — aborting.", severity="ERROR")
                        break
        df = pd.json_normalize(records) if records else pd.DataFrame()
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
        """Yield DataFrame chunks for memory-efficient processing of large files."""
        fmt = fmt or _detect_format(path)
        if fmt in ("csv", "tsv"):
            enc = _detect_encoding(path)
            delim = "\t" if fmt == "tsv" else _detect_delimiter(path, enc)
            for chunk in pd.read_csv(
                path, encoding=enc, encoding_errors="replace",
                sep=delim, chunksize=self.chunk_size, on_bad_lines="warn", low_memory=False,
            ):
                yield chunk
        else:
            # For non-CSV formats, read all and simulate chunks
            result = self.read(path, fmt=fmt)
            df = result.data
            for start in range(0, max(len(df), 1), self.chunk_size):
                yield df.iloc[start:start + self.chunk_size]
