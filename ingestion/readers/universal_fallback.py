"""
ingestion/readers/universal_fallback.py
-----------------------------------------
Universal Fallback Reader — handles ANY data format, including unseen ones.

Strategy cascade (applied in order until one succeeds):
--------------------------------------------------------
 1. Magic-byte format detection (PNG, PDF, ZIP, GZIP, Parquet, Avro, Arrow)
 2. Try all known formats: CSV, TSV, JSON, JSONL, XML, Excel, Parquet, Feather, Log
 3. HTML table extraction (pd.read_html)
 4. Text extraction from PDF (pdfplumber if available)
 5. ZIP / GZIP decompression + recurse
 6. Fixed-width text parsing (pd.read_fwf)
 7. Key=Value log parsing
 8. Pipe/semicolon/tab/colon delimiter auto-trial with sniffer
 9. Raw line-by-line text → each line becomes a row (last resort)
10. Binary → base64 encode metadata row (ensures we never return empty-handed)

The result always contains:
  - a non-empty DataFrame (even if it's partial or metadata-only)
  - a `strategy_used` dict — fed back into the AdaptiveLearner
  - `bad_row_count` — rows that could not be parsed

Principle: NEVER crash. NEVER return empty-handed. ALWAYS explain what was done.
"""

from __future__ import annotations

import base64
import csv
import gzip
import io
import json
import logging
import os
import re
import struct
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ingestion.error_handler import DataFormatError

logger = logging.getLogger("dipex.ingestion.readers.fallback")

# MIME / magic byte signatures
_MAGIC_BYTES = {
    b"\x89PNG": "png_image",
    b"%PDF": "pdf",
    b"PK\x03\x04": "zip",
    b"\x1f\x8b": "gzip",
    b"PAR1": "parquet",
    b"OBj\x01": "avro",
    b"ARROW1": "feather",
    b"\xd0\xcf\x11\xe0": "excel_xls",    # legacy .xls
    b"PK": "xlsx_or_zip",
}


@dataclass
class FallbackReadResult:
    data: pd.DataFrame
    row_count: int
    strategy_used: Dict[str, Any]
    bad_row_count: int = 0
    format_detected: str = "unknown"
    warnings: List[str] = field(default_factory=list)
    is_partial: bool = False


class UniversalFallbackReader:
    """
    Reads ANY file, even formats not seen before, using a progressive strategy cascade.
    Integrates with AdaptiveLearner via strategy_used metadata.
    """

    def __init__(self, max_fallback_rows: int = 5_000_000) -> None:
        self.max_rows = max_fallback_rows

    def read(self, path: str, hint_format: Optional[str] = None,
             hint_encoding: Optional[str] = None) -> FallbackReadResult:
        path = str(path)
        if not os.path.exists(path):
            raise DataFormatError(f"File not found: {path}")

        file_size = os.path.getsize(path)
        ext = Path(path).suffix.lower().lstrip(".")

        logger.info("UniversalFallbackReader: %s (%.1f KB, ext=.%s)", path, file_size / 1024, ext)

        # Try hint format first
        if hint_format:
            result = self._try_format(path, hint_format, hint_encoding)
            if result is not None:
                return result

        # 1. Magic-byte detection
        fmt = self._detect_magic(path)
        if fmt:
            result = self._try_format(path, fmt, hint_encoding)
            if result is not None:
                return result

        # 2. Extension-based + exhaustive format cascade
        fmt_cascade = self._build_cascade(ext)
        for fmt in fmt_cascade:
            result = self._try_format(path, fmt, hint_encoding)
            if result is not None:
                logger.info("Fallback succeeded with format: %s", fmt)
                return result

        # 3. HTML table extraction
        result = self._try_html(path)
        if result is not None:
            return result

        # 4. PDF text extraction
        result = self._try_pdf(path)
        if result is not None:
            return result

        # 5. ZIP / GZIP decompression
        result = self._try_compressed(path, hint_encoding)
        if result is not None:
            return result

        # 6. Fixed-width format
        result = self._try_fwf(path, hint_encoding)
        if result is not None:
            return result

        # 7. Exhaustive delimiter sniffer (any separator char)
        result = self._try_sniffer(path, hint_encoding)
        if result is not None:
            return result

        # 8. Raw lines as rows (last resort — always succeeds on text files)
        result = self._try_raw_lines(path, hint_encoding)
        if result is not None:
            return result

        # 9. Binary last resort: emit file metadata only
        return self._binary_metadata_row(path, file_size)

    # ── Format-specific readers ───────────────────────────────────────────────

    def _try_format(self, path: str, fmt: str, encoding: Optional[str]) -> Optional[FallbackReadResult]:
        try:
            enc = encoding or "utf-8"
            df: Optional[pd.DataFrame] = None

            if fmt in ("csv", "tsv"):
                sep = "\t" if fmt == "tsv" else self._sniff_sep(path, enc)
                df = pd.read_csv(path, sep=sep, encoding=enc, encoding_errors="replace",
                                 on_bad_lines="warn", nrows=self.max_rows, low_memory=False)

            elif fmt in ("json",):
                with open(path, encoding=enc, errors="replace") as f:
                    raw = json.load(f)
                if isinstance(raw, list):
                    df = pd.json_normalize(raw)
                elif isinstance(raw, dict):
                    # Try to find array value
                    for v in raw.values():
                        if isinstance(v, list) and v:
                            df = pd.json_normalize(v); break
                    if df is None:
                        df = pd.DataFrame([raw])

            elif fmt == "jsonl":
                records = []
                bad = 0
                with open(path, encoding=enc, errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            records.append(json.loads(line))
                        except Exception:  # noqa: BLE001
                            bad += 1
                if records:
                    df = pd.json_normalize(records)

            elif fmt in ("xlsx", "excel", "xls"):
                df = pd.read_excel(path, nrows=self.max_rows)

            elif fmt == "parquet":
                df = pd.read_parquet(path)
                if len(df) > self.max_rows:
                    df = df.head(self.max_rows)

            elif fmt == "feather":
                import pyarrow.feather as pf
                df = pf.read_feather(path)

            elif fmt == "avro":
                import fastavro
                with open(path, "rb") as f:
                    reader = fastavro.reader(f)
                    records = [r for r in reader][:self.max_rows]
                    df = pd.DataFrame(records)

            elif fmt in ("xml",):
                df = pd.read_xml(path)

            elif fmt in ("log",):
                df = self._parse_log(path, enc)

            elif fmt == "pdf":
                return self._try_pdf(path)

            if df is not None and not df.empty:
                df = df.dropna(how="all").dropna(axis=1, how="all")
                return FallbackReadResult(
                    data=df, row_count=len(df), format_detected=fmt,
                    strategy_used={"format": fmt, "encoding": enc},
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Fallback format %s failed: %s", fmt, exc)
        return None

    def _try_html(self, path: str) -> Optional[FallbackReadResult]:
        try:
            tables = pd.read_html(path)
            if tables:
                df = tables[0]
                return FallbackReadResult(
                    data=df, row_count=len(df), format_detected="html",
                    strategy_used={"format": "html", "encoding": "utf-8"},
                    warnings=["Data extracted from HTML table"],
                )
        except Exception:  # noqa: BLE001
            pass
        return None

    def _try_pdf(self, path: str) -> Optional[FallbackReadResult]:
        try:
            import pdfplumber
            rows = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages[:10]:   # first 10 pages
                    text = page.extract_text()
                    if text:
                        for line in text.split("\n"):
                            if line.strip():
                                rows.append({"_text": line.strip(), "_source": "pdf"})
                    # Try table extraction too
                    tables = page.extract_tables()
                    for tbl in (tables or []):
                        if tbl and tbl[0]:
                            header = [str(h) for h in tbl[0]]
                            for row in tbl[1:]:
                                rows.append(dict(zip(header, row)))
            if rows:
                df = pd.DataFrame(rows)
                return FallbackReadResult(
                    data=df, row_count=len(df), format_detected="pdf",
                    strategy_used={"format": "pdf", "encoding": "utf-8"},
                    warnings=["Text/tables extracted from PDF — structural fidelity may vary"],
                )
        except ImportError:
            logger.debug("pdfplumber not installed — skipping PDF extraction")
        except Exception as exc:  # noqa: BLE001
            logger.debug("PDF extraction failed: %s", exc)
        return None

    def _try_compressed(self, path: str, encoding: Optional[str]) -> Optional[FallbackReadResult]:
        try:
            if zipfile.is_zipfile(path):
                with zipfile.ZipFile(path) as z:
                    names = z.namelist()
                    if names:
                        inner = names[0]
                        data  = z.read(inner)
                        tmp   = tempfile.NamedTemporaryFile(
                            suffix=Path(inner).suffix, delete=False)
                        tmp.write(data); tmp.close()
                        try:
                            result = self.read(tmp.name, hint_encoding=encoding)
                            if result and not result.data.empty:
                                result.strategy_used["format"] = f"zip>{result.format_detected}"
                                result.format_detected = f"zip>{result.format_detected}"
                                result.warnings.append(f"Extracted from ZIP: {inner}")
                                return result
                        finally:
                            os.unlink(tmp.name)
        except Exception:  # noqa: BLE001
            pass

        try:
            with gzip.open(path, "rb") as gz:
                data = gz.read(10 * 1024 * 1024)   # max 10MB decompressed
                tmp  = tempfile.NamedTemporaryFile(
                    suffix=Path(path).stem, delete=False)
                tmp.write(data); tmp.close()
                try:
                    result = self.read(tmp.name, hint_encoding=encoding)
                    if result and not result.data.empty:
                        result.strategy_used["format"] = f"gzip>{result.format_detected}"
                        result.warnings.append("Decompressed from GZIP")
                        return result
                finally:
                    os.unlink(tmp.name)
        except Exception:  # noqa: BLE001
            pass

        return None

    def _try_fwf(self, path: str, encoding: Optional[str]) -> Optional[FallbackReadResult]:
        try:
            df = pd.read_fwf(path, encoding=encoding or "utf-8",
                              encoding_errors="replace", nrows=self.max_rows)
            if not df.empty and len(df.columns) > 1:
                return FallbackReadResult(
                    data=df, row_count=len(df), format_detected="fwf",
                    strategy_used={"format": "fwf", "encoding": encoding or "utf-8"},
                    warnings=["Fixed-width format detected"],
                )
        except Exception:  # noqa: BLE001
            pass
        return None

    def _try_sniffer(self, path: str, encoding: Optional[str]) -> Optional[FallbackReadResult]:
        delimiters = ["|", ";", ":", "\t", ",", " "]
        enc = encoding or "utf-8"
        for delim in delimiters:
            try:
                df = pd.read_csv(path, sep=delim, encoding=enc,
                                 encoding_errors="replace", on_bad_lines="warn",
                                 nrows=self.max_rows, low_memory=False)
                if not df.empty and len(df.columns) > 1:
                    return FallbackReadResult(
                        data=df, row_count=len(df), format_detected="csv",
                        strategy_used={"format": "csv", "delimiter": delim, "encoding": enc},
                        warnings=[f"Non-standard delimiter '{delim}' auto-detected"],
                    )
            except Exception:  # noqa: BLE001
                pass
        return None

    def _try_raw_lines(self, path: str, encoding: Optional[str]) -> Optional[FallbackReadResult]:
        """Read any text file line by line — guaranteed to work on text files."""
        for enc in [encoding or "utf-8", "latin-1", "cp1252", "utf-16"]:
            try:
                rows = []
                with open(path, encoding=enc, errors="replace") as f:
                    for i, line in enumerate(f):
                        if i >= self.max_rows:
                            break
                        line = line.rstrip("\n\r")
                        if line.strip():
                            rows.append({"_line": i + 1, "_content": line})
                if rows:
                    df = pd.DataFrame(rows)
                    return FallbackReadResult(
                        data=df, row_count=len(df), format_detected="raw_text",
                        strategy_used={"format": "raw_text", "encoding": enc},
                        warnings=["File read as raw lines — structure not recognised"],
                        is_partial=True,
                    )
            except Exception:  # noqa: BLE001
                pass
        return None

    def _binary_metadata_row(self, path: str, file_size: int) -> FallbackReadResult:
        """Last-resort: return a metadata row for any binary file."""
        with open(path, "rb") as f:
            header_bytes = f.read(64)
        b64 = base64.b64encode(header_bytes).decode("ascii")
        df = pd.DataFrame([{
            "_file_path": path,
            "_file_size_bytes": file_size,
            "_file_ext": Path(path).suffix,
            "_header_b64": b64,
            "_note": "Binary file — could not extract structured data",
        }])
        return FallbackReadResult(
            data=df, row_count=1, format_detected="binary",
            strategy_used={"format": "binary", "encoding": "none"},
            warnings=["Binary file: only metadata row returned"],
            is_partial=True,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_magic(path: str) -> Optional[str]:
        """Detect format from magic bytes."""
        try:
            with open(path, "rb") as f:
                header = f.read(8)
            for magic, fmt in _MAGIC_BYTES.items():
                if header.startswith(magic):
                    return fmt
        except Exception:  # noqa: BLE001
            pass
        return None

    @staticmethod
    def _build_cascade(ext: str) -> List[str]:
        """Build format try-order based on file extension hint."""
        ext_to_fmt = {
            "csv": ["csv", "tsv", "jsonl", "log"],
            "tsv": ["tsv", "csv"],
            "txt": ["csv", "tsv", "jsonl", "log", "fwf"],
            "json": ["json", "jsonl"],
            "jsonl": ["jsonl", "json"],
            "xml": ["xml", "html"],
            "html": ["html", "xml"],
            "htm": ["html"],
            "xlsx": ["excel", "csv"],
            "xls": ["excel"],
            "parquet": ["parquet"],
            "avro": ["avro"],
            "feather": ["feather"],
            "log": ["log", "jsonl", "csv"],
            "gz": ["gzip"],
            "zip": ["zip"],
            "pdf": ["pdf"],
        }
        # Always include all formats in case extension is misleading
        default = ["csv", "tsv", "json", "jsonl", "excel", "xml", "parquet", "feather", "avro", "log"]
        specific = ext_to_fmt.get(ext, [])
        seen = set()
        result = []
        for f in specific + default:
            if f not in seen:
                seen.add(f)
                result.append(f)
        return result

    @staticmethod
    def _sniff_sep(path: str, encoding: str) -> str:
        """Sniff CSV delimiter using csv.Sniffer."""
        try:
            with open(path, encoding=encoding, errors="replace") as f:
                sample = f.read(4096)
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample, delimiters=",;\t|:")
            return dialect.delimiter
        except Exception:  # noqa: BLE001
            return ","

    @staticmethod
    def _parse_log(path: str, encoding: str) -> pd.DataFrame:
        """Parse log files: JSON-per-line or key=value format."""
        rows = []
        with open(path, encoding=encoding, errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                    continue
                except Exception:  # noqa: BLE001
                    pass
                kv = dict(re.findall(r"(\w[\w.]*)\s*=\s*([^\s]+)", line))
                if kv:
                    rows.append(kv)
                else:
                    rows.append({"_line": line})
        return pd.DataFrame(rows) if rows else pd.DataFrame()
