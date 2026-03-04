"""
analyst/documentation_generator.py
-------------------------------------
Production-grade automated documentation generator.

Generates:
  - Metric definitions dict: column → {mean, std, source, transform_applied}
  - KPI dictionary: {name, formula, owner, threshold, current_value}
  - Data lineage docs: Bronze → Silver → Gold in Markdown and JSON
  - Change log: timestamped list of {what, when, by, operator}

Output: JSON (machine-readable) + Markdown (human-readable)
Governance: All docs reference Gold artefact IDs only.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger("dipex.documentation")


# ══════════════════════════════════════════════════════════════════════════════
# KPI Definition Dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class KPIDefinition:
    name: str
    formula: str
    owner: str
    threshold: Optional[float]
    current_value: Optional[float]
    unit: str = ""
    description: str = ""
    source_artefact_id: Optional[str] = None
    last_updated: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
# DocumentationGenerator
# ══════════════════════════════════════════════════════════════════════════════

class DocumentationGenerator:
    """
    Generates machine-readable JSON + human-readable Markdown documentation
    for Gold artefacts, lineage chains, KPI dictionaries, and change logs.
    """

    def __init__(self, docs_dir: Optional[str] = None) -> None:
        """
        Args:
            docs_dir: Optional directory for auto-exporting documentation.
                      If None, all exports must be requested explicitly.
        """
        self.docs_dir = Path(docs_dir) if docs_dir else None
        if self.docs_dir:
            self.docs_dir.mkdir(parents=True, exist_ok=True)

    def generate_metric_definitions(
        self,
        df: pd.DataFrame,
        gold_artefact_id: Optional[str] = None,
        transform_log: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate a metric definition dict for all numeric columns.

        Args:
            df: Gold artefact DataFrame
            gold_artefact_id: Reference to Gold artefact
            transform_log: List of transformation descriptions applied

        Returns:
            {column_name: {mean, std, p25, p50, p75, min, max, nulls, source, transform_applied}}
        """
        transforms = transform_log or []
        definitions: Dict[str, Dict[str, Any]] = {}

        numeric_cols = df.select_dtypes(include="number").columns
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            desc = series.describe()
            definitions[col] = {
                "mean": round(float(desc.get("mean", 0.0)), 6),
                "std": round(float(desc.get("std", 0.0)), 6),
                "min": round(float(desc.get("min", 0.0)), 6),
                "p25": round(float(desc.get("25%", 0.0)), 6),
                "p50": round(float(desc.get("50%", 0.0)), 6),
                "p75": round(float(desc.get("75%", 0.0)), 6),
                "max": round(float(desc.get("max", 0.0)), 6),
                "null_count": int(df[col].isna().sum()),
                "null_pct": round(float(df[col].isna().mean()) * 100, 2),
                "source": gold_artefact_id or "unknown_gold_artefact",
                "transform_applied": ", ".join(transforms) or "none",
            }

        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            series = df[col].dropna()
            definitions[col] = {
                "dtype": "categorical",
                "cardinality": int(series.nunique()),
                "top_values": series.value_counts().head(5).to_dict(),
                "null_count": int(df[col].isna().sum()),
                "null_pct": round(float(df[col].isna().mean()) * 100, 2),
                "source": gold_artefact_id or "unknown_gold_artefact",
                "transform_applied": ", ".join(transforms) or "none",
            }

        return definitions

    def generate_kpi_dictionary(
        self,
        df_or_kpis,
        dataset_id: Optional[str] = None,
    ):
        """
        Dual-mode KPI dictionary generator.

        - If passed a DataFrame + dataset_id: auto-infer KPIs from numeric columns → list
        - If passed a List[KPIDefinition]: return dict {name: {...}}
        """
        if isinstance(df_or_kpis, pd.DataFrame):
            # Test-contract mode: DataFrame → list of KPI dicts
            df = df_or_kpis
            kpi_list = []
            for col in df.select_dtypes(include="number").columns:
                series = df[col].dropna()
                if len(series) == 0:
                    continue
                kpi_list.append({
                    "name": col,
                    "formula": f"SUM({col}) or AVG({col})",
                    "owner": "auto-inferred",
                    "dataset_id": dataset_id or "unknown",
                    "mean": round(float(series.mean()), 4),
                    "std": round(float(series.std()), 4),
                    "null_pct": round(float(df[col].isna().mean()) * 100, 2),
                })
            return kpi_list
        else:
            # Original KPIDefinition-list mode → dict
            return {kpi.name: kpi.to_dict() for kpi in df_or_kpis}

    def generate_data_contract(
        self,
        df: pd.DataFrame,
        dataset_id: str,
    ) -> Dict[str, Any]:
        """
        Generate a data contract specifying schema, types, and nullability for a dataset.

        Returns:
            {dataset_id, schema: {col: {dtype, nullable, n_unique}}, row_count}
        """
        schema = {}
        for col in df.columns:
            schema[col] = {
                "dtype": str(df[col].dtype),
                "nullable": bool(df[col].isna().any()),
                "n_unique": int(df[col].nunique()),
                "null_pct": round(float(df[col].isna().mean()) * 100, 2),
            }
        return {
            "dataset_id": dataset_id,
            "schema": schema,
            "row_count": len(df),
            "column_count": len(df.columns),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def log_change(
        self,
        dataset_id: str,
        version: str,
        change_type: str,
        description: str,
        operator: str = "system",
    ) -> None:
        """
        Append a change log entry for a dataset.
        Persisted to docs_dir/changelog_{dataset_id}.json.
        """
        entry = {
            "dataset_id": dataset_id,
            "version": version,
            "change_type": change_type,
            "description": description,
            "operator": operator,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if self.docs_dir:
            log_path = self.docs_dir / f"changelog_{dataset_id}.json"
            existing: List[Dict[str, Any]] = []
            if log_path.exists():
                with open(log_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            existing.append(entry)
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
        else:
            logger.warning("log_change: docs_dir not set; entry discarded.")

    def get_changelog(
        self,
        dataset_id: str,
    ) -> List[Dict[str, Any]]:
        """Retrieve all change log entries for a dataset."""
        if not self.docs_dir:
            return []
        log_path = self.docs_dir / f"changelog_{dataset_id}.json"
        if not log_path.exists():
            return []
        with open(log_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def generate_lineage_document(
        self,
        dataset_id: str,
        bronze_checksum: str,
        silver_id: str,
        gold_lineage_id: str,
    ) -> Dict[str, Any]:
        """
        Generate a minimal lineage document for a Gold artefact.

        Returns:
            {dataset_id, bronze_checksum, silver_id, gold_lineage_id, generated_at}
        """
        doc = {
            "dataset_id": dataset_id,
            "bronze_checksum": bronze_checksum,
            "silver_id": silver_id,
            "gold_lineage_id": gold_lineage_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "lineage_chain": f"{bronze_checksum} → {silver_id} → {gold_lineage_id}",
        }
        if self.docs_dir:
            doc_path = self.docs_dir / f"lineage_{dataset_id}.json"
            with open(doc_path, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2)
        return doc


    def kpi_dict_to_markdown(self, kpi_dict: Dict[str, Dict[str, Any]]) -> str:
        """Convert KPI dictionary to Markdown table format."""
        lines = [
            "# KPI Dictionary\n",
            "| KPI | Formula | Owner | Threshold | Current Value | Unit |",
            "|---|---|---|---|---|---|",
        ]
        for name, kpi in kpi_dict.items():
            lines.append(
                f"| {name} | {kpi.get('formula','—')} | {kpi.get('owner','—')} "
                f"| {kpi.get('threshold','—')} | {kpi.get('current_value','—')} "
                f"| {kpi.get('unit','—')} |"
            )
        return "\n".join(lines)

    def generate_lineage_docs(
        self,
        dataset_id: str,
        bronze_meta: Optional[Dict[str, Any]] = None,
        silver_meta: Optional[Dict[str, Any]] = None,
        gold_meta: Optional[Dict[str, Any]] = None,
        operations: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate data lineage documentation for a dataset.

        Returns both JSON and Markdown representations.
        """
        chain = {
            "dataset_id": dataset_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "lineage": {
                "bronze": bronze_meta or {
                    "status": "not_provided",
                    "note": "Bronze snapshot metadata not provided"
                },
                "silver": silver_meta or {
                    "status": "not_provided",
                    "note": "Silver snapshot metadata not provided"
                },
                "gold": gold_meta or {
                    "status": "not_provided",
                    "note": "Gold derivation metadata not provided"
                },
            },
            "operations_applied": operations or [],
        }

        # Generate Markdown representation
        md_lines = [
            f"# Data Lineage: `{dataset_id}`\n",
            f"_Generated: {chain['generated_at']}_\n",
            "## Bronze Layer (Raw Immutable Snapshot)",
        ]
        bronze = chain["lineage"]["bronze"]
        for k, v in bronze.items():
            md_lines.append(f"- **{k}**: {v}")

        md_lines.append("\n## Silver Layer (Validated, ISSF-normalized)")
        silver = chain["lineage"]["silver"]
        for k, v in silver.items():
            md_lines.append(f"- **{k}**: {v}")

        md_lines.append("\n## Gold Layer (Analyst Operations Applied)")
        gold = chain["lineage"]["gold"]
        for k, v in gold.items():
            md_lines.append(f"- **{k}**: {v}")

        if operations:
            md_lines.append("\n## Operations Applied (in order)")
            for op in operations:
                md_lines.append(
                    f"- `{op.get('operation', '?')}` by {op.get('by', '?')} "
                    f"at {op.get('timestamp', '?')}"
                )

        chain["markdown"] = "\n".join(md_lines)
        return chain

    def generate_change_log(
        self, operations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate a timestamped, ordered change log from a list of operation records.

        Args:
            operations: List of {what, when, by, operator, dataset_id, ...}

        Returns:
            Sorted list of standardized change log entries
        """
        log = []
        for op in operations:
            entry = {
                "timestamp": op.get("when") or op.get("timestamp") or
                             datetime.now(timezone.utc).isoformat(),
                "operation": op.get("what") or op.get("operation") or "unknown",
                "performed_by": op.get("by") or op.get("operator") or "unknown",
                "dataset_id": op.get("dataset_id"),
                "gold_artefact_id": op.get("gold_artefact_id"),
                "parameters": op.get("parameters") or {},
                "outcome": op.get("outcome") or "unknown",
            }
            log.append(entry)

        return sorted(log, key=lambda e: e["timestamp"])

    def export_json(
        self, data: Any, path: str, indent: int = 2
    ) -> Path:
        """Write any documentation object as JSON to disk."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
        logger.info("Documentation exported to %s", out_path)
        return out_path

    def export_markdown(self, content: str, path: str) -> Path:
        """Write markdown content to disk."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info("Markdown documentation exported to %s", out_path)
        return out_path
