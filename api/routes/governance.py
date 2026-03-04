"""
api/routes/governance.py
--------------------------
Governance engine, data catalog, and policy management endpoints.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/governance", tags=["Governance"])
logger = logging.getLogger("dipex.api.governance")


class EvaluateGovernanceRequest(BaseModel):
    run_id: str
    confidence_score: float = 0.0
    gate1_decision: str = "PASS"
    gate2_decision: str = "PASS"
    df_columns: Optional[List[str]] = None


class CatalogRegisterRequest(BaseModel):
    column_name: str
    classification: str = "INTERNAL"   # PII | SENSITIVE | INTERNAL | PUBLIC
    description: str = ""
    data_type: str = ""
    owner: str = ""
    allowed_in_output: bool = True
    allowed_in_training: bool = True
    tags: Optional[List[str]] = None


@router.post("/evaluate")
async def evaluate_governance(req: EvaluateGovernanceRequest):
    """Run governance policies for a pipeline run."""
    import yaml
    from governance.governance_engine import GovernanceEngine
    from governance.data_catalog import DataCatalog

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    catalog = DataCatalog()
    pii_columns = catalog.get_pii_columns()

    engine = GovernanceEngine.from_config(config)
    decision = engine.evaluate(
        run_id=req.run_id,
        confidence_score=req.confidence_score,
        gate1_decision=req.gate1_decision,
        gate2_decision=req.gate2_decision,
        df_columns=req.df_columns,
        pii_columns=pii_columns,
    )
    return decision.to_dict()


@router.get("/policies")
async def list_policies():
    """List all built-in governance policies."""
    return {
        "built_in_policies": [
            {"id": "G001", "name": "Confidence Floor", "description": "Enforces minimum confidence score by domain"},
            {"id": "G002", "name": "PII Enforcement", "description": "Blocks PII columns from appearing in approved outputs"},
            {"id": "G003", "name": "Audit Mandatory", "description": "Requires a non-empty audit log"},
            {"id": "G004", "name": "Data Quality Gates", "description": "Both Hard Gate 1 and 2 must PASS"},
            {"id": "G005", "name": "Banking AML", "description": "Flags AML/suspicious activity in banking domain"},
            {"id": "G006", "name": "Healthcare HIPAA/PHI", "description": "Blocks PHI columns in healthcare domain output"},
        ],
    }


@router.get("/catalog")
async def list_catalog(classification: Optional[str] = None):
    """List data catalog entries."""
    from governance.data_catalog import DataCatalog
    catalog = DataCatalog()
    return {
        "entries": catalog.list(classification=classification),
        "pii_columns": catalog.get_pii_columns(),
        "sensitive_columns": catalog.get_sensitive_columns(),
    }


@router.post("/catalog/register")
async def register_catalog_entry(req: CatalogRegisterRequest):
    """Register a column in the data catalog."""
    from governance.data_catalog import DataCatalog
    catalog = DataCatalog()
    try:
        entry = catalog.register(
            column_name=req.column_name,
            classification=req.classification,
            description=req.description,
            data_type=req.data_type,
            owner=req.owner,
            allowed_in_output=req.allowed_in_output,
            allowed_in_training=req.allowed_in_training,
            tags=req.tags or [],
        )
        return {"status": "registered", "entry": entry}
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from exc


@router.get("/catalog/{column_name}")
async def get_catalog_entry(column_name: str):
    """Get catalog metadata for a specific column."""
    from governance.data_catalog import DataCatalog
    catalog = DataCatalog()
    entry = catalog.get(column_name)
    if not entry:
        raise HTTPException(404, detail=f"Column '{column_name}' not found in catalog.")
    return entry
