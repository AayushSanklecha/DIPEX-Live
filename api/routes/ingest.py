import uuid
import shutil
import pandas as pd
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException

from ingestion.universal_intake import UniversalIntake, SourceConfig

router = APIRouter(prefix="/api/ingest", tags=["ingestion"])

@router.post("/")
async def ingest_file(file: UploadFile = File(...)):
    """Upload a data file and register it for processing via POST /api/run.

    The file is saved as ``{run_id}_sample.csv`` so that the downstream
    ``POST /api/run`` endpoint can locate it without needing the original
    filename.  The original filename is returned in the response for
    reference.
    """
    run_id = str(uuid.uuid4())
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Always store as {run_id}_sample.csv so /api/run can find it
    file_path = upload_dir / f"{run_id}_sample.csv"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "status": "UPLOADED",
        "run_id": run_id,
        "original_filename": file.filename,
        "file_path": str(file_path),
        "next_step": f"POST /api/run with {{\"run_id\": \"{run_id}\", \"target\": \"<your_target_col>\"}}"
    }

@router.post("/direct")
async def ingest_direct(file: UploadFile = File(...), dataset_id: str = "direct_upload"):
    """
    Directly feed a file (CSV/JSON) into the DIPEX Universal Intake pipeline.
    Bypasses external databases completely.
    """
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"direct_{uuid.uuid4()}_{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Load into pandas to feed straight to DIPEX
        if file.filename.endswith(".json"):
            df = pd.read_json(file_path)
        else:
            df = pd.read_csv(file_path)
            
        intake = UniversalIntake()
        snapshot = intake.ingest_dataframe(df, dataset_id=dataset_id)
        
        return {
            "status": "PROCESSED",
            "dataset_id": dataset_id,
            "is_compliant": snapshot.is_compliant,
            "rows": len(snapshot.data),
            "columns": len(snapshot.data.columns),
            "schema": {col: str(dtype) for col, dtype in snapshot.data.dtypes.items()}
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Direct ingestion failed: {str(e)}")

@router.post("/fetch")
async def ingest_fetch(source_type: str, dataset_id: str, config: dict):
    """
    Directly pull data into DIPEX that is ALREADY stored inside a database 
    (PostgreSQL, MongoDB, Redis, etc.). Bypasses local file uploads.

    Example JSON Body (config):
    {
        "uri": "postgresql://admin:supersecret@localhost:5432/dipex",
        "table": "financial_tx"
    }
    """
    from ingestion.connectors.factory import ConnectorFactory
    
    try:
        # Extract from the requested database
        connector = ConnectorFactory.create(source_type, config)
        df_raw = connector.extract()
        
        if df_raw.empty:
            raise HTTPException(status_code=400, detail="Requested database/table is empty or not found.")
            
        # Push straight into DIPEX ML pipeline
        intake = UniversalIntake()
        snapshot = intake.ingest_dataframe(df_raw, dataset_id=dataset_id)
        
        return {
            "status": "FETCHED_AND_PROCESSED",
            "source_type": source_type,
            "dataset_id": dataset_id,
            "is_compliant": snapshot.is_compliant,
            "rows_extracted": len(snapshot.data),
            "columns": len(snapshot.data.columns),
            "schema": {col: str(dtype) for col, dtype in snapshot.data.dtypes.items()}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database fetch failed: {str(e)}")
