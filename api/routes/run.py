"""
api/routes/run.py
-----------------
Endpoint to trigger a pipeline run in the background.
"""

import logging
import uuid

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, Field

from verify_pipeline import orchestrate_pipeline

router = APIRouter(prefix="/api/run", tags=["pipeline"])
logger = logging.getLogger(__name__)


class PipelineTrigger(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique run identifier")
    target_column: str = Field(..., description="Name of the target column in the uploaded dataset")


class RunResponse(BaseModel):
    status: str
    run_id: str
    message: str


@router.post("/", response_model=RunResponse, summary="Trigger full DIPEX pipeline")
async def trigger_pipeline(
    trigger: PipelineTrigger,
    background_tasks: BackgroundTasks,
) -> RunResponse:
    """
    Enqueues the DIPEX pipeline for the given ``run_id`` and ``target_column``.

    The pipeline executes asynchronously in the background.  Poll ``/api/results/{run_id}``
    for completion status (or integrate a proper task-queue such as Celery for production).
    """
    logger.info("Pipeline run queued: run_id=%s  target=%s", trigger.run_id, trigger.target_column)
    background_tasks.add_task(orchestrate_pipeline, trigger.run_id, trigger.target_column)

    return RunResponse(
        status="QUEUED",
        run_id=trigger.run_id,
        message="Pipeline has been queued and is executing in the background.",
    )
