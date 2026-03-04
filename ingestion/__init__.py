"""
ingestion/__init__.py
-----------------------
Universal Data Intake & Processing Layer public API.
"""
from ingestion.universal_intake import UniversalIntake, SourceConfig
from ingestion.issf import ISSFSnapshot, ColumnMeta, IngestionError
from ingestion.normaliser import Normaliser
from ingestion.schema_registry import SchemaRegistry, SchemaDriftReport
from ingestion.quality_gate import QualityGate, QualityReport
from ingestion.pipeline_bridge import PipelineBridge, PipelineResult
from ingestion.batch_processor import BatchProcessor, BatchConfig, CDCTracker, WatermarkStore, ArchiveManager
from ingestion.error_handler import (
    SafeExecutor, ErrorAggregator,
    SchemaError, DataFormatError, EncodingError,
    APITimeoutError, DBConnectionError, StreamLagError, QualityGateError,
)
from ingestion.adaptive_learner import AdaptiveLearner, IngestionOutcome
from ingestion.readers.universal_fallback import UniversalFallbackReader

# ── Layer isolation ──────────────────────────────────────────────────────────
from ingestion.data_layers import LayerManager, GoldArtefact
from ingestion.immutability_guard import (
    ImmutableDataFrame,
    ImmutabilityViolationError,
    ChecksumMismatchError,
    LayerAccessViolationError,
    LayerWriteGuard,
    MutationProbe,
    DataFrameSignature,
)
from ingestion.lineage import LineageRecord, LineageStore, TransformationStep

__all__ = [
    # Core intake
    "UniversalIntake", "SourceConfig",
    "ISSFSnapshot", "ColumnMeta", "IngestionError",
    "Normaliser", "SchemaRegistry", "SchemaDriftReport",
    "QualityGate", "QualityReport",
    "PipelineBridge", "PipelineResult",
    "BatchProcessor", "BatchConfig", "CDCTracker", "WatermarkStore", "ArchiveManager",
    "SafeExecutor", "ErrorAggregator",
    "SchemaError", "DataFormatError", "EncodingError",
    "APITimeoutError", "DBConnectionError", "StreamLagError", "QualityGateError",
    "AdaptiveLearner", "IngestionOutcome",
    "UniversalFallbackReader",
    # Layer isolation
    "LayerManager", "GoldArtefact",
    "ImmutableDataFrame",
    "ImmutabilityViolationError", "ChecksumMismatchError",
    "LayerAccessViolationError", "LayerWriteGuard",
    "MutationProbe", "DataFrameSignature",
    "LineageRecord", "LineageStore", "TransformationStep",
]
