import pytest
import pandas as pd
import numpy as np
from ingestion.batch_loader import BatchLoader
from ingestion.snapshot import SnapshotManager
from validation.qa_gate import QAGate
from validation.schema_validator import SchemaValidator

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id": [1, 2, 3],
        "value": [10.5, 20.0, 15.2],
        "category": ["A", "B", "A"]
    })

def test_batch_loader_csv(tmp_path, sample_df):
    csv_file = tmp_path / "test.csv"
    sample_df.to_csv(csv_file, index=False)
    loaded_df = BatchLoader.load_csv(str(csv_file))
    assert len(loaded_df) == 3
    assert "value" in loaded_df.columns

def test_snapshot_determinism(sample_df):
    manager = SnapshotManager(snapshot_dir="data/test_snapshots")
    hash1 = manager.calculate_hash(sample_df)
    hash2 = manager.calculate_hash(sample_df)
    assert hash1 == hash2

def test_qa_gate_fail():
    gate = QAGate(severity_threshold="ERROR")
    errors = [{"severity": "ERROR", "message": "Critical failure", "type": "TEST_ERROR"}]
    assert gate.evaluate(errors) is False

def test_qa_gate_pass():
    gate = QAGate(severity_threshold="ERROR")
    errors = [{"severity": "WARNING", "message": "Minor issue", "type": "TEST_WARN"}]
    assert gate.evaluate(errors) is True
