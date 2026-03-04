from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional

@dataclass
class IngestionMetadata:
    """Metadata envelope for an ingested dataset."""
    run_id: str
    source_name: str
    source_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    fingerprint: Optional[str] = None
    schema_version: str = "1.0"
    row_count: int = 0
    col_count: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "source_name": self.source_name,
            "source_type": self.source_type,
            "timestamp": self.timestamp.isoformat(),
            "fingerprint": self.fingerprint,
            "schema_version": self.schema_version,
            "row_count": self.row_count,
            "col_count": self.col_count,
            "extra": self.extra
        }
