import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

class ProfilingReport:
    """Generates JSON and (future) HTML profiling reports."""
    
    def __init__(self, report_dir: str = "reports"):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, run_id: str, profile_data: Dict[str, Any], drift_data: Dict[str, float]) -> str:
        """Generates a JSON report file."""
        report = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "profile": profile_data,
            "drift": drift_data
        }
        
        filename = f"profile_{run_id}.json"
        report_path = self.report_dir / filename
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        return str(report_path)
