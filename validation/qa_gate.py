import logging
from typing import List, Dict, Any

class QAGate:
    """Hard gate that decides whether to proceed or halt the pipeline."""
    
    def __init__(self, severity_threshold: str = "ERROR"):
        self.severity_threshold = severity_threshold
        self.logger = logging.getLogger("dipex.qa_gate")

    def evaluate(self, validation_errors: List[Dict[str, Any]]) -> bool:
        """
        Evaluates errors against the threshold.
        Returns True if the pipeline can proceed, False if it must halt.
        """
        critical_errors = [e for e in validation_errors if e.get('severity') == "ERROR"]
        
        if critical_errors:
            self.logger.error(f"QA Gate Failed: {len(critical_errors)} critical errors found.")
            for e in critical_errors:
                self.logger.error(f"  [{e.get('type')}] {e.get('message')}")
            return False
            
        warnings = [e for e in validation_errors if e.get('severity') == "WARNING"]
        if warnings:
            self.logger.warning(f"QA Gate Passed with {len(warnings)} warnings.")
            for w in warnings:
                self.logger.warning(f"  [{w.get('type')}] {w.get('message')}")
        else:
            self.logger.info("QA Gate Passed: No issues found.")
            
        return True
