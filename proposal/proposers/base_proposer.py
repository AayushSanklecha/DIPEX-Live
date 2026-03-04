"""
proposal/proposers/base_proposer.py
------------------------------------
Abstract Base Class for all Proposers in Step 4.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd

class BaseProposer(ABC):
    """
    Base class for all intelligence proposers.
    Proposers generate hypotheses or suggestions without executing them.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

    @abstractmethod
    def propose(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generates a proposal based on the input data.
        """
        pass
