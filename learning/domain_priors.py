"""
learning/domain_priors.py
--------------------------
Enhancement 3: Warm-Start Priors (Cold-Start Fix).

Defines pre-computed Q-value priors per domain so that on the very first run
(episode=0) the bandit has an informed starting point instead of starting
from 0.5 for all actions.

Domain-specific priors were derived from expert knowledge of what retry
strategies work best for different data types:
  - banking    : favour `adjust_hyperparameters`, penalise `change_model_class`
  - healthcare : favour `apply_feature_selection`, `increase_regularization`
  - finance    : favour `restart_from_eda`, `reduce_feature_count`
  - default    : balanced starting point for unknown domains

Integration:
    Call `get_prior(domain)` in MetaRLEngine.__init__() when no state file exists.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger("dipex.domain_priors")

# Action space (must match MetaRLEngine.DEFAULT_ACTION_SPACE)
_ACTION_SPACE = [
    "restart_from_eda",
    "restart_from_proposal",
    "restart_full_pipeline",
    "adjust_hyperparameters",
    "apply_feature_selection",
    "change_model_class",
    "increase_regularization",
    "reduce_feature_count",
]

# Expert-derived Q-value priors per domain
# Values represent the expected usefulness [0.0, 1.0] of each action
# for that domain's typical data characteristics.
_DOMAIN_PRIORS: Dict[str, Dict[str, float]] = {
    "banking": {
        "restart_from_eda":          0.55,
        "restart_from_proposal":     0.50,
        "restart_full_pipeline":     0.40,
        "adjust_hyperparameters":    0.75,   # ← Strong prior: banking data is well-structured
        "apply_feature_selection":   0.65,   # many correlated features → selection helps
        "change_model_class":        0.35,   # ← Weak prior: model changes rarely help
        "increase_regularization":   0.60,
        "reduce_feature_count":      0.60,
    },
    "healthcare": {
        "restart_from_eda":          0.60,
        "restart_from_proposal":     0.55,
        "restart_full_pipeline":     0.35,
        "adjust_hyperparameters":    0.60,
        "apply_feature_selection":   0.80,   # ← Strong: clinical datasets often over-featured
        "change_model_class":        0.55,
        "increase_regularization":   0.75,   # ← Strong: small healthcare samples → regularise
        "reduce_feature_count":      0.70,
    },
    "finance": {
        "restart_from_eda":          0.75,   # ← Strong: financial data often has outliers/periods
        "restart_from_proposal":     0.60,
        "restart_full_pipeline":     0.45,
        "adjust_hyperparameters":    0.65,
        "apply_feature_selection":   0.60,
        "change_model_class":        0.55,
        "increase_regularization":   0.65,
        "reduce_feature_count":      0.70,   # ← Strong: financial datasets often noisy wide tables
    },
    "default": {
        # Balanced starting point — all actions start at 0.5 with slight EDA bias
        "restart_from_eda":          0.60,
        "restart_from_proposal":     0.50,
        "restart_full_pipeline":     0.40,
        "adjust_hyperparameters":    0.55,
        "apply_feature_selection":   0.55,
        "change_model_class":        0.45,
        "increase_regularization":   0.55,
        "reduce_feature_count":      0.50,
    },
}

# Additional domain aliases
_DOMAIN_ALIASES: Dict[str, str] = {
    "bank":           "banking",
    "financial":      "finance",
    "medical":        "healthcare",
    "clinical":       "healthcare",
    "ecommerce":      "default",
    "retail":         "default",
    "manufacturing":  "default",
}


def get_prior(domain: Optional[str] = None) -> Dict[str, float]:
    """
    Return the warm-start Q-value prior for the given domain.

    Parameters
    ----------
    domain : Domain identifier string (e.g., 'banking', 'healthcare').
             Case-insensitive. Falls back to 'default' if unknown.

    Returns
    -------
    Dict mapping action name → initial Q-value in [0.0, 1.0]
    """
    key = (domain or "default").lower().strip()
    key = _DOMAIN_ALIASES.get(key, key)

    if key not in _DOMAIN_PRIORS:
        logger.info(
            "DomainPriors: domain '%s' not found — using 'default' prior", domain
        )
        key = "default"

    prior = dict(_DOMAIN_PRIORS[key])

    # Ensure all actions from the full action space are present
    for action in _ACTION_SPACE:
        if action not in prior:
            prior[action] = 0.50  # neutral default for any missing action

    logger.info("DomainPriors: warm-start prior loaded for domain='%s'", key)
    return prior


def list_domains() -> list:
    """Return all registered domain names."""
    return list(_DOMAIN_PRIORS.keys())
