"""analyst/__init__.py — Analyst Intelligence Automation Layer public API."""

# ── Tier analysts ─────────────────────────────────────────────────────────────
from analyst.junior_analyst import JuniorAnalyst
from analyst.mid_analyst import MidAnalyst
from analyst.senior_analyst import SeniorAnalyst

# ── Orchestrator ──────────────────────────────────────────────────────────────
from analyst.analyst_orchestrator import AnalystOrchestrator

# ── Supporting engines ────────────────────────────────────────────────────────
from analyst.sql_automation import SQLAutomationEngine
from analyst.excel_engine import ExcelEngine

# ── Visualization ─────────────────────────────────────────────────────────────
from analyst.visualization_engine import (
    VisualizationEngine,
    VisualizationMisleadingDetector,
    select_chart_type,
)

# ── Documentation ─────────────────────────────────────────────────────────────
from analyst.documentation_generator import (
    DocumentationGenerator,
    KPIDefinition,
)

# ── Reporting Intelligence ────────────────────────────────────────────────────
from analyst.reporting_intelligence import ReportingIntelligence

# ── Problem Framing ───────────────────────────────────────────────────────────
from analyst.problem_framing import (
    ProblemFraming,
    ProblemFramingEngine,
    KPICandidate,
    FramedProblem,
    ProposedKPI,
)

# ── Experiment Designer ───────────────────────────────────────────────────────
from analyst.experiment_designer import (
    ExperimentDesigner,
    ExperimentDesign,
)

# ── Strategic Advisor ─────────────────────────────────────────────────────────
from analyst.strategic_advisor import StrategicAdvisor

# ── Mentorship Engine ─────────────────────────────────────────────────────────
from analyst.mentorship_engine import (
    MentorshipEngine,
    SQLReviewResult,
    InterpretationReview,
    SignOffResult,
    ReviewComment,
    MentorshipReview,
)

# ── RL Optimizer ──────────────────────────────────────────────────────────────
from analyst.rl_optimizer import (
    AnalystRLOptimizer,
    RLOptimizer,
    RLProposal,
    StrategyDomain,
    encode_state,
)

__all__ = [
    # Tiers
    "JuniorAnalyst",
    "MidAnalyst",
    "SeniorAnalyst",
    "AnalystOrchestrator",
    # Supporting engines
    "SQLAutomationEngine",
    "ExcelEngine",
    # Visualization
    "VisualizationEngine",
    "VisualizationMisleadingDetector",
    "select_chart_type",
    # Documentation
    "DocumentationGenerator",
    "KPIDefinition",
    # Reporting
    "ReportingIntelligence",
    # Problem Framing
    "ProblemFraming",
    "ProblemFramingEngine",
    "KPICandidate",
    "FramedProblem",
    "ProposedKPI",
    # Experiment Designer
    "ExperimentDesigner",
    "ExperimentDesign",
    # Strategic Advisor
    "StrategicAdvisor",
    # Mentorship Engine
    "MentorshipEngine",
    "SQLReviewResult",
    "InterpretationReview",
    "SignOffResult",
    "ReviewComment",
    "MentorshipReview",
    # RL
    "AnalystRLOptimizer",
    "RLOptimizer",
    "RLProposal",
    "StrategyDomain",
    "encode_state",
]
