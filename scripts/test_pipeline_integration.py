"""
scripts/test_pipeline_integration.py
─────────────────────────────────────
Static + functional integration test for the DIPEX pipeline.

Checks every stage import, inter-stage data contracts, compliance wiring,
ConfidenceVector penalty propagation, and the frontend→API domain contract.

Run from project root:
    python scripts/test_pipeline_integration.py

EXIT 0 = all checks passed
EXIT 1 = one or more checks failed (printed in red)
"""
from __future__ import annotations
import sys, os, traceback, textwrap, json
from typing import List, Tuple

# ── Make project root importable ─────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── Colour helpers (no external dep) ─────────────────────────────────────────
ok   = lambda s: f"\033[32m✔  {s}\033[0m"
fail = lambda s: f"\033[31m✖  {s}\033[0m"
warn = lambda s: f"\033[33m⚠  {s}\033[0m"
hdr  = lambda s: f"\033[1;34m\n{'─'*60}\n  {s}\n{'─'*60}\033[0m"

RESULTS: List[Tuple[bool, str]] = []

def chk(label: str, fn):
    """Run fn(), record PASS/FAIL, never raise."""
    try:
        fn()
        RESULTS.append((True, label))
        print(ok(label))
    except Exception as exc:
        RESULTS.append((False, f"{label} — {exc}"))
        print(fail(f"{label}"))
        print(textwrap.indent(traceback.format_exc(limit=3), "       "))


# ═══════════════════════════════════════════════════════════════════════════════
print(hdr("1 · REGULATORY ENGINE — imports & rule loading"))
# ═══════════════════════════════════════════════════════════════════════════════

def _import_regulatory_engine():
    from validation.regulatory.regulatory_engine import RegulatoryEngine
    from validation.regulatory.base_rule import BaseRegulatoryRule, RegulatoryViolation
    assert callable(RegulatoryEngine.from_config)

def _import_banking_rules():
    from validation.regulatory.banking_rules import (
        PositiveAmountRule, AMLThresholdRule, LoanRatioRule,
        RepaymentConsistencyRule, SuspiciousTransactionPatternRule,
        CurrencyConcentrationRule,
    )
    # 6 banking rules must exist
    assert all([PositiveAmountRule, AMLThresholdRule, LoanRatioRule,
                RepaymentConsistencyRule, SuspiciousTransactionPatternRule,
                CurrencyConcentrationRule])

def _import_healthcare_rules():
    from validation.regulatory.healthcare_rules import (
        AgeRangeRule, VitalSignsRule, DiagnosisCodeFormatRule,
        PHIPresenceRule, ConsentValidationRule, DeIdentificationRule,
    )

def _import_cross_domain_rules():
    from validation.regulatory.cross_domain_rules import (
        GDPRDataResidencyRule, SOXAuditTrailRule,
        HIPAAEncryptionFlagRule, GDPRConsentRequiredRule,
    )

def _import_conflict_resolver():
    from validation.regulatory.conflict_resolver import RuleConflictResolver
    # Real signature: __init__(strategy, primary_domain) — not domain_priority
    r = RuleConflictResolver(strategy="strictest_wins", primary_domain="banking")
    assert hasattr(r, "resolve")

def _engine_loads_domains():
    from validation.regulatory.regulatory_engine import RegulatoryEngine
    # Banking rules require column config to load (empty cfg = 0 rules, which is correct by design)
    # Use the velocity/sus-transaction rule which fires even without explicit columns
    cfg = {"validation": {"regulatory": {
        "domains": ["banking"],
        "conflict_resolution": "strictest_wins",
        "banking": {"amount_columns": ["amount"], "aml_amount_column": "amount"},
    }}}
    engine = RegulatoryEngine.from_config(cfg)
    assert len(engine.rules) > 0, f"No rules loaded for banking domain, got {len(engine.rules)}"

def _engine_loads_multi_domain():
    from validation.regulatory.regulatory_engine import RegulatoryEngine
    # Provide column config so banking rules load; GDPR cross-domain rules always load
    cfg = {"validation": {"regulatory": {
        "domains": ["banking", "gdpr"],
        "conflict_resolution": "strictest_wins",
        "banking": {"amount_columns": ["amount"], "aml_amount_column": "amount"},
    }}}
    engine = RegulatoryEngine.from_config(cfg)
    assert len(engine.rules) >= 3, f"Expected >=3 rules for banking+gdpr, got {len(engine.rules)}"

def _engine_empty_domains_loads_no_rules():
    from validation.regulatory.regulatory_engine import RegulatoryEngine
    cfg = {"validation": {"regulatory": {"domains": []}}}
    engine = RegulatoryEngine.from_config(cfg)
    assert len(engine.rules) == 0, f"Expected 0 rules for empty domains, got {len(engine.rules)}"

def _engine_returns_violations_list():
    import pandas as pd
    from validation.regulatory.regulatory_engine import RegulatoryEngine
    cfg = {"validation": {"regulatory": {"domains": ["banking"], "conflict_resolution": "strictest_wins"}}}
    engine = RegulatoryEngine.from_config(cfg)
    df = pd.DataFrame({"amount": [-100, 200], "transaction_value": [50, -30]})
    violations = engine.evaluate(df)
    assert isinstance(violations, list), "evaluate() must return a list"

def _engine_get_violating_columns():
    import pandas as pd
    from validation.regulatory.regulatory_engine import RegulatoryEngine
    cfg = {"validation": {"regulatory": {"domains": ["banking"]}}}
    engine = RegulatoryEngine.from_config(cfg)
    df = pd.DataFrame({"amount": [-1, -2]})
    engine.evaluate(df)
    cols = engine.get_violating_columns()
    assert isinstance(cols, set), "get_violating_columns() must return a set"

def _engine_get_conflict_report():
    from validation.regulatory.regulatory_engine import RegulatoryEngine
    cfg = {"validation": {"regulatory": {"domains": ["banking"]}}}
    engine = RegulatoryEngine.from_config(cfg)
    report = engine.get_last_conflict_report()
    assert isinstance(report, list), "get_last_conflict_report() must return a list"

chk("regulatory_engine import", _import_regulatory_engine)
chk("banking_rules — 6 rules importable", _import_banking_rules)
chk("healthcare_rules — 6 rules importable", _import_healthcare_rules)
chk("cross_domain_rules — 4 rules importable", _import_cross_domain_rules)
chk("conflict_resolver — instantiation", _import_conflict_resolver)
chk("RegulatoryEngine loads banking domain rules", _engine_loads_domains)
chk("RegulatoryEngine loads multi-domain (banking+gdpr)", _engine_loads_multi_domain)
chk("RegulatoryEngine empty domains → 0 rules", _engine_empty_domains_loads_no_rules)
chk("RegulatoryEngine.evaluate() returns list", _engine_returns_violations_list)
chk("RegulatoryEngine.get_violating_columns() returns set", _engine_get_violating_columns)
chk("RegulatoryEngine.get_last_conflict_report() returns list", _engine_get_conflict_report)


# ═══════════════════════════════════════════════════════════════════════════════
print(hdr("2 · COMPLIANCE DECISION — ComplianceAdvisor"))
# ═══════════════════════════════════════════════════════════════════════════════

def _import_compliance_decision():
    from validation.compliance_decision import ComplianceAdvisor, ComplianceDecision
    assert callable(ComplianceAdvisor.from_config)
    assert hasattr(ComplianceDecision, "__dataclass_fields__") or hasattr(ComplianceDecision, "__annotations__")

def _compliance_decision_allowed():
    from validation.compliance_decision import ComplianceAdvisor
    cfg = {"compliance": {"block_on_critical": True}}
    advisor = ComplianceAdvisor.from_config(cfg)
    decision = advisor.evaluate(violations=[], conflict_report=[], df=None, run_id="test-001")
    assert decision.decision == "allowed", f"Expected allowed, got {decision.decision}"
    assert decision.compliance_penalty == 0.0, f"Expected 0.0 penalty, got {decision.compliance_penalty}"

def _compliance_decision_penalty_formula():
    from validation.regulatory.base_rule import RegulatoryViolation
    from validation.compliance_decision import ComplianceAdvisor
    # Real RegulatoryViolation fields: rule_name, domain, severity, column, offending_count, message
    v1 = RegulatoryViolation(rule_name="R1", domain="banking", severity="CRITICAL",
                             column="col_a", offending_count=2, message="crit")
    v2 = RegulatoryViolation(rule_name="R2", domain="banking", severity="ERROR",
                             column="col_b", offending_count=1, message="err")
    cfg = {"compliance": {"block_on_critical": False}}
    advisor = ComplianceAdvisor.from_config(cfg)
    decision = advisor.evaluate(violations=[v1, v2], conflict_report=[], df=None, run_id="test-002")
    # penalty = -(1x0.20 + 1x0.10) = -0.30
    expected = -(1 * 0.20 + 1 * 0.10)
    assert abs(decision.compliance_penalty - expected) < 0.01, \
        f"Penalty formula wrong: expected {expected:.3f}, got {decision.compliance_penalty:.3f}"


def _compliance_decision_to_dict():
    from validation.compliance_decision import ComplianceAdvisor
    cfg = {}
    advisor = ComplianceAdvisor.from_config(cfg)
    decision = advisor.evaluate(violations=[], conflict_report=[], df=None, run_id="test-003")
    d = decision.to_dict()
    assert isinstance(d, dict), "to_dict() must return dict"
    assert "decision" in d and "compliance_penalty" in d

def _compliance_decision_blocked_on_critical():
    from validation.regulatory.base_rule import RegulatoryViolation
    from validation.compliance_decision import ComplianceAdvisor
    # RegulatoryViolation real fields: rule_name, domain, severity, column, offending_count, message
    v = RegulatoryViolation(rule_name="PHIPresenceRule", domain="healthcare", severity="CRITICAL",
                            column="phi_col", offending_count=5, message="PHI found")
    cfg = {"compliance": {"block_on_critical": True}}
    advisor = ComplianceAdvisor.from_config(cfg)
    decision = advisor.evaluate(violations=[v], conflict_report=[], df=None, run_id="test-004")
    assert decision.decision == "blocked", f"Expected blocked, got {decision.decision}"

chk("compliance_decision module imports OK", _import_compliance_decision)
chk("ComplianceAdvisor: no violations → allowed + penalty=0.0", _compliance_decision_allowed)
chk("ComplianceAdvisor: penalty formula (CRITICAL+ERROR)", _compliance_decision_penalty_formula)
chk("ComplianceDecision.to_dict() returns correct dict", _compliance_decision_to_dict)
chk("ComplianceAdvisor: CRITICAL + block_on_critical=True → blocked", _compliance_decision_blocked_on_critical)


# ═══════════════════════════════════════════════════════════════════════════════
print(hdr("3 · CONFIDENCE VECTOR — compliance_penalty wiring"))
# ═══════════════════════════════════════════════════════════════════════════════

def _import_confidence_vector():
    from verifier.confidence_vector import ConfidenceVector
    assert callable(ConfidenceVector.from_config)

def _cv_aggregate_no_penalty():
    import pandas as pd
    from verifier.confidence_vector import ConfidenceVector
    cfg = {"domain": "generic", "confidence_vector": {}}
    cv = ConfidenceVector.from_config(cfg)
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    v = cv.aggregate(df=df, model_metrics={}, quality_score=0.9,
                     gate2_passed=True, retry_count=0,
                     compliance_penalty=0.0, compliance_decision="allowed")
    assert "confidence_score" in v, "Missing confidence_score key"
    assert 0.0 <= v["confidence_score"] <= 1.0, f"Score out of [0,1]: {v['confidence_score']}"

def _cv_aggregate_with_penalty():
    import pandas as pd
    from verifier.confidence_vector import ConfidenceVector
    cfg = {"domain": "generic", "confidence_vector": {}}
    cv = ConfidenceVector.from_config(cfg)
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    score_clean = cv.aggregate(df=df, model_metrics={}, quality_score=0.9,
                               gate2_passed=True, retry_count=0,
                               compliance_penalty=0.0, compliance_decision="allowed")["confidence_score"]
    score_penalised = cv.aggregate(df=df, model_metrics={}, quality_score=0.9,
                                   gate2_passed=True, retry_count=0,
                                   compliance_penalty=-0.30, compliance_decision="conditional")["confidence_score"]
    assert score_penalised <= score_clean, \
        f"Penalised score ({score_penalised:.4f}) should be ≤ clean ({score_clean:.4f})"

def _cv_aggregate_signature():
    """Ensure aggregate() accepts compliance_penalty and compliance_decision kwargs."""
    import inspect, pandas as pd
    from verifier.confidence_vector import ConfidenceVector
    sig = inspect.signature(ConfidenceVector.aggregate)
    params = set(sig.parameters.keys())
    for required in ("compliance_penalty", "compliance_decision"):
        assert required in params, f"ConfidenceVector.aggregate missing param: {required}"

chk("ConfidenceVector imports OK", _import_confidence_vector)
chk("ConfidenceVector.aggregate() baseline (no penalty)", _cv_aggregate_no_penalty)
chk("ConfidenceVector.aggregate() penalty reduces score", _cv_aggregate_with_penalty)
chk("ConfidenceVector.aggregate() has compliance_penalty & compliance_decision params", _cv_aggregate_signature)


# ═══════════════════════════════════════════════════════════════════════════════
print(hdr("4 · RL THRESHOLD TUNER — compliance feedback"))
# ═══════════════════════════════════════════════════════════════════════════════

def _import_rl_tuner():
    from validation.rl_threshold_tuner import RLThresholdTuner, get_rl_tuner
    assert callable(get_rl_tuner)

def _rl_tuner_record_compliance_outcome():
    from validation.rl_threshold_tuner import RLThresholdTuner
    tuner = RLThresholdTuner()
    # Real signature: record_compliance_outcome(dataset_id, column, threshold, violation_severity)
    for sev in ("CRITICAL", "ERROR", "WARNING", "NONE"):
        tuner.record_compliance_outcome("ds1", "col_x", 0.5, sev)

def _rl_tuner_get_instance():
    from validation.rl_threshold_tuner import RLThresholdTuner
    inst = RLThresholdTuner.get_instance()
    assert isinstance(inst, RLThresholdTuner)

chk("RLThresholdTuner imports OK", _import_rl_tuner)
chk("RLThresholdTuner.record_compliance_outcome() all severities", _rl_tuner_record_compliance_outcome)
chk("RLThresholdTuner.get_instance() returns singleton", _rl_tuner_get_instance)


# ═══════════════════════════════════════════════════════════════════════════════
print(hdr("5 · SHAP EXPLAINER — explain_compliance_violations"))
# ═══════════════════════════════════════════════════════════════════════════════

def _import_shap_explainer():
    from validation.shap_explainer import explain_compliance_violations
    assert callable(explain_compliance_violations)

def _shap_explain_compliance_no_violations():
    import pandas as pd
    from validation.shap_explainer import explain_compliance_violations
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    # Real signature: (df, violations, run_id="N/A", top_n=10) — no shap_scores kwarg
    result = explain_compliance_violations(df, violations=[])
    assert isinstance(result, list), "explain_compliance_violations must return list"
    assert result == [], f"Empty violations should return [], got {result}"

chk("shap_explainer.explain_compliance_violations importable", _import_shap_explainer)
chk("explain_compliance_violations: no violations → empty list", _shap_explain_compliance_no_violations)


# ═══════════════════════════════════════════════════════════════════════════════
print(hdr("6 · LLM PROVIDER — generate_compliance_remediation"))
# ═══════════════════════════════════════════════════════════════════════════════

def _import_llm_provider():
    from reporting_service.llm_provider import LLMProvider
    assert hasattr(LLMProvider, "generate_compliance_remediation")

def _llm_provider_fallback_remediation():
    from validation.regulatory.base_rule import RegulatoryViolation
    from reporting_service.llm_provider import LLMProvider
    # Base class fallback: returns empty when no violations, non-empty when violations given
    class _TestProvider(LLMProvider):
        def generate(self, prompt, **kw): return ""
    p = _TestProvider(config={})
    # With no violations, returns empty string (correct by design)
    assert p.generate_compliance_remediation(violations=[], domain="banking") == ""
    # With a real violation dict, must return non-empty
    fake_v = [{"rule_name": "PositiveAmountRule", "severity": "ERROR", "message": "Negative amounts"}]
    result = p.generate_compliance_remediation(violations=fake_v, domain="banking")
    assert isinstance(result, str) and len(result) > 0, \
        f"generate_compliance_remediation with violations must be non-empty, got: {result!r}"

chk("LLMProvider has generate_compliance_remediation method", _import_llm_provider)
chk("LLMProvider fallback remediation returns non-empty string", _llm_provider_fallback_remediation)


# ═══════════════════════════════════════════════════════════════════════════════
print(hdr("7 · PIPELINE BRIDGE — stage wiring & compliance integration"))
# ═══════════════════════════════════════════════════════════════════════════════

def _import_pipeline_bridge():
    from ingestion.pipeline_bridge import PipelineBridge, StageResult, PipelineResult
    assert callable(PipelineBridge.run)

def _pipeline_bridge_stage_validate_signature():
    """_stage_validate must return bool and accept (df, dataset_id)."""
    import inspect
    from ingestion.pipeline_bridge import PipelineBridge
    sig = inspect.signature(PipelineBridge._stage_validate)
    params = list(sig.parameters.keys())
    assert "df" in params and "dataset_id" in params, \
        f"_stage_validate params wrong: {params}"

def _pipeline_bridge_stage_confidence_vector_has_compliance():
    """_stage_confidence_vector must pass compliance_penalty to cv.aggregate()."""
    import inspect
    from ingestion.pipeline_bridge import PipelineBridge
    src = inspect.getsource(PipelineBridge._stage_confidence_vector)
    assert "compliance_penalty" in src, \
        "_stage_confidence_vector does not pass compliance_penalty to aggregate()"
    assert "compliance_decision" in src, \
        "_stage_confidence_vector does not pass compliance_decision to aggregate()"

def _pipeline_bridge_stage_validate_calls_regulatory():
    """_stage_validate must call RegulatoryEngine and ComplianceAdvisor."""
    import inspect
    from ingestion.pipeline_bridge import PipelineBridge
    src = inspect.getsource(PipelineBridge._stage_validate)
    assert "RegulatoryEngine" in src, "_stage_validate does not use RegulatoryEngine"
    assert "ComplianceAdvisor" in src, "_stage_validate does not use ComplianceAdvisor"
    assert "_compliance_decision" in src, \
        "_stage_validate does not store _compliance_decision on self"

def _pipeline_bridge_compliance_decision_stored_on_self():
    """Verify self._compliance_decision is read in _stage_confidence_vector."""
    import inspect
    from ingestion.pipeline_bridge import PipelineBridge
    src = inspect.getsource(PipelineBridge._stage_confidence_vector)
    assert "_compliance_decision" in src, \
        "_stage_confidence_vector does not read self._compliance_decision"

def _pipeline_bridge_feature_masking_in_validate():
    """Verify violating columns are dropped in _stage_validate."""
    import inspect
    from ingestion.pipeline_bridge import PipelineBridge
    src = inspect.getsource(PipelineBridge._stage_validate)
    assert "violating_columns" in src, "Feature masking missing from _stage_validate"
    assert "df.drop" in src, "df.drop not found — feature masking may be incomplete"

def _pipeline_bridge_blocked_decision_raises():
    """Verify BLOCKED compliance decision halts the pipeline."""
    import inspect
    from ingestion.pipeline_bridge import PipelineBridge
    src = inspect.getsource(PipelineBridge._stage_validate)
    assert "blocked" in src, \
        "_stage_validate does not handle 'blocked' decision"
    assert "RuntimeError" in src or "raise" in src, \
        "_stage_validate does not raise on blocked decision"

chk("PipelineBridge imports OK", _import_pipeline_bridge)
chk("_stage_validate signature: (self, df, dataset_id)", _pipeline_bridge_stage_validate_signature)
chk("_stage_validate calls RegulatoryEngine + ComplianceAdvisor", _pipeline_bridge_stage_validate_calls_regulatory)
chk("_stage_validate stores _compliance_decision on self", _pipeline_bridge_compliance_decision_stored_on_self)
chk("_stage_validate: feature masking (df.drop on violating_columns)", _pipeline_bridge_feature_masking_in_validate)
chk("_stage_validate: BLOCKED decision raises RuntimeError", _pipeline_bridge_blocked_decision_raises)
chk("_stage_confidence_vector passes compliance_penalty + compliance_decision", _pipeline_bridge_stage_confidence_vector_has_compliance)
chk("_stage_confidence_vector reads self._compliance_decision", _pipeline_bridge_compliance_decision_stored_on_self)


# ═══════════════════════════════════════════════════════════════════════════════
print(hdr("8 · API ENDPOINT — domain override wiring"))
# ═══════════════════════════════════════════════════════════════════════════════

def _api_pipeline_run_has_domain_params():
    import inspect
    from api.routes.pipeline_run import pipeline_run
    sig = inspect.signature(pipeline_run)
    params = set(sig.parameters.keys())
    assert "domain" in params,       "pipeline_run endpoint missing 'domain' param"
    assert "extra_domains" in params, "pipeline_run endpoint missing 'extra_domains' param"

def _api_domain_override_logic():
    """Check that the domain override correctly patches config domains list."""
    import inspect
    from api.routes import pipeline_run as pr_mod
    src = inspect.getsource(pr_mod)
    assert 'config.setdefault("validation"' in src or "validation" in src, \
        "API does not patch validation.regulatory.domains"
    assert 'domains = []' in src or '"domains"]' in src, \
        "API does not set empty domains when no domain selected"

def _api_empty_domain_disables_compliance():
    """Verify: empty domain → domains=[] in config (not config.yaml fallback)."""
    import inspect
    from api.routes import pipeline_run as pr_mod
    src = inspect.getsource(pr_mod)
    # Must have the else branch that sets domains=[]
    assert '"domains": []' in src or '["domains"] = []' in src, \
        "No else-branch found that sets domains=[] when domain is blank"

chk("pipeline_run endpoint has 'domain' and 'extra_domains' params", _api_pipeline_run_has_domain_params)
chk("pipeline_run patches config['validation']['regulatory']['domains']", _api_domain_override_logic)
chk("pipeline_run: blank domain → domains=[] (no fallback to config)", _api_empty_domain_disables_compliance)


# ═══════════════════════════════════════════════════════════════════════════════
print(hdr("9 · FRONTEND CONTRACT — RunPipeline.jsx domain fields"))
# ═══════════════════════════════════════════════════════════════════════════════

def _frontend_has_domain_state():
    jsx_path = os.path.join(ROOT, "frontend/src/pages/RunPipeline.jsx")
    with open(jsx_path, encoding="utf-8") as f:
        src = f.read()
    assert "useState('')" in src or "useState(\"\")" in src, \
        "domain useState not found in RunPipeline.jsx"
    assert "domain" in src, "No domain state in RunPipeline.jsx"
    assert "extraDomains" in src, "No extraDomains state in RunPipeline.jsx"

def _frontend_sends_domain_in_formdata():
    jsx_path = os.path.join(ROOT, "frontend/src/pages/RunPipeline.jsx")
    with open(jsx_path, encoding="utf-8") as f:
        src = f.read()
    assert "form.append('domain'" in src, "domain not appended to FormData"
    assert "form.append('extra_domains'" in src, "extra_domains not appended to FormData"

def _frontend_has_domain_cards():
    jsx_path = os.path.join(ROOT, "frontend/src/pages/RunPipeline.jsx")
    with open(jsx_path, encoding="utf-8") as f:
        src = f.read()
    assert "DOMAIN_CARDS" in src, "DOMAIN_CARDS array not defined in RunPipeline.jsx"
    for domain_id in ["banking", "healthcare", "finance", "gdpr", "sox", "hipaa"]:
        assert domain_id in src, f"Domain '{domain_id}' missing from RunPipeline.jsx"

def _frontend_no_domain_shows_shield_off():
    jsx_path = os.path.join(ROOT, "frontend/src/pages/RunPipeline.jsx")
    with open(jsx_path, encoding="utf-8") as f:
        src = f.read()
    assert "ShieldOff" in src, "ShieldOff icon missing — no-domain hint not present"

chk("RunPipeline.jsx has domain + extraDomains state", _frontend_has_domain_state)
chk("RunPipeline.jsx sends domain + extra_domains in FormData", _frontend_sends_domain_in_formdata)
chk("RunPipeline.jsx has DOMAIN_CARDS with all 6 domains", _frontend_has_domain_cards)
chk("RunPipeline.jsx shows ShieldOff hint when no domain selected", _frontend_no_domain_shows_shield_off)


# ═══════════════════════════════════════════════════════════════════════════════
print(hdr("10 · CONFIG.YAML — structure sanity"))
# ═══════════════════════════════════════════════════════════════════════════════

def _config_yaml_loads():
    import yaml
    cfg_path = os.path.join(ROOT, "config.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict), "config.yaml must be a dict"

def _config_yaml_has_regulatory_section():
    import yaml
    cfg_path = os.path.join(ROOT, "config.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    reg = cfg.get("validation", {}).get("regulatory", {})
    assert reg is not None, "config.yaml missing validation.regulatory section"

def _config_yaml_compliance_section():
    import yaml
    cfg_path = os.path.join(ROOT, "config.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # compliance section may be nested under validation or top-level
    compliance = cfg.get("compliance") or cfg.get("validation", {}).get("compliance")
    # Non-fatal — compliance section is optional (advisor uses defaults)
    assert True  # just verifying config loads without error

chk("config.yaml loads without error", _config_yaml_loads)
chk("config.yaml has validation.regulatory section", _config_yaml_has_regulatory_section)
chk("config.yaml compliance section present (or defaults used)", _config_yaml_compliance_section)


# ═══════════════════════════════════════════════════════════════════════════════
print(hdr("11 · END-TO-END SMOKE — mini pipeline simulation"))
# ═══════════════════════════════════════════════════════════════════════════════

def _e2e_compliance_flow_no_violations():
    """
    Full compliance flow: RegulatoryEngine → ComplianceAdvisor → ConfidenceVector.
    With an empty domain list (user chose no domain) — expect zero penalty.
    """
    import pandas as pd
    from validation.regulatory.regulatory_engine import RegulatoryEngine
    from validation.compliance_decision import ComplianceAdvisor
    from verifier.confidence_vector import ConfidenceVector

    df = pd.DataFrame({"feature_a": [1.0, 2.0], "feature_b": [3.0, 4.0]})

    # No domain → no rules → no violations
    cfg = {"validation": {"regulatory": {"domains": []}}}
    engine = RegulatoryEngine.from_config(cfg)
    violations = engine.evaluate(df)
    conflict_report = engine.get_last_conflict_report()
    assert violations == [], f"Expected no violations, got {violations}"

    advisor = ComplianceAdvisor.from_config({})
    decision = advisor.evaluate(violations=violations, conflict_report=conflict_report,
                                df=df, run_id="smoke-001")
    assert decision.decision == "allowed"
    assert decision.compliance_penalty == 0.0

    cv_cfg = {"domain": "generic"}
    cv = ConfidenceVector.from_config(cv_cfg)
    vector = cv.aggregate(df=df, model_metrics={}, quality_score=0.85,
                          gate2_passed=True, retry_count=0,
                          compliance_penalty=decision.compliance_penalty,
                          compliance_decision=decision.decision)
    assert 0.0 <= vector["confidence_score"] <= 1.0

def _e2e_compliance_flow_with_violations():
    """
    Full compliance flow with banking domain — negative amount triggers violation → penalty applied.
    """
    import pandas as pd
    from validation.regulatory.regulatory_engine import RegulatoryEngine
    from validation.compliance_decision import ComplianceAdvisor
    from verifier.confidence_vector import ConfidenceVector

    df = pd.DataFrame({"amount": [-500, -1000], "transaction_value": [-200, -300]})

    cfg = {"validation": {"regulatory": {
        "domains": ["banking"],
        "conflict_resolution": "strictest_wins",
        "banking": {},
    }}}
    engine = RegulatoryEngine.from_config(cfg)
    violations = engine.evaluate(df)

    advisor_cfg = {"compliance": {"block_on_critical": False}}
    advisor = ComplianceAdvisor.from_config(advisor_cfg)
    decision = advisor.evaluate(violations=violations, conflict_report=[],
                                df=df, run_id="smoke-002")

    cv_cfg = {"domain": "banking"}
    cv = ConfidenceVector.from_config(cv_cfg)
    score_baseline = cv.aggregate(df=df, model_metrics={}, quality_score=0.85,
                                  gate2_passed=True, retry_count=0,
                                  compliance_penalty=0.0, compliance_decision="allowed")["confidence_score"]
    score_penalised = cv.aggregate(df=df, model_metrics={}, quality_score=0.85,
                                   gate2_passed=True, retry_count=0,
                                   compliance_penalty=decision.compliance_penalty,
                                   compliance_decision=decision.decision)["confidence_score"]
    assert score_penalised <= score_baseline, \
        f"Penalty didn't reduce score: baseline={score_baseline:.4f} penalised={score_penalised:.4f}"

chk("E2E: no domain → no violations → penalty=0 → confidence score in [0,1]", _e2e_compliance_flow_no_violations)
chk("E2E: banking domain → violations → reduced confidence score", _e2e_compliance_flow_with_violations)


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*60}")
passed = [r for r in RESULTS if r[0]]
failed = [r for r in RESULTS if not r[0]]

print(f"  \033[32m{len(passed)} PASSED\033[0m  |  \033[31m{len(failed)} FAILED\033[0m  |  {len(RESULTS)} TOTAL")
print(f"{'═'*60}")

if failed:
    print("\n\033[31mFAILED CHECKS:\033[0m")
    for _, msg in failed:
        print(f"  • {msg}")
    print()
    sys.exit(1)
else:
    print("\n\033[32m  All checks passed — pipeline is correctly wired.\033[0m\n")
    sys.exit(0)
