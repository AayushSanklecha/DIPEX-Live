"""
ingestion/pipeline_bridge.py
-----------------------------
Bridges the UDIL (Universal Data Intake) layer to ALL downstream DIPEX operations.

After any ISSF snapshot is produced (from any source — file, API, DB, stream),
this module routes the clean DataFrame through:

  1. Preprocessing          — clean, impute, scale, encode (sklearn Pipeline)
  2. Deterministic Validation — hard gate, schema, null, range, integrity
  3. Profiling              — statistical profile report
  4. Governance             — PII scan, policy enforcement, data catalog update
  5. Statistical Analysis   — descriptive, hypothesis tests, drift vs. prior
  6. ML Modeling            — train/evaluate if target_col provided
  7. Executive Reporting    — generate HTML report
  8. Audit Trail            — emit structured audit event

All stages are fault-isolated: a failure in stage N does not block stages N+1..8.
Each stage result is returned in a PipelineResult dataclass.
"""

from __future__ import annotations

import logging
import os
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from ingestion.issf import ISSFSnapshot

logger = logging.getLogger("dipex.ingestion.pipeline_bridge")


# ── Dtype restoration helper (Bug 6: int→float64 NaN coercion) ───────────────

def _restore_integer_dtypes(
    df: "pd.DataFrame", original_dtypes: "Dict[str, Any]"
) -> "pd.DataFrame":
    """
    After imputation, pandas promotes int columns to float64 to accommodate NaN.
    This function converts them back to pd.Int64Dtype() (nullable integer) if all
    non-null values are whole numbers AND the original dtype was integer.
    """
    for col, orig_dtype in original_dtypes.items():
        if col not in df.columns:
            continue
        orig_str = str(orig_dtype)
        if orig_str.startswith(("int", "uint", "Int")):
            current = str(df[col].dtype)
            if current.startswith("float"):
                series = df[col].dropna()
                if not series.empty and (series % 1 == 0).all():
                    try:
                        df[col] = df[col].astype(pd.Int64Dtype())
                    except (ValueError, TypeError):
                        pass  # Leave as float if conversion fails
    return df


# ── Stage Result ──────────────────────────────────────────────────────────────

@dataclass
class StageResult:
    stage: str
    status: str        # PASS | FAIL | SKIP | WARN
    elapsed_ms: float
    output: Any = None
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {"stage": self.stage, "status": self.status,
                "elapsed_ms": round(self.elapsed_ms, 2), "error": self.error}


@dataclass
class PipelineResult:
    run_id: str
    dataset_id: str
    snapshot_id: str
    started_at: str
    completed_at: Optional[str] = None
    stages: List[StageResult] = field(default_factory=list)
    preprocessed_df: Optional[pd.DataFrame] = None
    model_metrics: Optional[Dict] = None
    report_path: Optional[str] = None
    gate_decision: str  = "PENDING"    # PASS | FAIL | WARN
    gate1_decision: str = "PENDING"    # PASS | REJECT | ADVISORY_REJECT
    gate2_decision: str = "PENDING"    # PASS | REJECT | NOT_RUN
    confidence_vector: Optional[Dict] = None
    retry_count: int = 0
    analytics_result: Optional[Dict] = None   # AI & Analytics Layer output
    governance_report: Optional[Dict] = None  # PII & policy enforcement report
    regulatory_report: Optional[List[Dict]] = None # Compliance Engine Violations/Risk Flags
    quarantine_df: Optional[pd.DataFrame] = None   # rows too null to process
    cleaning_audit: Optional[Dict] = None           # full audit of every data fix applied

    @property
    def is_success(self) -> bool:
        fails = [s for s in self.stages if s.status == "FAIL"]
        return not any(s.stage in ("validation", "governance") and s.status == "FAIL" for s in fails)

    def summary(self) -> Dict:
        return {
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "snapshot_id": self.snapshot_id,
            "gate_decision": self.gate_decision,
            "stages": [s.to_dict() for s in self.stages],
            "report_path": self.report_path,
            "model_metrics": self.model_metrics,
            "governance_report": self.governance_report,
            "regulatory_report": self.regulatory_report,
            "silver_id": getattr(self, 'silver_id', None),
            "gold_artefacts": len(getattr(self, 'gold_artefacts', [])),
            "quarantine_rows": len(self.quarantine_df) if self.quarantine_df is not None else 0,
            "cleaning_audit": self.cleaning_audit,
        }


# ── Pipeline Bridge ───────────────────────────────────────────────────────────

class PipelineBridge:
    """
    Connects UDIL output (ISSFSnapshot) to all downstream DIPEX operations.

    Usage::

        bridge = PipelineBridge(config)
        result = bridge.run(snapshot, target_col="churn")
        print(result.summary())
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        self.config = config or {}
        self.run_id = str(uuid.uuid4())
        self._rl_plan: Optional[Dict] = None   # populated by RLOrchestrator each run
        self._rl_recs: Optional[Dict[str, Any]] = None  # actively used by Preprocessing stages
        # ── Layer isolation ─────────────────────────────────────────────────
        try:
            from ingestion.data_layers import LayerManager
            dl_cfg  = self.config.get("data_layers", {})
            _base   = dl_cfg.get("bronze_dir", "data").replace("/bronze", "") or "data"
            self._lm: Optional[Any] = LayerManager(base_dir=_base)
        except Exception:  # noqa: BLE001
            self._lm = None

    def run(
        self,
        snapshot: ISSFSnapshot,
        target_col: Optional[str] = None,
        run_id: Optional[str] = None,
        skip_stages: Optional[List[str]] = None,
    ) -> PipelineResult:
        """
        Execute the full downstream pipeline from an ISSF snapshot.

        Parameters
        ----------
        snapshot    : ISSFSnapshot produced by UniversalIntake
        target_col  : column name for supervised ML (optional)
        run_id      : override run ID (else auto-generated)
        skip_stages : list of stage names to skip (e.g. ['modeling'])
        """
        self.run_id = run_id or str(uuid.uuid4())
        skip = set(skip_stages or [])
        df = snapshot.data
        self._run_start = time.perf_counter()  # for RL orchestrator feedback

        # ── RL / Experience attributes — must be initialised before any stage ─
        self._episode: str = self.run_id    # used by ReinforcementUpdateEngine
        self._rl_recs: Optional[Dict[str, Any]] = None  # To be filled at start of run()
        self._current_retry_count: int = 0  # used by _stage_confidence_vector

        result = PipelineResult(
            run_id=self.run_id,
            dataset_id=snapshot.dataset_id,
            snapshot_id=snapshot.snapshot_id,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        logger.info(
            "[%s] PipelineBridge started — dataset=%s rows=%d source=%s",
            self.run_id[:8], snapshot.dataset_id, snapshot.row_count, snapshot.source_type,
        )

        # ── RL Orchestrator: decide compute budget for this run ────────────
        try:
            from modeling.rl_orchestrator import get_rl_orchestrator
            rl_orch = get_rl_orchestrator()
            sla = self.config.get("orchestrator", {}).get("sla_minutes", 30.0)
            self._rl_plan = rl_orch.get_plan(
                sla_minutes=sla,
                row_count=snapshot.row_count,
                drift_detected=False,
            )
            logger.info(
                "[%s] RL Orchestrator plan: profile=%s preprocess=%s model=%s validation=%s",
                self.run_id[:8],
                self._rl_plan.get("profile_depth"),
                self._rl_plan.get("preprocess_depth"),
                self._rl_plan.get("model_complexity"),
                self._rl_plan.get("validation_depth"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("RL Orchestrator unavailable: %s", exc)
            self._rl_plan = None

        if df is None or df.empty:
            logger.warning("[%s] Empty DataFrame in snapshot — aborting pipeline.", self.run_id[:8])
            result.gate_decision = "FAIL"
            result.completed_at = datetime.now(timezone.utc).isoformat()
            return result

        # ── Step 0: RL Context — call recommend() to arm the learning loop ────
        # IMPORTANT: recommend() must be called HERE (before pipeline runs) so
        # that agent._current_state/_current_action_indices is set.
        # record_outcome() at Stage 11 reads those attributes to compute reward.
        self._rl_recs: Optional[Dict[str, Any]] = None
        self._gate_decision_cache: str = "WARN"       # updated after Gate 2
        self._model_metrics_cache: Dict = {}          # updated after modeling stage
        try:
            from learning.rl_agent.agent import PPOAgent
            from learning.rl_agent.state_encoder import StateEncoder
            _ppo_agent = PPOAgent.from_config(self.config)
            # Build context from snapshot for recommend()
            _ctx = {
                "n_rows":       snapshot.row_count,
                "n_cols":       len(df.columns),
                "null_rate":    float(df.isnull().mean().mean()),
                "anomaly_rate": 0.0,   # updated by drift stage later
                "drift_psi":    0.0,   # updated by drift stage later
                "data_health":  50.0,  # updated by analyst brain later
                "domain":       str(snapshot.meta.get("domain", "generic"))
                                if hasattr(snapshot, "meta") else "generic",
                "target_col":   target_col,
                "prior_confidence": self.config.get("pipeline", {}).get(
                    "prior_confidence", 0.5),
                "quarantine_frac":  0.0,
                "retry_count":      self._current_retry_count,
            }
            _action = _ppo_agent.recommend(_ctx)
            self._rl_recs = _action.to_dict()  # steers AnalystBrain + MDE
            # Persist the live agent instance on self so Stage 11 reuses it
            # (avoids re-loading checkpoint and losing _current_state)
            self._ppo_agent_instance = _ppo_agent
            logger.info(
                "[%s] RL recommend(): episode=%d shadow=%s "
                "imputation=%s cv=%s/%s conf=%.2f complexity=%s",
                self.run_id[:8],
                _ppo_agent._episode_count,
                _ppo_agent.in_shadow_mode,
                _action.imputation, _action.cv_folds, _action.cv_strategy,
                _action.confidence_threshold, _action.model_complexity,
            )
        except Exception as _e:
            logger.debug("[%s] RL context bootstrap skipped: %s", self.run_id[:8], _e)
            self._ppo_agent_instance = None

        # ── Build Silver ImmutableDataFrame for Gold derivation ───────────────
        _silver_imm = None
        if self._lm is not None:
            try:
                from ingestion.data_layers import ImmutableDataFrame
                _silver_imm = ImmutableDataFrame(
                    df.copy(), layer="silver", dataset_id=snapshot.dataset_id
                )
                logger.debug(
                    "[%s] Silver ImmutableDataFrame ready for Gold derivation",
                    self.run_id[:8],
                )
            except Exception as _e:  # noqa: BLE001
                logger.warning("Gold isolation unavailable: %s", _e)
                _silver_imm = None

        def _stage_on_gold(stage_name: str, fn, *fn_args):
            """Run fn on a Gold copy if LayerManager is available; else run on df."""
            if _silver_imm is not None and self._lm is not None:
                try:
                    art = self._lm.derive_gold(
                        _silver_imm,
                        dataset_id=f"{snapshot.dataset_id}_{stage_name}",
                        component=stage_name,
                        transform_fn=lambda gdf: fn(gdf, *fn_args[1:]),
                        step_name=stage_name,
                        source_snapshot_id=snapshot.snapshot_id,
                    )
                    result.gold_artefacts.append({
                        "stage": stage_name,
                        "lineage_id": art.lineage.lineage_id,
                        "checksum": art.checksum,
                        "shape": list(art.data.shape),
                    })
                    return art.data
                except Exception as _e:  # noqa: BLE001
                    logger.warning(
                        "[%s] Gold derivation failed for '%s': %s — fallback to direct",
                        self.run_id[:8], stage_name, _e,
                    )
            return fn(df, *fn_args[1:])

        # ── Stage 0: Streaming Window Engine ─────────────────────────────────
        # Activated for Kafka / stream sources — partitions df into windows.
        # For non-streaming sources this stage is a no-op (skipped).
        source_type = getattr(snapshot, "source_type", "").lower()
        if source_type in ("kafka", "stream", "streaming") and "streaming_window" not in skip:
            window_out = self._run_stage(
                result, "streaming_window", skip,
                self._stage_streaming_window, df,
            )
            # Use first window batch if windows were produced, else keep original df
            if isinstance(window_out, list) and window_out:
                df = window_out[0].data
                logger.info(
                    "[%s] StreamingWindowEngine: using window 1/%d rows=%d",
                    self.run_id[:8], len(window_out), len(df),
                )
        elif "streaming_window" not in skip:
            result.stages.append(StageResult("streaming_window", "SKIP", 0.0))

        # ── Stage 0.4: Analyst Intelligence Brain ─────────────────────────────
        # The "senior data analyst brain" — runs FIRST on raw data.
        # Determines for EVERY column:
        #   • Semantic type (ID / datetime / currency / categorical / etc.)
        #   • Transform strategy (log1p / sqrt / yeo-johnson / none)
        #   • Outlier policy (IQR / winsorise / flag)
        #   • Imputation hint (median / mode / knn / mice / forward_fill)
        #   • Business rule violations (negative age, impossible dates, etc.)
        # Results are attached to df.attrs so all downstream stages can use them.
        brain_out = self._run_stage(
            result, "analyst_brain", skip,
            self._stage_analyst_brain, df, target_col,
        )
        if isinstance(brain_out, dict):
            df = brain_out.get("df", df)
            # Expose brain report on result for API / reporting access
            if not hasattr(result, "brain_report"):
                object.__setattr__(result, "brain_report", brain_out.get("report", {}))

        # ── Stage 0.5: Robust Data Triage ─────────────────────────────────────
        # Runs BEFORE preprocessing to handle all real-world data pathologies:
        # high-null cols, mixed types, zero variance, high cardinality, skew, imbalance.
        triage_out = self._run_stage(
            result, "triage", skip,
            self._stage_triage, df, target_col,
        )
        if isinstance(triage_out, pd.DataFrame):
            df = triage_out
        # Carry triage imbalance info for use in the modeling stage
        self._triage_imbalance_info: Dict[str, Any] = (
            triage_out.attrs.get("triage_imbalance", {})
            if isinstance(triage_out, pd.DataFrame) else {}
        )

        # ── Stage 0.6: Missing Data Engine ────────────────────────────────────
        # Unified intelligent missing-data handler. Runs after Triage so that
        # the gross structural issues (mixed types, zero-variance) are fixed first.
        # Classifies MCAR/MAR/MNAR per column, applies correct imputation strategy,
        # quarantines rows that are ≥80% null even after imputation.
        mde_out = self._run_stage(
            result, "missing_data_engine", skip,
            self._stage_missing_data_engine, df, target_col,
        )
        if isinstance(mde_out, dict):
            df               = mde_out.get("df", df)
            result.quarantine_df  = mde_out.get("quarantine_df")
            result.cleaning_audit = mde_out.get("report")

        # ── Early exit: if triage+cleaning produced an empty DataFrame ────────
        if df is None or df.empty or len(df.columns) == 0:
            logger.warning(
                "[%s] DataFrame empty after triage/cleaning — skipping remaining stages.",
                self.run_id[:8],
            )
            result.gate_decision = "FAIL"
            result.completed_at = datetime.now(timezone.utc).isoformat()
            self._audit(result, snapshot)
            return result

        # ── Stage 1: Preprocessing ────────────────────────────────────────────
        # Stage 0.75: Missing pattern analysis — runs before main imputation
        # so the DataCleaner can use the correct strategy per column.
        mp_out = self._run_stage(
            result, "missing_patterns", skip,
            self._stage_missing_patterns, df,
        )
        if isinstance(mp_out, pd.DataFrame):
            df = mp_out

        # Save raw df BEFORE preprocessing so the regulatory engine always
        # evaluates original column values (strings, real amounts, etc.)
        self._df_raw_for_compliance = df.copy()

        prep_out = self._run_stage(result, "preprocessing", skip,
                             self._stage_preprocess, df, target_col)
        if prep_out is not None:
            df = prep_out

        # ── Stage 1.5: Schema Drift Detection ─────────────────────────────────
        # Detects column additions/removals/dtype changes vs. prior run.
        self._run_stage(
            result, "drift_detection", skip,
            self._stage_drift, df, snapshot.dataset_id,
        )

        # ── Stage 2: Hard Gate 1 — Deterministic Validation (NON-BLOCKING) ──────
        # advisory_mode=True (default) means gate flags issues but NEVER halts.
        # All downstream stages always run — the gate is purely informational.
        gate1_out = self._run_stage(result, "validation", skip,
                                    self._stage_validate, df, snapshot.dataset_id)
        if gate1_out is False:
            # Strict-mode reject (advisory_mode=False in config). Even so, we
            # continue — pipeline does not halt. Status is recorded for audit.
            result.gate1_decision = "ADVISORY_REJECT"
            logger.warning(
                "[%s] Hard Gate 1 ADVISORY_REJECT — pipeline continues in advisory mode. "
                "All downstream stages will still execute.",
                self.run_id[:8],
            )
        else:
            result.gate1_decision = "PASS"

        # ── Stage 3: Data Profiling ───────────────────────────────────────
        profile_result = self._run_stage(result, "profiling", skip,
                                         self._stage_profile, df, self.run_id)
        drift_psi: Optional[float] = None
        if isinstance(profile_result, dict):
            drift_psi = profile_result.get("psi_score")

        # ── Stage 4: AI & Analytics Service Layer (AutoEDA + FE + Insights + LLM) ───
        analytics_out = self._run_stage(
            result, "analytics", skip,
            self._stage_analytics, df, target_col,
        )
        if isinstance(analytics_out, dict):
            result.analytics_result = analytics_out
            # Promote enriched df if feature engineering ran
            enriched = analytics_out.get("_enriched_df")
            if enriched is not None and not enriched.empty:
                df = enriched

        # ── Stage 5: Governance ───────────────────────────────────────────────
        gov_out = self._run_stage(result, "governance", skip, self._stage_governance, df, snapshot.dataset_id)
        if isinstance(gov_out, dict):
            result.governance_report = gov_out
            # CRITICAL: if governance redacted PII, promote the cleansed DataFrame
            # so downstream stages (stats, leakage, model, report) never see raw PII.
            _cleansed = gov_out.pop("_cleansed_df", None)
            if _cleansed is not None and not _cleansed.empty:
                df = _cleansed
                logger.info("[Bridge] Governance redaction applied — df replaced with cleansed copy.")

        # ── Stage 5.2: Statistical Analysis ───────────────────────────────────
        self._run_stage(result, "statistics", skip, self._stage_stats, df, target_col)

        # ── Stage 5.5: Leakage Detection ──────────────────────────────────────
        # Detect and remove data leakage before the model sees the features.
        if target_col and target_col in df.columns:
            leak_out = self._run_stage(
                result, "leakage_detection", skip,
                self._stage_leakage, df, target_col,
            )
            if isinstance(leak_out, pd.DataFrame):
                df = leak_out

        # ── Stage 5.7: Multicollinearity Check ─────────────────────────────────
        # VIF-based removal of collinear features before modeling.
        mc_out = self._run_stage(
            result, "multicollinearity", skip,
            self._stage_multicollinearity, df, target_col,
        )
        if isinstance(mc_out, pd.DataFrame):
            df = mc_out

        # ── Stage 6: ML Modeling ──────────────────────────────────────────────
        model_metrics: Dict[str, Any] = {}
        if target_col and target_col in df.columns:
            metrics_out = self._run_stage(result, "modeling", skip,
                                          self._stage_model, df, target_col)
            if isinstance(metrics_out, dict):
                model_metrics = metrics_out
                result.model_metrics = model_metrics

        # ── Stage 6.5: Confidence Calibration ────────────────────────────────
        # Calibrate raw model probabilities to reduce overconfidence.
        if model_metrics and target_col and target_col in df.columns:
            cal_out = self._run_stage(
                result, "calibration", skip,
                self._stage_calibrate, df, target_col, model_metrics,
            )
            if isinstance(cal_out, dict) and cal_out:
                # Merge calibration metadata into model_metrics
                model_metrics["calibration"] = cal_out
                result.model_metrics = model_metrics

        # ── Stage 6.7: Feature Importance Stability Check ─────────────────────
        # Compare feature importances with prior run to detect upstream data issues.
        if model_metrics:
            stability_out = self._run_stage(
                result, "feature_stability", skip,
                self._stage_feature_stability, model_metrics, snapshot.dataset_id,
            )
            if isinstance(stability_out, dict) and stability_out:
                model_metrics["feature_stability"] = stability_out
                result.model_metrics = model_metrics

        # ── Stage 7: Hard Gate 2 — Independent Statistical Verifier ──────────
        gate2_ok = self._run_stage(result, "verification", skip,
                                    self._stage_verify, df, model_metrics, self.run_id)
        result.gate2_decision = "PASS" if gate2_ok is True else "REJECT"

        # ── Stage 8: Confidence Vector Aggregation ────────────────────────────
        conf_vector = self._run_stage(result, "confidence_vector", skip,
                                       self._stage_confidence_vector,
                                       df, model_metrics, snapshot, gate2_ok)
        confidence_score: float = 0.5
        if isinstance(conf_vector, dict):
            result.confidence_vector = conf_vector
            confidence_score = float(conf_vector.get("confidence_score", 0.5))

        # ── Stage 9: Intelligent Retry Engine ─────────────────────────────────
        # B8 fix: multi-level domain lookup with fallbacks
        _pipe_cfg = self.config.get("pipeline", {})
        domain = (
            _pipe_cfg.get("domain")
            or _pipe_cfg.get("regulatory", {}).get("domain")
            or self.config.get("domain")
            or "default"
        )
        conf_thresh = float(
            self.config.get("pipeline", {}).get("confidence", {}).get(
                "threshold",
                {"banking": 0.85, "healthcare": 0.90, "default": 0.70}.get(domain, 0.70),
            )
        )
        # Only run retries if confidence_vector succeeded (not None) and gate2 passed
        if conf_vector is not None and confidence_score < conf_thresh and result.gate2_decision == "PASS":
            self._retry_engine_loop(
                result=result, snapshot=snapshot, df=df,
                target_col=target_col, skip=skip,
                confidence_score=confidence_score, conf_thresh=conf_thresh,
            )
            if isinstance(result.confidence_vector, dict):
                confidence_score = float(
                    result.confidence_vector.get("confidence_score", confidence_score)
                )

        # ── Stage 10: Approved Results + Experience Memory ────────────────────
        approved = (
            result.gate1_decision == "PASS"
            and result.gate2_decision == "PASS"
            and confidence_score >= conf_thresh
        )
        if approved:
            self._run_stage(result, "experience_memory", skip,
                            self._stage_record_experience,
                            snapshot, conf_vector or {}, model_metrics)

        # ── Stage 11: RL Update (Gate 1 must have passed) ─────────────────────
        q_rows = len(result.quarantine_df) if getattr(result, "quarantine_df", None) is not None else 0
        tot_rows = max(getattr(snapshot, "row_count", 0) or (q_rows + (len(df) if df is not None else 0)), 1)
        q_frac = float(q_rows / tot_rows)
        
        # Extract data health if AnalystBrain was run successfully
        data_health: float = 50.0 # default baseline
        try:
            brain_report = df.attrs.get("brain_report") if df is not None else None
            if brain_report and "data_health_score" in brain_report:
                data_health = float(brain_report["data_health_score"])
        except Exception:
            pass
        
        self._run_stage(result, "rl_update", skip,
                        self._stage_rl_update, snapshot.snapshot_id, drift_psi, q_frac, data_health)

        # ── Stage 12: Executive Reporting ─────────────────────────────────────
        report_path = self._run_stage(result, "reporting", skip,
                                       self._stage_report, result)
        if isinstance(report_path, str):
            result.report_path = report_path

        # ── Stage 13: Audit Trail ─────────────────────────────────────────────
        hard_fails = [s for s in result.stages
                      if s.status == "FAIL" and s.stage in (
                          "validation", "governance", "verification", 
                          "confidence_vector", "rl_update", "proposal"
                      )]
                      
        if hard_fails:
            result.gate_decision = "FAIL"
        elif result.gate1_decision == "REJECT" or result.gate2_decision == "REJECT":
            result.gate_decision = "FAIL"
        elif confidence_score < conf_thresh:
            result.gate_decision = "WARN"
        else:
            result.gate_decision = "PASS"
        result.preprocessed_df = df
        result.completed_at   = datetime.now(timezone.utc).isoformat()
        self._audit(result, snapshot)

        logger.info(
            "[%s] Pipeline completed — gate1=%s gate2=%s conf=%.3f decision=%s stages=%d",
            self.run_id[:8], result.gate1_decision, result.gate2_decision,
            confidence_score, result.gate_decision, len(result.stages),
        )
        return result


    # ── Stage runner  ─────────────────────────────────────────────────────────

    def _run_stage(self, result: PipelineResult, name: str, skip: set,
                   fn, *args) -> Any:
        if name in skip:
            result.stages.append(StageResult(name, "SKIP", 0.0))
            return None
        t0 = time.perf_counter()
        try:
            out = fn(*args)
            elapsed = (time.perf_counter() - t0) * 1000
            result.stages.append(StageResult(name, "PASS", elapsed, output=out))
            logger.info("[%s] Stage '%s' PASS (%.0fms)", self.run_id[:8], name, elapsed)
            return out
        except Exception as exc:  # noqa: BLE001
            elapsed = (time.perf_counter() - t0) * 1000
            msg = f"{type(exc).__name__}: {exc}"
            result.stages.append(StageResult(name, "FAIL", elapsed, error=msg))
            logger.error("[%s] Stage '%s' FAIL: %s", self.run_id[:8], name, msg)
            logger.debug(traceback.format_exc())
            return None

    # ── Universal DataFrame guard ──────────────────────────────────────────────

    def _guard_df(self, df: "Optional[pd.DataFrame]", stage_name: str) -> bool:
        """
        Returns False if the DataFrame cannot support this stage at all.
        0-row DataFrames return True — schema/profiling can still run on structure.
        """
        if df is None:
            logger.warning("[%s] Stage '%s': df is None — skipped.", self.run_id[:8], stage_name)
            return False
        if len(df.columns) == 0:
            logger.warning("[%s] Stage '%s': df has 0 columns — skipped.", self.run_id[:8], stage_name)
            return False
        if len(df) == 0:
            logger.info(
                "[%s] Stage '%s': df has 0 rows — limited analysis only (schema/profiling still run).",
                self.run_id[:8], stage_name,
            )
        elif len(df) < 5:
            logger.info(
                "[%s] Stage '%s': df has only %d row(s) — some analytics will be limited.",
                self.run_id[:8], stage_name, len(df),
            )
        return True

    # ── Stage implementations ─────────────────────────────────────────────────

    def _stage_analyst_brain(
        self, df: pd.DataFrame, target_col: "Optional[str]"
    ) -> "Dict[str, Any]":
        """
        Stage 0.4 — Senior Expert Analyst Intelligence Brain.

        Analyses every single column with the reasoning of a seasoned data
        analyst:
          • Detects semantic type (ID / datetime / email / currency / categorical…)
          • Chooses the correct transform strategy (log1p / sqrt / yeo-johnson)
          • Selects the best outlier policy (IQR / winsorise / flag)
          • Determines the right imputation method per column (MCAR→median,
            MAR→KNN, MNAR→MICE, datetime→forward_fill, categorical→mode)
          • Checks every business rule (negative ages/salaries, impossible dates,
            future hire dates, percentage out of range, etc.)
          • Detects exact duplicate columns
          • Flags near-perfect correlations (multicollinearity)
          • Identifies potential data leakage columns
          • Auto-suggests a target variable if none is given
          • Computes data health score 0–100

        Every single decision is logged in plain English — full audit trail.
        Results are attached to df.attrs for ALL downstream stages to use.
        Returns dict {"df": annotated_df, "report": brain_report_dict}.
        """
        if not self._guard_df(df, "analyst_brain"):
            return {"df": df, "report": {}}
        try:
            from preprocessing.analyst_brain import AnalystBrain
            brain = AnalystBrain.from_config(self.config)
            annotated_df, brain_report = brain.run(
                df, run_id=self.run_id, target_col=target_col,
                rl_recommendations=self._rl_recs
            )
            report_dict = brain_report.to_dict()

            # Log key findings so they appear in the stage log immediately
            drops = [col for col, d in brain_report.column_decisions.items() if d.should_drop]
            violations = sum(
                len(d.violations) for d in brain_report.column_decisions.values()
            )
            logger.info(
                "[%s] AnalystBrain: domain=%s health=%.1f cols=%d "
                "drops_recommended=%d violations=%d",
                self.run_id[:8],
                brain_report.detected_domain,
                brain_report.data_health_score,
                len(df.columns),
                sum(1 for d in brain_report.column_decisions.values() if d.should_drop),
                sum(len(d.violations) for d in brain_report.column_decisions.values()),
            )
            if drops:
                logger.info("[%s] Brain recommends dropping: %s", self.run_id[:8], drops)
            for note in brain_report.dataset_level_notes:
                logger.info("[%s] Brain note: %s", self.run_id[:8], note)

            return {"df": annotated_df, "report": report_dict}

        except ImportError:
            logger.warning("[%s] AnalystBrain not available — skipping (non-fatal).", self.run_id[:8])
            return {"df": df, "report": {}}
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[%s] AnalystBrain failed (non-fatal): %s — continuing without brain annotations.",
                self.run_id[:8], exc,
            )
            return {"df": df, "report": {}}

    def _stage_missing_data_engine(
        self, df: pd.DataFrame, target_col: "Optional[str]"
    ) -> "Dict[str, Any]":
        """
        Stage 0.6 — Missing Data Intelligence Engine.

        Runs after RobustTriage (which fixes types/structure) and before
        MissingPatternAnalyzer. Handles:
          • String/numeric sentinel → NaN replacement
          • >90%-null column dropping
          • Per-column MCAR/MAR/MNAR classification
          • Strategy-correct imputation (median→KNN→MICE)
          • MNAR indicator column addition
          • ≥80%-null row quarantine

        Non-fatal: any failure returns the original df unchanged.
        """
        if not self._guard_df(df, "missing_data_engine"):
            return {"df": df, "quarantine_df": pd.DataFrame(), "report": {}}
        try:
            from preprocessing.missing_data_engine import MissingDataEngine
            engine = MissingDataEngine(config=self.config)

            # Capture original dtypes BEFORE imputation so we can restore int columns after
            original_dtypes = df.dtypes.to_dict()

            clean_df, quarantine_df, md_report = engine.run(
                df, run_id=self.run_id, target_col=target_col,
                rl_recommendations=self._rl_recs
            )

            # Restore integer columns that were promoted to float64 by NaN injection
            clean_df = _restore_integer_dtypes(clean_df, original_dtypes)

            logger.info(
                "[%s] MissingDataEngine: original=%s final=%s "
                "dropped_cols=%d quarantine=%d sentinels=%d lib=%s",
                self.run_id[:8],
                md_report.original_shape, md_report.final_shape,
                len(md_report.columns_dropped),
                md_report.quarantine_rows,
                md_report.sentinel_replacements_total,
                md_report.imputation_library_used,
            )
            return {
                "df": clean_df,
                "quarantine_df": quarantine_df,
                "report": md_report.to_dict(),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[%s] MissingDataEngine failed (non-fatal): %s — df unchanged.",
                self.run_id[:8], exc,
            )
            return {"df": df, "quarantine_df": pd.DataFrame(), "report": {"error": str(exc)}}

    def _stage_unsupervised(self, df: pd.DataFrame) -> "Dict[str, Any]":
        """
        Unsupervised analysis path — runs when:
          • No target_col was provided
          • Modeling guards fire (too few rows, no numeric cols, single class)

        Executes:
          1. Isolation Forest — anomaly detection (contamination=5%)
          2. KMeans — cluster summary (2-5 clusters, adaptive)

        Never crashes: all exceptions return empty results.
        """
        results: "Dict[str, Any]" = {"mode": "unsupervised"}
        num_df = df.select_dtypes(include="number")
        if num_df.empty or len(df) < 5:
            logger.info("[%s] Unsupervised: insufficient numeric data (%d rows, %d numeric cols).",
                        self.run_id[:8], len(df), len(num_df.columns))
            results["info"] = "insufficient_data_for_unsupervised"
            return results

        X = num_df.fillna(num_df.median())

        # ── 1. Isolation Forest anomaly detection ─────────────────────────────
        try:
            from sklearn.ensemble import IsolationForest
            iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=50)
            preds = iso.fit_predict(X)
            results["anomaly_count"] = int((preds == -1).sum())
            results["anomaly_pct"]   = round(float((preds == -1).mean()) * 100, 2)
            logger.info(
                "[%s] Unsupervised IsoForest: %d anomalies (%.1f%%)",
                self.run_id[:8], results["anomaly_count"], results["anomaly_pct"],
            )
        except Exception as exc:
            logger.debug("[%s] IsolationForest failed: %s", self.run_id[:8], exc)
            results["anomaly_error"] = str(exc)

        # ── 2. KMeans clustering ──────────────────────────────────────────────
        try:
            from sklearn.cluster import KMeans
            n_clusters = max(2, min(5, len(df) // max(10, len(df) // 10)))
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            import pandas as _pd
            cluster_sizes = _pd.Series(labels).value_counts().to_dict()
            results["n_clusters"]    = n_clusters
            results["cluster_sizes"] = {str(k): int(v) for k, v in cluster_sizes.items()}
            logger.info(
                "[%s] Unsupervised KMeans: %d clusters — sizes: %s",
                self.run_id[:8], n_clusters, results["cluster_sizes"],
            )
        except Exception as exc:
            logger.debug("[%s] KMeans failed: %s", self.run_id[:8], exc)
            results["cluster_error"] = str(exc)

        return results

    # ── Stage implementations ─────────────────────────────────────────────────


    def _stage_preprocess(self, df: pd.DataFrame, target_col: Optional[str]) -> pd.DataFrame:
        try:
            # ── Optional: apply registered TransformRegistry transforms first ─
            try:
                from transforms.transform_registry import TransformRegistry
                registry = TransformRegistry()
                # Load transforms from config (if any declared)
                declared = (
                    self.config.get("preprocessing", {})
                               .get("transform_registry", [])
                )
                if declared:
                    df = registry.chain(declared, df)
                    logger.info("TransformRegistry: applied %d transforms", len(declared))
            except Exception as exc:  # noqa: BLE001
                logger.debug("TransformRegistry skipped (non-fatal): %s", exc)

            # ── Clean Data (Drop duplicates, Cap Outliers, etc) ───────────
            try:
                from preprocessing.cleaner import DataCleaner
                cleaner = DataCleaner.from_config(self.config)
                df, cl_report = cleaner.clean(df, run_id=self.run_id)
                logger.debug("DataCleaner dropped %d duplicate rows.", cl_report.duplicates_removed)
            except Exception as exc:
                logger.warning("DataCleaner skipped (non-fatal): %s", exc)

            from preprocessing.pipeline_builder import PipelineBuilder
            builder = PipelineBuilder(self.config)
            feature_cols = [c for c in df.columns if c != target_col]

            # ── Guard: PipelineBuilder needs ≥1 numeric column ───────────────
            numeric_feature_cols = [
                c for c in df.select_dtypes(include="number").columns if c != target_col
            ]
            if not numeric_feature_cols:
                logger.warning(
                    "[%s] Preprocessing: 0 numeric features — skipping PipelineBuilder "
                    "(sklearn requires numeric input). Returning cleaned df.",
                    self.run_id[:8],
                )
                return df

            pipe = builder.build(df, target_col=target_col)
            X = df[feature_cols]
            X_t = pipe.fit_transform(X)
            import numpy as np
            feat_names = builder.get_feature_names(pipe, X)
            df_out = pd.DataFrame(X_t, columns=feat_names[:X_t.shape[1]])
            if target_col and target_col in df.columns:
                df_out[target_col] = df[target_col].values

            # ── RL Feature Selector: prune low-value features ─────────────
            try:
                from preprocessing.rl_feature_selector import RLFeatureSelector
                out_feats = [c for c in df_out.columns if c != target_col]
                if out_feats:
                    selector = RLFeatureSelector(out_feats, max_features=len(out_feats))
                    active = selector.get_active_features()
                    keep = [c for c in active if c in df_out.columns]
                    if target_col and target_col in df_out.columns:
                        keep.append(target_col)
                    df_out = df_out[keep]
                    logger.info("[RL] Feature selector kept %d/%d features", len(active), len(out_feats))
            except Exception as exc:  # noqa: BLE001
                logger.debug("RL Feature selector unavailable: %s", exc)

            logger.info("Preprocessing: %d → %d features", len(feature_cols), df_out.shape[1])
            return df_out
        except Exception as exc:
            logger.warning("Preprocessing skipped (non-fatal): %s", exc)
            return df

    def _stage_validate(self, df: pd.DataFrame, dataset_id: str) -> bool:
        # ── RL Threshold Tuner: adapt thresholds per column ───────────────
        try:
            from validation.rl_threshold_tuner import get_rl_tuner
            tuner = get_rl_tuner()
            for col in df.columns:
                default_t = self.config.get("validation", {}).get("null_threshold", 0.10)
                rl_thresh = tuner.get_threshold(dataset_id, col, default=default_t)
                logger.debug("[RL] Threshold for %s::%s → %.4f", dataset_id, col, rl_thresh)
        except Exception:  # noqa: BLE001
            pass

        # ── Hard Gate 1 ────────────────────────────────────────────────────
        try:
            from validation.hard_gate import HardGate
            gate = HardGate.from_config(self.config)
            result_gate = gate.run(df, run_id=self.run_id)
            if result_gate.decision == "REJECT":
                raise RuntimeError(f"Hard gate REJECTED: {result_gate.reason}")
        except ImportError:
            raise RuntimeError("Validation engine missing: HardGate is required for deterministic validation.")

        # ── Regulatory Rule Engine + Compliance Advisor ────────────────────
        try:
            from validation.regulatory.regulatory_engine import RegulatoryEngine
            from validation.compliance_decision import ComplianceAdvisor

            reg_engine = RegulatoryEngine.from_config(self.config)
            # Use raw (pre-preprocessing) df so engine sees original string BICs,
            # real transaction amounts — not label-encoded/scaled numeric values.
            df_for_compliance = getattr(self, "_df_raw_for_compliance", df)
            violations = reg_engine.evaluate(df_for_compliance)
            conflict_report = reg_engine.get_last_conflict_report()

            advisor = ComplianceAdvisor.from_config(self.config)
            decision = advisor.evaluate(
                violations=violations,
                conflict_report=conflict_report,
                df=df,
                run_id=self.run_id,
            )

            # Store on self so _stage_confidence_vector can pick it up
            self._compliance_decision = decision

            # Feature masking: drop PII-violating columns to prevent downstream leakage
            if decision.violating_columns:
                cols_to_drop = [c for c in decision.violating_columns if c in df.columns]
                if cols_to_drop:
                    logger.info(
                        "Dropping %d PII/compliance-violating columns: %s",
                        len(cols_to_drop), cols_to_drop,
                    )
                    df.drop(columns=cols_to_drop, inplace=True)
                    # Audit record for PII column removal
                    audit_record = {
                        "event": "pii_columns_dropped",
                        "columns_removed": cols_to_drop,
                        "run_id": self.run_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "regulatory_engine_version": getattr(reg_engine, "version", "unknown"),
                    }
                    logger.info("AUDIT: %s", audit_record)
                else:
                    logger.info("No PII columns flagged for removal.")

            # Block pipeline entirely on BLOCKED decision (critical violations)
            if decision.decision == "blocked":
                raise RuntimeError(
                    f"Compliance BLOCKED: {decision.n_critical} CRITICAL violation(s). "
                    "Pipeline halted. See audit/compliance.jsonl for details."
                )

            logger.info(
                "[%s] Compliance: decision=%s penalty=%.3f violations(C=%d E=%d W=%d)",
                self.run_id[:8], decision.decision, decision.compliance_penalty,
                decision.n_critical, decision.n_error, decision.n_warning,
            )

        except ImportError as ie:
            logger.debug("RegulatoryEngine/ComplianceAdvisor not available: %s", ie)
            self._compliance_decision = None

        return True

    def _stage_profile(self, df: pd.DataFrame, run_id: str) -> dict:
        try:
            from profiling.profile_report import ProfileReport
            pr = ProfileReport(config=self.config)
            report = pr.generate(df, run_id=run_id)
            return report
        except ImportError:
            logger.warning("ProfileReport not available — skipping profiling")
            return {}

    def _stage_governance(self, df: pd.DataFrame, dataset_id: str) -> dict:
        try:
            from validation.governance.governor import DataGovernor
            governor = DataGovernor(self.config)
            cleansed_df, gov_report = governor.enforce(df, dataset_id=dataset_id)

            # CRITICAL: when policy='redact', governor returns a NEW DataFrame with
            # PII stripped. We must propagate it downstream so the original PII-
            # containing df is NOT used for modeling/reporting.
            # Attach under a private key so the bridge can promote it.
            if gov_report.get("status") == "redacted":
                gov_report["_cleansed_df"] = cleansed_df
                logger.info(
                    "[Governance] PII redacted — cleansed DataFrame will replace source df downstream."
                )

            return gov_report
        except ImportError:
            logger.warning("DataGovernor not available — skipping governance")
            return {}
        except Exception as exc:
            logger.error("Governance stage failed: %s", exc)
            return {"status": "error", "message": str(exc)}

    def _stage_stats(self, df: pd.DataFrame, target_col: Optional[str]) -> dict:
        try:
            from stats.descriptive import DescriptiveStats
            stats_out = DescriptiveStats().analyze(df)
            if target_col and target_col in df.columns:
                try:
                    from stats.hypothesis_tests import run_tests
                    feature_cols = [c for c in df.select_dtypes(include="number").columns if c != target_col]
                    if feature_cols:
                        run_tests(df, target_col=target_col, feature_cols=feature_cols[:10])
                except Exception:  # noqa: BLE001
                    pass
            return stats_out if isinstance(stats_out, dict) else {}
        except ImportError:
            logger.warning("Stats module not available")
            return {}

    def _stage_model(self, df: pd.DataFrame, target_col: str) -> Dict:
        """
        Stage 6 — ML Modeling.

        Four crash guards run before any sklearn/ModelTrainer code:
          1. Too few rows (< 10) → unsupervised analysis
          2. No numeric features → unsupervised analysis
          3. Target column missing from df → unsupervised analysis
          4. Single-class target → unsupervised analysis (can't train classifier)

        Supervised path only runs when ALL four guards pass.
        """
        if not self._guard_df(df, "modeling"):
            return {"error": "empty_dataframe"}

        # ── Guard 1: too few rows ──────────────────────────────────────────────
        if len(df) < 10:
            logger.warning(
                "[%s] Modeling guard 1: only %d row(s) — running unsupervised analysis.",
                self.run_id[:8], len(df),
            )
            return self._stage_unsupervised(df)

        # ── Guard 2: no numeric features ───────────────────────────────────────
        feature_cols = [c for c in df.select_dtypes(include="number").columns if c != target_col]
        if not feature_cols:
            logger.warning(
                "[%s] Modeling guard 2: no numeric features — running unsupervised analysis.",
                self.run_id[:8],
            )
            return self._stage_unsupervised(df)

        # ── Guard 3: target column missing entirely ────────────────────────────
        if target_col not in df.columns:
            logger.warning(
                "[%s] Modeling guard 3: target_col '%s' not in df — running unsupervised analysis.",
                self.run_id[:8], target_col,
            )
            return self._stage_unsupervised(df)

        # ── Guard 4: single class in target (can't train a classifier) ─────────
        y = df[target_col]
        n_unique = int(y.dropna().nunique())
        if n_unique < 2:
            logger.warning(
                "[%s] Modeling guard 4: target '%s' has only %d unique value(s) — "
                "running unsupervised analysis.",
                self.run_id[:8], target_col, n_unique,
            )
            return self._stage_unsupervised(df)

        # ── Supervised modeling path ───────────────────────────────────────────
        try:
            # RL AutoML: select scaler/model/imputer triple
            rl_pipeline = None
            null_rate = 0.0
            try:
                from modeling.rl_automl import get_rl_automl
                rl_auto = get_rl_automl()
                null_rate = float(df.isnull().mean().mean())
                rl_pipeline = rl_auto.select_pipeline(
                    n_rows=len(df), n_cols=df.shape[1],
                    null_rate=null_rate, task="classification",
                )
                logger.info(
                    "[RL] AutoML selected: scaler=%s model=%s imputer=%s",
                    *rl_pipeline,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("RL AutoML unavailable: %s", exc)

            from modeling.trainer import ModelTrainer
            trainer = ModelTrainer(config=self.config)

            # MissingDataEngine + Triage should have handled all NaNs.
            # Defensive fallback: if NaNs remain, fill with column medians.
            remaining_nulls = int(df[feature_cols].isnull().sum().sum())
            if remaining_nulls > 0:
                logger.warning(
                    "[Model] %d NaN(s) remain after preprocessing — filling with column medians.",
                    remaining_nulls,
                )
                X = df[feature_cols].fillna(df[feature_cols].median())
            else:
                X = df[feature_cols]

            train_result = trainer.train(X, y, run_id=self.run_id)
            metrics = train_result.get("metrics", {}) if isinstance(train_result, dict) else {}

            # RL AutoML feedback loop
            if rl_pipeline is not None:
                try:
                    cv_score = float(metrics.get("roc_auc", metrics.get("accuracy", 0.0)))
                    rl_auto.record_outcome(
                        n_rows=len(df), n_cols=df.shape[1],
                        null_rate=null_rate, task="classification",
                        pipeline=rl_pipeline, cv_score=cv_score,
                        training_time_s=0.0,
                    )
                except Exception as exc:
                    logger.warning(
                        "RL outcome recording failed — non-fatal, continuing pipeline. Reason: %s",
                        exc,
                        exc_info=True,
                    )
            return metrics
        except ImportError:
            logger.warning("[%s] ModelTrainer not available — running unsupervised.", self.run_id[:8])
            return self._stage_unsupervised(df)
        except Exception as exc:
            logger.warning("[%s] Modeling failed (non-fatal): %s — returning error dict.", self.run_id[:8], exc)
            return {"error": str(exc)}

    def _stage_proposal(self, df: pd.DataFrame,
                        target_col: Optional[str]) -> Dict[str, Any]:
        """
        Stage 4 — Proposal Layer (Assistive Intelligence).

        Generates candidate hypotheses via AutoML, anomaly detection, feature
        ranking, contextual bandits, and RAG-based experience recall.
        Results are advisory only — pipeline continues regardless of outcome.
        """
        try:
            from proposal.proposal_engine import ProposalEngine
            engine = ProposalEngine(self.config)
            proposals = engine.generate_proposals(
                df, run_id=self.run_id, target_col=target_col
            )
            candidates = proposals.get("candidates", {})
            n_anomalies = (
                candidates.get("anomaly", {})
                          .get("anomaly_candidates", {})
                          .get("detected_outlier_count", 0)
            )
            top_feature = (
                candidates.get("ranker", {}).get("top_insight_candidate")
            )
            logger.info(
                "[%s] Proposal Layer: anomalies=%d top_feature=%s",
                self.run_id[:8], n_anomalies, top_feature,
            )
            return proposals
        except ImportError:
            raise RuntimeError("Proposal engine missing: ProposalEngine is strictly required.")

    def _stage_report(self, pipeline_result: "PipelineResult") -> str:
        """Stage 12 — Executive Report Generation.

        Passes the correct gate1/gate2 decisions and the full confidence
        vector (not just roc_auc) to the report generator.  Also attempts
        LLM-backed narrative via _generate_narrative().
        """
        try:
            from reporting_service.executive_report import ExecutiveReportGenerator
            reporter = ExecutiveReportGenerator(config=self.config)
            
            from reporting_service.llm_provider import get_llm_provider
            llm = get_llm_provider(self.config)
            
            confidence_score = pipeline_result.confidence_vector.get("confidence_score", 0.0) if pipeline_result.confidence_vector else 0.0
            
            verified_result = {
                "gate_decision": pipeline_result.gate_decision,
                "confidence_score": confidence_score,
                "confidence_vector": pipeline_result.confidence_vector,
                "metrics": pipeline_result.model_metrics
            }
            narrative = llm.generate_summary(verified_result, run_id=pipeline_result.run_id)

            risk_flags = []
            if hasattr(self, "_compliance_decision") and self._compliance_decision:
                for viol in getattr(self._compliance_decision, "violation_summary", []):
                    sev = viol.get("severity", "WARNING")
                    sev_level = "HIGH" if sev == "CRITICAL" else ("MEDIUM" if sev == "ERROR" else "LOW")
                    risk_flags.append({
                        # Structured fields — read directly by _build_regulatory in analytics.py
                        "rule_name":       viol.get("rule_name", "Unknown Rule"),
                        "severity":        sev,
                        "level":           sev_level,
                        "domain":          viol.get("domain", "banking"),
                        "column":          viol.get("column", "N/A"),
                        "offending_count": viol.get("offending_count", 0),
                        "message":         viol.get("message", ""),
                        "remediation":     viol.get("remediation", ""),
                        "category":        f"Regulatory ({viol.get('domain', 'banking').upper()})",
                        "type":            "REGULATORY_VIOLATION",
                    })


            pipeline_result.regulatory_report = risk_flags

            path = reporter.generate(
                run_id=pipeline_result.run_id,
                confidence_vector=pipeline_result.confidence_vector or {},
                gate1_decision=pipeline_result.gate1_decision,
                gate2_decision=pipeline_result.gate2_decision,
                model_metrics=pipeline_result.model_metrics or {},
                narrative=narrative or "",
                risk_flags=risk_flags,
                brain_report=getattr(pipeline_result, "brain_report", None),
            )
            return path
        except ImportError:
            logger.warning("ExecutiveReportGenerator not available")
            return ""
        except Exception as exc:
            logger.warning("Reporting failed (non-fatal): %s", exc)
            return ""

    def _audit(self, result: PipelineResult, snapshot: ISSFSnapshot) -> None:
        try:
            import json
            os.makedirs("audit", exist_ok=True)
            
            conf_score = 0.0
            if getattr(result, "confidence_vector", None) and isinstance(result.confidence_vector, dict):
                conf_score = result.confidence_vector.get("confidence_score", 0.0)
                
            entry = {
                "event": "PIPELINE_RUN",
                "run_id": result.run_id,
                "dataset_id": result.dataset_id,
                "snapshot_id": result.snapshot_id,
                "source_type": getattr(snapshot, "source_type", "unknown"),
                "data_mode": getattr(snapshot, "data_mode", "unknown"),
                "schema_version": getattr(snapshot, "schema_version", "unknown"),
                "row_count": getattr(snapshot, "row_count", 0),
                "quality_score": getattr(snapshot, "quality_score", 0.0),
                "gate_decision": result.gate_decision,
                "gate1_decision": getattr(result, "gate1_decision", "UNKNOWN"),
                "gate2_decision": getattr(result, "gate2_decision", "UNKNOWN"),
                "confidence_score": conf_score,
                "confidence_vector": getattr(result, "confidence_vector", {}),
                "stages": [s.to_dict() for s in getattr(result, "stages", [])],
                "timestamp": getattr(result, "completed_at", ""),
            }
            with open("audit/audit.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:  # noqa: BLE001
            pass

        # AdaptiveLearner code has been removed per architecture simplification

        # ── RL Orchestrator & RL Updater: close the feedback loop ──────────
        try:
            from modeling.rl_orchestrator import get_rl_orchestrator
            if self._rl_plan is not None:
                actual_min = (time.perf_counter() - self._run_start) / 60.0
                quality = (
                    result.model_metrics.get("roc_auc", 0.5)
                    if result.model_metrics else 0.5
                )
                sla = self.config.get("orchestrator", {}).get("sla_minutes", 30.0)
                get_rl_orchestrator().record_outcome(
                    self._rl_plan, actual_min, quality, sla
                )
                logger.info(
                    "[RL] Orchestrator feedback: actual=%.1fm quality=%.3f sla=%.0fm",
                    actual_min, quality, sla,
                )
        except Exception:  # noqa: BLE001
            pass

        # RLUpdater has been removed per architecture simplification

    # ── New Stage Implementations ──────────────────────────────────────────────

    def _stage_streaming_window(self, df: pd.DataFrame) -> list:
        """
        Stage 0 — Streaming Window Engine (Layer 2: Data Processing Layer).

        Activated for Kafka / stream source types only.
        Partitions the incoming DataFrame into tumbling / sliding / session windows
        using the StreamingWindowEngine. Returns a list of WindowBatch objects.
        The pipeline uses the first batch for downstream stages; full batch
        iteration is available for multi-window processing use cases.
        """
        from ingestion.streaming_window import StreamingWindowEngine
        engine = StreamingWindowEngine.from_config(self.config)
        batches = engine.process(df)
        logger.info(
            "[%s] StreamingWindowEngine: %d rows → %d window batches (strategy=%s)",
            self.run_id[:8], len(df), len(batches),
            type(engine.strategy).__name__,
        )
        return batches

    def _stage_analytics(
        self,
        df: pd.DataFrame,
        target_col: Optional[str],
    ) -> Dict[str, Any]:
        """
        Stage 4 — AI & Analytics Service Layer.

        Sequences the full analytics stack via AnalyticsOrchestrator:
          A. Automated EDA      (distributions, correlations, outliers, insights)
          B. Feature Engineering (lag, interactions, freq-encoding, binning + RL prune)
          C. Insight Ranking    (proposal/insight_ranker.py)
          D. LLM Summarization  (reporting_service/llm_provider.py)

        The enriched DataFrame is promoted back to the main pipeline via
        the `_enriched_df` key so downstream stages benefit from new features.
        Falls back to the legacy ProposalEngine if AnalyticsOrchestrator fails.
        """
        try:
            from analytics.orchestrator import AnalyticsOrchestrator
            orch = AnalyticsOrchestrator(config=self.config)
            analytics_result = orch.run(
                df=df,
                target_col=target_col,
                run_id=self.run_id,
            )
            # Package result as a plain dict (PipelineResult.analytics_result)
            out = analytics_result.to_dict(include_df=False)
            # Carry the enriched df under a private key for the bridge to consume
            if analytics_result.enriched_df is not None:
                out["_enriched_df"] = analytics_result.enriched_df
            logger.info(
                "[%s] Analytics: %d insights, llm=%d chars",
                self.run_id[:8],
                len(analytics_result.insights),
                len(analytics_result.llm_summary),
            )
            return out
        except ImportError:
            pass  # AnalyticsOrchestrator not yet installed — fall back

        # Fallback: legacy ProposalEngine
        try:
            from proposal.proposal_engine import ProposalEngine
            engine = ProposalEngine(self.config)
            proposals = engine.generate_proposals(
                df, run_id=self.run_id, target_col=target_col
            )
            return proposals if isinstance(proposals, dict) else {}
        except ImportError:
            raise RuntimeError(
                "Analytics stage failed: neither AnalyticsOrchestrator "
                "nor ProposalEngine is available."
            )

    def _stage_verify(self, df: pd.DataFrame, model_metrics: Dict,
                      run_id: str) -> bool:
        """
        Hard Gate 2 — Independent Verifier Stack.

        Runs all statistical verifiers independently of stage 2 (Hard Gate 1).
        If any verifier returns a hard REJECT, this stage fails.
        Gate 2 failure triggers the Retry Engine, NOT a full pipeline abort.
        """
        try:
            from verifier.confidence_vector import ConfidenceVector
            cv = ConfidenceVector.from_config(self.config)
            gate_result = cv.run_verification_gate(
                df=df,
                model_metrics=model_metrics,
                run_id=run_id,
            )
            if gate_result.get("decision") == "REJECT":
                logger.warning(
                    "[%s] Hard Gate 2 (statistical verifier) REJECTED: %s",
                    run_id[:8], gate_result.get("reason", "unknown"),
                )
                return False
            return True
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Verifier engine error: {exc}")

    def _stage_confidence_vector(self, df: pd.DataFrame, model_metrics: Dict,
                                  snapshot: Any, gate2_ok: Any) -> Dict[str, Any]:
        """
        Confidence Vector Aggregation — Step 6 of DIPEX architecture.

        Produces weighted scalar confidence score ∈ [0,1] from:
          data_quality, statistical_strength, stability, drift_robustness,
          compliance_penalty (new), retry_penalty.
        Domain thresholds: banking=0.85, healthcare=0.90, default=0.70.
        """
        # Retrieve compliance penalty from ComplianceAdvisor (set in _stage_validate)
        comp_decision = getattr(self, "_compliance_decision", None)
        compliance_penalty: float = 0.0
        compliance_decision_str: str = "allowed"
        compliance_decision_dict = None
        if comp_decision is not None:
            compliance_penalty = float(getattr(comp_decision, "compliance_penalty", 0.0))
            compliance_decision_str = str(getattr(comp_decision, "decision", "allowed"))
            try:
                compliance_decision_dict = comp_decision.to_dict()
            except Exception:  # noqa: BLE001
                compliance_decision_dict = None

        try:
            from verifier.confidence_vector import ConfidenceVector
            cv = ConfidenceVector.from_config(self.config)
            vector = cv.aggregate(
                df=df,
                model_metrics=model_metrics,
                quality_score=float(snapshot.quality_score or 0.5),
                gate2_passed=(gate2_ok is not False),
                retry_count=getattr(self, "_current_retry_count", 0),
                compliance_penalty=compliance_penalty,
                compliance_decision=compliance_decision_str,
            )
            # Embed full compliance decision in the confidence vector for reporting
            if compliance_decision_dict:
                vector["compliance"] = compliance_decision_dict
            logger.info(
                "[%s] Confidence Vector: score=%.3f (compliance_penalty=%.3f decision=%s)",
                self.run_id[:8], vector.get("confidence_score", 0.0),
                compliance_penalty, compliance_decision_str,
            )
            return vector
        except ImportError:
            logger.warning("ConfidenceVector engine missing — skipping.")
            return {"confidence_score": 0.5}
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"ConfidenceVector aggregation failed: {exc}")
    def _retry_engine_loop(
        self,
        result: "PipelineResult",
        snapshot: Any,
        df: pd.DataFrame,
        target_col: Optional[str],
        skip: set,
        confidence_score: float,
        conf_thresh: float,
    ) -> bool:
        """
        Intelligent Retry Engine — Step 7 of DIPEX architecture.

        Bounded retries (max N from config). Each attempt:
        1. Selects an alternate bandit strategy (never the same path twice).
        2. Restarts from the appropriate stage (EDA / Proposal / Full).
        3. Logs every attempt: strategy, confidence delta, outcome.
        4. Escalates if retry budget exhausted.
        Returns True if confidence eventually crosses threshold, else False.
        """
        max_retries = int(
            self.config.get("pipeline", {}).get("retry", {}).get("max_retries", 3)
        )
        tried_strategies: set = set()
        logger.warning(
            "[%s] Retry Engine activated: conf=%.3f < threshold=%.3f max=%d",
            self.run_id[:8], confidence_score, conf_thresh, max_retries,
        )

        for attempt in range(1, max_retries + 1):
            result.retry_count = attempt
            self._attempt = attempt                  # track for _stage_record_experience
            self._current_retry_count = attempt      # track for _stage_confidence_vector

            # Select alternate bandit strategy
            conf_before = confidence_score
            strategy = self._select_retry_strategy(tried_strategies)
            tried_strategies.add(strategy)
            logger.info(
                "[%s] Retry %d/%d — strategy=%s tried=%s",
                self.run_id[:8], attempt, max_retries, strategy, tried_strategies,
            )

            # Re-run stats + verification with alternate strategy
            self._run_stage(result, f"retry_{attempt}_stats", skip,
                            self._stage_stats, df, target_col)

            gate2_ok = self._run_stage(result, f"retry_{attempt}_verify", skip,
                                        self._stage_verify, df,
                                        result.model_metrics or {}, self.run_id)
            result.gate2_decision = "PASS" if gate2_ok is not False else "REJECT"

            new_cv = self._run_stage(result, f"retry_{attempt}_confidence", skip,
                                      self._stage_confidence_vector,
                                      df, result.model_metrics or {}, snapshot, gate2_ok)
            if isinstance(new_cv, dict):
                result.confidence_vector = new_cv
                new_conf = float(new_cv.get("confidence_score", 0.0))
                delta = new_conf - confidence_score
                confidence_score = new_conf
                logger.info(
                    "[%s] Retry %d result: conf=%.3f delta=%+.3f threshold=%.3f",
                    self.run_id[:8], attempt, new_conf, delta, conf_thresh,
                )

                # ── Bandit Q-update: persist outcome for this strategy ─────
                try:
                    from ingestion.pipeline_bridge_helpers import record_retry_outcome
                    record_retry_outcome(
                        self.config, strategy, conf_before, new_conf
                    )
                except Exception:  # noqa: BLE001
                    pass

                if new_conf >= conf_thresh:
                    logger.info("[%s] Retry Engine SUCCESS at attempt %d",
                                self.run_id[:8], attempt)
                    return True

        # Budget exhausted
        logger.error(
            "[%s] Retry Engine EXHAUSTED after %d attempts (final conf=%.3f < %.3f). "
            "Escalating to monitoring.",
            self.run_id[:8], max_retries, confidence_score, conf_thresh,
        )
        try:
            import json, os
            os.makedirs("audit", exist_ok=True)
            with open("audit/retry_escalations.jsonl", "a") as f:
                f.write(json.dumps({
                    "run_id": self.run_id,
                    "final_confidence": confidence_score,
                    "threshold": conf_thresh,
                    "retries": max_retries,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }) + "\n")
        except Exception:  # noqa: BLE001
            pass
        return False

    def _select_retry_strategy(self, tried: set) -> str:
        """Select next bandit strategy. Never repeat a tried strategy."""
        try:
            from ingestion.pipeline_bridge_helpers import load_bandit_state
            q_table = load_bandit_state(self.config)
        except Exception:  # noqa: BLE001
            q_table = {}
        candidates = [
            "resample_oversample", "feature_prune_aggressive",
            "impute_knn", "outlier_winsorize", "scale_robust",
        ]
        available = [c for c in candidates if c not in tried]
        if not available:
            # All tried; pick least-regret from candidates
            return candidates[0]
        # Sort by Q-value descending (higher = preferred)
        available.sort(key=lambda a: q_table.get(a, 0.5), reverse=True)
        return available[0]

    def _stage_record_experience(self, snapshot: Any, conf_vector: Dict,
                                  model_metrics: Dict) -> bool:
        """
        Step 9 — Experience Memory: store approved run details for future RAG recall.

        Writes to THREE stores in priority order:
          1. ExperienceMemoryV2  — HMAC-signed, append-only JSONL (system-of-record)
          2. ExperienceMemory v1 — ChromaDB full-text store (simpler retrieval)
          3. ExperienceRecall    — RAGRetriever / sentence-transformer semantic store
                                   (feeds the Proposal Layer's RAG on next run)

        Only called for APPROVED runs (Gate 1 PASS + Gate 2 PASS + conf >= threshold).
        """
        try:
            conf_score = float(conf_vector.get("confidence_score", 0.0))
            winning_strategy = {
                "top_insight_candidate": None,
                "strategy_family": "balanced",
                "model_metrics": model_metrics,
            }
            narrative = (
                f"Run {self.run_id}: confidence={conf_score:.3f}, "
                f"dataset={snapshot.dataset_id}, rows={snapshot.row_count}"
            )

            # ── Store 1: ExperienceMemoryV2 — HMAC-signed JSONL ───────────
            from learning.experience_memory_v2 import ExperienceMemoryV2
            mem_v2 = ExperienceMemoryV2.from_config(self.config)
            mem_v2.record_approved_output(
                run_id=self.run_id,
                fingerprint=snapshot.fingerprint,
                approved_output={
                    "dataset_id": snapshot.dataset_id,
                    "snapshot_id": snapshot.snapshot_id,
                    "confidence_vector": conf_vector,
                    "schema_version": snapshot.schema_version,
                    "source_type": snapshot.source_type,
                },
                winning_strategy=winning_strategy,
                confidence_score=conf_score,
                attempt=self._attempt,
                narrative=narrative,
            )

            # ── Store 2: ExperienceMemory v1 — ChromaDB full-text ─────────
            try:
                from learning.experience_memory import ExperienceMemory
                mem_v1 = ExperienceMemory()
                mem_v1.store(
                    run_id=self.run_id,
                    summary=narrative,
                    metadata={
                        "dataset_id": snapshot.dataset_id,
                        "confidence_score": conf_score,
                        "source_type": snapshot.source_type,
                        "row_count": snapshot.row_count,
                        "schema_version": snapshot.schema_version,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("ExperienceMemory v1 mirror failed (non-fatal): %s", exc)

            # ── Store 3: ExperienceRecall RAG — closes the Proposal feedback loop ──
            # Writes the approved run into the RAGRetriever (ChromaDB + sentence-
            # transformer) so that the NEXT pipeline run's Proposal Stage can recall
            # this approved experience via semantic similarity search.
            try:
                from proposal.rag.experience_recall import ExperienceRecall
                recall_store = ExperienceRecall(self.config)
                recall_store.store_experience(
                    run_id=self.run_id,
                    df=snapshot.data,
                    metadata={
                        "dataset_id": snapshot.dataset_id,
                        "confidence_score": conf_score,
                        "source_type": snapshot.source_type,
                        "row_count": int(snapshot.row_count),
                        "schema_version": snapshot.schema_version,
                        "gate1": "PASS",
                        "gate2": "PASS",
                        "attempt": int(self._attempt),
                    },
                )
                logger.info(
                    "[%s] RAG feedback loop: approved run written to ExperienceRecall store",
                    self.run_id[:8],
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "ExperienceRecall store failed (non-fatal — RAG recall unaffected): %s", exc
                )

            logger.info(
                "[%s] Experience Memory: approved run recorded across all 3 stores (conf=%.3f)",
                self.run_id[:8], conf_score,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Experience memory record failed (non-fatal): %s", exc)
            return False

    def _stage_rl_update(self, snapshot_id: str,
                          drift_psi: Optional[float], quarantine_frac: float = 0.0, data_health_score: float = 50.0) -> bool:
        """
        Stage 11 — RL Update via ReinforcementUpdateEngine (Meta-RL, regret, EWC).

        Always non-blocking. Returns False on any failure instead of raising.
        Previously raised RuntimeError which caused a hard stage FAIL — fixed.
        """
        try:
            from learning.rl_agent.agent import PPOAgent
            # Reconstruct context since recommend must be called before record_outcome
            engine = PPOAgent.from_config(self.config)
            
            # The agent needs context to recommend an action 
            dummy_context = {
                "n_rows": getattr(self, "_stage_rows", 1000), 
                "n_cols": 20,
                "null_rate": quarantine_frac, 
                "anomaly_rate": 0.0,
                "drift_psi": drift_psi or 0.0, 
                "data_health": data_health_score, 
                "domain": "generic",
            }
            engine.recommend(dummy_context, greedy=True)
            
            # Now we can record outcome
            result_summary = getattr(self, "_last_result_summary", {})
            analytics = {"data_health_score": data_health_score, "drift_psi": drift_psi}
            
            summary = engine.record_outcome(
                result_summary=result_summary,
                analytics=analytics,
                user_approved_plan=True
            )
            logger.info(
                "[%s] PPO RL Update: episode=%s reward=%s shadow_mode=%s",
                self.run_id[:8],
                summary.get("episode_count"),
                summary.get("reward"),
                summary.get("in_shadow_mode")
            )
            return True
        except ImportError:
            logger.info("[%s] RL Update engine not installed — skipping (non-fatal).", self.run_id[:8])
            return False
        except Exception as exc:  # noqa: BLE001
            # Previously: raise RuntimeError(f"RL Update failed: {exc}")
            # Now: non-blocking warning. RL failure never halts the pipeline.
            logger.warning("[%s] RL Update failed (non-fatal): %s", self.run_id[:8], exc)
            return False

    # ── Robustness Stage Implementations ──────────────────────────────────────

    def _stage_triage(
        self, df: pd.DataFrame, target_col: Optional[str]
    ) -> pd.DataFrame:
        """
        Stage 0.5 — Robust Data Triage.

        Profiles each column and applies adaptive remediation:
          • Drop near-all-null columns
          • Coerce mixed-type columns to numeric
          • Drop zero-variance (constant) columns
          • Hash-encode high-cardinality categoricals
          • Auto log1p skewed positive numeric columns
          • Label-encode remaining object columns
          • Detect class imbalance (advisory only)
        """
        from preprocessing.robust_triage import RobustTriage
        triager = RobustTriage.from_config(self.config)
        clean_df, triage_report = triager.triage(
            df, target_col=target_col, run_id=self.run_id
        )
        logger.info(
            "[%s] Triage: dropped=%d filled=%d zero_fixed=%d coerced=%d "
            "hash_enc=%d label_enc=%d resampled=%s",
            self.run_id[:8],
            len(triage_report.columns_dropped),
            len(triage_report.columns_filled),
            len(triage_report.zero_fixed_columns),
            len(triage_report.columns_coerced),
            len(triage_report.columns_hash_encoded),
            len(triage_report.columns_label_encoded),
            triage_report.resample_info.get("method", "none") if triage_report.resample_info else "none",
        )
        # Store triage metadata as DataFrame attributes for downstream use
        clean_df.attrs["triage_imbalance"] = triage_report.imbalance_info
        clean_df.attrs["triage_report"] = triage_report.to_dict()
        return clean_df

    def _stage_drift(self, df: pd.DataFrame, dataset_id: str) -> dict:
        """
        Stage 1.5 — Schema & Distribution Drift Detection.

        Compares the current DataFrame against the stored schema fingerprint
        for dataset_id. Emits warnings for column additions/removals/dtype
        changes and PSI-based distribution drift.

        Returns the DriftReport dict for the audit trail.
        """
        from validation.drift_detector import SchemaDriftDetector
        detector = SchemaDriftDetector.from_config(self.config)
        drift_report = detector.detect(df, dataset_id=dataset_id, run_id=self.run_id)
        n_err = sum(1 for v in drift_report.violations if v.severity == "ERROR")
        n_warn = sum(1 for v in drift_report.violations if v.severity == "WARNING")
        if drift_report.is_first_run:
            logger.info(
                "[%s] Drift detection: first run — schema fingerprint written.",
                self.run_id[:8],
            )
        else:
            logger.info(
                "[%s] Drift detection: %d ERROR(s), %d WARNING(s).",
                self.run_id[:8], n_err, n_warn,
            )
        return drift_report.to_dict()

    def _stage_leakage(
        self, df: pd.DataFrame, target_col: str
    ) -> pd.DataFrame:
        """
        Stage 5.5 — Data Leakage Detection.

        Scans all features for:
          • Near-perfect correlation with target (numeric)
          • Near-perfect categorical alignment (Cramér's V)
          • ID-like columns (near-unique values)

        CRITICAL leakage columns are removed from df.
        """
        from validation.leakage_detector import LeakageDetector
        detector = LeakageDetector.from_config(self.config)
        clean_df, leak_report = detector.detect(
            df, target_col=target_col, run_id=self.run_id
        )
        n_crit = sum(1 for v in leak_report.violations if v.severity == "CRITICAL")
        n_warn = sum(1 for v in leak_report.violations if v.severity == "WARNING")
        logger.info(
            "[%s] Leakage detection: %d CRITICAL (removed), %d WARNING(s).",
            self.run_id[:8], n_crit, n_warn,
        )
        return clean_df

    def _stage_calibrate(
        self,
        df: pd.DataFrame,
        target_col: str,
        model_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Stage 6.5 — Confidence Calibration.

        Applies Platt Scaling or Isotonic Regression to correct the model's
        over/under-confidence. Only applies calibration if ECE improves by
        the configured minimum threshold.

        Returns the CalibrationReport dict for embedding in model_metrics.
        """
        from qa_control.calibrator import ConfidenceCalibrator

        cal = ConfidenceCalibrator.from_config(self.config)
        feature_cols = [c for c in df.select_dtypes(include="number").columns
                        if c != target_col]
        if not feature_cols or target_col not in df.columns:
            return {}

        # Retrieve the trained model from model_metrics if available
        model = model_metrics.get("_fitted_model")
        if model is None:
            logger.debug("[Calibrator] No fitted model in model_metrics — skipping.")
            return {}

        X = df[feature_cols].fillna(0)
        y = df[target_col]

        _, cal_report = cal.calibrate(
            model=model, X_train=X, y_train=y, run_id=self.run_id
        )
        logger.info(
            "[%s] Calibration: applied=%s method=%s ECE %.4f→%.4f",
            self.run_id[:8],
            cal_report.applied,
            cal_report.method,
            cal_report.ece_before or 0,
            cal_report.ece_after or cal_report.ece_before or 0,
        )
        return cal_report.to_dict()

    def _stage_missing_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 0.75 — Missing Data Pattern Analysis (MCAR / MAR / MNAR).

        Runs before DataCleaner so imputation strategies are informed by WHY
        data is missing, not just that it is missing.

        MNAR columns get a `{col}_was_null` indicator feature added so the
        model can learn from the fact that a value was absent.
        """
        from preprocessing.missing_pattern_analyzer import MissingPatternAnalyzer
        analyzer = MissingPatternAnalyzer.from_config(self.config)
        enriched_df, mp_report = analyzer.analyze(df, run_id=self.run_id)
        logger.info(
            "[%s] Missing patterns: MNAR=%d MAR=%d MCAR=%d — %d indicator cols added.",
            self.run_id[:8],
            len(mp_report.mnar_columns),
            len(mp_report.mar_columns),
            len(mp_report.mcar_columns),
            sum(1 for p in mp_report.profiles if p.indicator_added),
        )
        # Persist pattern report in df attributes for downstream use
        enriched_df.attrs["missing_pattern_report"] = mp_report.to_dict()
        return enriched_df

    def _stage_multicollinearity(
        self, df: pd.DataFrame, target_col: Optional[str]
    ) -> pd.DataFrame:
        """
        Stage 5.7 — VIF-Based Multicollinearity Detection.

        Identifies numeric features with high VIF (collinear with other features).
        Drops the higher-VIF column from each collinear pair, keeping the one
        with more independent predictive power.
        """
        from validation.multicollinearity_detector import MulticollinearityDetector
        detector = MulticollinearityDetector.from_config(self.config)
        clean_df, mc_report = detector.detect(
            df, target_col=target_col, run_id=self.run_id
        )
        n_err  = sum(1 for v in mc_report.violations if v.severity == "ERROR")
        n_warn = sum(1 for v in mc_report.violations if v.severity == "WARNING")
        logger.info(
            "[%s] Multicollinearity: %d ERROR(s) (%d dropped), %d WARNING(s). "
            "%d collinear pairs.",
            self.run_id[:8],
            n_err, len(mc_report.columns_dropped),
            n_warn, len(mc_report.correlated_pairs),
        )
        return clean_df

    def _stage_feature_stability(
        self,
        model_metrics: Dict[str, Any],
        dataset_id: str,
    ) -> Dict[str, Any]:
        """
        Stage 6.7 — Feature Importance Stability Monitor.

        Compares SHAP/feature importance rankings from the current run against
        the stored baseline. Emits WARNING if top features have shifted,
        ERROR if they've changed completely.

        Catches silent upstream data pipeline bugs that pass all validation gates.
        """
        from monitoring.feature_stability_monitor import FeatureStabilityMonitor
        monitor = FeatureStabilityMonitor.from_config(self.config)

        # Extract feature importances from model_metrics
        importances: Dict[str, float] = {}
        # Try SHAP importances first (most accurate), fall back to model importances
        shap_imp = model_metrics.get("shap_importances") or \
                   model_metrics.get("feature_importances") or {}
        if isinstance(shap_imp, dict):
            importances = {k: float(v) for k, v in shap_imp.items() if v is not None}

        if not importances:
            logger.debug("[FeatureStability] No importances found in model_metrics — skipping.")
            return {}

        stability_report = monitor.check(
            feature_importances=importances,
            dataset_id=dataset_id,
            run_id=self.run_id,
        )
        logger.info(
            "[%s] Feature stability: status=%s tau=%s gained=%s lost=%s",
            self.run_id[:8],
            stability_report.status,
            f"{stability_report.tau:.3f}" if stability_report.tau is not None else "N/A",
            stability_report.features_gained,
            stability_report.features_lost,
        )
        return stability_report.to_dict()

    def _stage_rl_update(
        self,
        snapshot_id: str,
        drift_psi: Optional[float],
        quarantine_frac: float,
        data_health: float,
    ) -> Dict[str, Any]:
        """
        Stage 11 — PPO Agent Feedback Loop (closes the RL learning cycle).

        This is the CRITICAL closing step: without it the PPOAgent receives no
        reward signal from real pipeline runs and cannot learn.

        What happens here:
          1. Reconstruct the PipelineResult summary for the agent.
          2. Build analytics dict from stage outputs accumulated on self.
          3. Call agent.record_outcome() — this triggers PPO weight update
             once the buffer has >= 32 transitions.
          4. Log reward breakdown for monitoring.

        Non-fatal: any exception is caught and logged; pipeline result unaffected.
        """
        try:
            # Use the SAME agent instance from Step 0 (has _current_state set by recommend())
            agent = getattr(self, "_ppo_agent_instance", None)
            if agent is None:
                logger.debug("[%s] RL update: no agent instance from Step 0.", self.run_id[:8])
                return {"skipped": "no_agent_instance"}

            # Build result_summary from accumulated pipeline state
            result_summary = {
                "gate_decision":   getattr(self, "_gate_decision_cache", "WARN"),
                "model_metrics":   getattr(self, "_model_metrics_cache", {}),
                "quarantine_rows": int(quarantine_frac * 1000),  # normalised proxy
                "retry_count":     getattr(self, "_current_retry_count", 0),
            }

            analytics = {
                "data_health_score": data_health,
                "drift_psi":         drift_psi or 0.0,
            }

            # user_approved_plan: True if a pre-analysis plan was accepted this run
            user_approved = bool(
                self.config.get("pipeline", {}).get("plan_approved", False)
            )

            outcome = agent.record_outcome(
                result_summary=result_summary,
                analytics=analytics,
                user_approved_plan=user_approved,
            )

            logger.info(
                "[%s] RL update: episode=%d reward=%.4f shadow=%s training_metrics=%s",
                self.run_id[:8],
                outcome.get("episode_count", 0),
                outcome.get("reward", 0.0),
                outcome.get("in_shadow_mode", True),
                {k: round(v, 4) for k, v in
                 outcome.get("training_metrics", {}).items()
                 if isinstance(v, float)},
            )
            return outcome

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[%s] RL update stage failed (non-fatal): %s",
                self.run_id[:8], exc,
            )
            return {"error": str(exc)}
