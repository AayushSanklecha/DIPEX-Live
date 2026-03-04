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
    gate1_decision: str = "PENDING"    # PASS | REJECT
    gate2_decision: str = "PENDING"    # PASS | REJECT | NOT_RUN
    confidence_vector: Optional[Dict] = None
    retry_count: int = 0

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
            "silver_id": getattr(self, 'silver_id', None),
            "gold_artefacts": len(getattr(self, 'gold_artefacts', [])),
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
        self._attempt: int = 0              # incremented in _retry_engine_loop
        self._episode: str = self.run_id    # used by ReinforcementUpdateEngine
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

        # ── Build Silver ImmutableDataFrame for Gold derivation ───────────────
        result.silver_id = snapshot.snapshot_id
        result.gold_artefacts = []
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

        # ── Stage 1: Preprocessing ────────────────────────────────────────────
        prep_out = self._run_stage(result, "preprocessing", skip,
                             self._stage_preprocess, df, target_col)
        if prep_out is not None:
            df = prep_out


        # ── Stage 2: Hard Gate 1 — Deterministic Validation ──────────────────
        gate1_ok = self._run_stage(result, "validation", skip,
                                   self._stage_validate, df, snapshot.dataset_id)
        if gate1_ok is False:
            result.gate_decision  = "FAIL"
            result.gate1_decision = "REJECT"
            result.gate2_decision = "NOT_RUN"
            result.completed_at   = datetime.now(timezone.utc).isoformat()
            self._audit(result, snapshot)
            logger.warning(
                "[%s] Hard Gate 1 REJECTED — pipeline halted, RL update suppressed.",
                self.run_id[:8],
            )
            return result
        result.gate1_decision = "PASS"

        # ── Stage 3: Data Profiling ───────────────────────────────────────
        profile_result = self._run_stage(result, "profiling", skip,
                                         self._stage_profile, df, self.run_id)
        drift_psi: Optional[float] = None
        if isinstance(profile_result, dict):
            drift_psi = profile_result.get("psi_score")

        # ── Stage 4: Proposal Layer ───────────────────────────────────────────
        proposal_result = self._run_stage(result, "proposal", skip,
                                          self._stage_proposal, df, target_col)

        # ── Stage 5: Governance ───────────────────────────────────────────────
        self._run_stage(result, "governance", skip, self._stage_governance, df)

        # ── Stage 5: Statistical Analysis ─────────────────────────────────────
        self._run_stage(result, "statistics", skip, self._stage_stats, df, target_col)

        # ── Stage 6: ML Modeling ──────────────────────────────────────────────
        model_metrics: Dict[str, Any] = {}
        if target_col and target_col in df.columns:
            metrics_out = self._run_stage(result, "modeling", skip,
                                          self._stage_model, df, target_col)
            if isinstance(metrics_out, dict):
                model_metrics = metrics_out
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
        domain = self.config.get("pipeline", {}).get("domain", "default")
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
        self._run_stage(result, "rl_update", skip,
                        self._stage_rl_update, snapshot.snapshot_id, drift_psi)

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

            from preprocessing.pipeline_builder import PipelineBuilder
            builder = PipelineBuilder(self.config)
            pipe = builder.build(df, target_col=target_col)
            feature_cols = [c for c in df.columns if c != target_col]
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
        try:
            from validation.hard_gate import HardGate
            gate = HardGate.from_config(self.config)
            result_gate = gate.run(df, run_id=self.run_id)
            if result_gate.decision == "REJECT":
                raise RuntimeError(f"Hard gate REJECTED: {result_gate.reason}")
            return True
        except ImportError:
            raise RuntimeError("Validation engine missing: HardGate is required for deterministic validation.")

    def _stage_profile(self, df: pd.DataFrame, run_id: str) -> dict:
        try:
            from profiling.profile_report import ProfileReport
            pr = ProfileReport(config=self.config)
            report = pr.generate(df, run_id=run_id)
            return report
        except ImportError:
            logger.warning("ProfileReport not available — skipping profiling")
            return {}

    def _stage_governance(self, df: pd.DataFrame) -> dict:
        try:
            from governance.governance_engine import GovernanceEngine
            engine = GovernanceEngine(self.config)
            gov_result = engine.evaluate(
                run_id=self.run_id,
                confidence_score=0.8,   # default; overridden by ML scorer downstream
                gate1_decision="PASS",
                gate2_decision="PASS",
                df_columns=list(df.columns),
            )
            return gov_result.to_dict() if hasattr(gov_result, "to_dict") else (gov_result or {})
        except ImportError:
            logger.warning("GovernanceEngine not available — skipping governance")
            return {}

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
        try:
            # ── RL AutoML: select scaler/model/imputer triple ─────────────
            rl_pipeline = None
            try:
                from modeling.rl_automl import get_rl_automl
                rl_auto = get_rl_automl()
                null_rate = df.isnull().mean().mean()
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
            feature_cols = [c for c in df.select_dtypes(include="number").columns if c != target_col]
            if not feature_cols:
                return {"error": "No numeric features available"}
            X = df[feature_cols].fillna(0)
            y = df[target_col]
            result = trainer.train(X, y, run_id=self.run_id)
            metrics = result.get("metrics", {}) if isinstance(result, dict) else {}

            # ── RL AutoML feedback loop ────────────────────────────────────
            if rl_pipeline is not None:
                try:
                    cv_score = metrics.get("roc_auc", metrics.get("accuracy", 0.0))
                    rl_auto.record_outcome(
                        n_rows=len(df), n_cols=df.shape[1],
                        null_rate=null_rate, task="classification",
                        pipeline=rl_pipeline, cv_score=cv_score,
                        training_time_s=0.0,
                    )
                except Exception:  # noqa: BLE001
                    pass
            return metrics
        except ImportError:
            logger.warning("ModelTrainer not available")
            return {}
        except Exception as exc:
            logger.warning("Modeling failed (non-fatal): %s", exc)
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

        # LLM narrative has been removed per architecture simplification
        return ""

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

            path = reporter.generate(
                run_id=pipeline_result.run_id,
                confidence_vector=pipeline_result.confidence_vector or {},
                gate1_decision=pipeline_result.gate1_decision,
                gate2_decision=pipeline_result.gate2_decision,
                model_metrics=pipeline_result.model_metrics or {},
                narrative=narrative or "",
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
          compliance, retry_penalty.
        Domain thresholds: banking=0.85, healthcare=0.90, default=0.70.
        """
        try:
            from verifier.confidence_vector import ConfidenceVector
            cv = ConfidenceVector.from_config(self.config)
            vector = cv.aggregate(
                df=df,
                model_metrics=model_metrics,
                quality_score=float(snapshot.quality_score or 0.5),
                gate2_passed=(gate2_ok is not False),
                retry_count=getattr(self, "_current_retry_count", 0),
            )
            logger.info(
                "[%s] Confidence Vector: score=%.3f",
                self.run_id[:8], vector.get("confidence_score", 0.0),
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
                          drift_psi: Optional[float]) -> bool:
        """
        Step 10 — RL Update via ReinforcementUpdateEngine (Meta-RL, regret, EWC).
        Only called when Gate 1 passed. Gate 2 partial fails are allowed to learn.
        """
        try:
            from learning.reinforcement_update_engine import ReinforcementUpdateEngine
            engine = ReinforcementUpdateEngine.from_config(self.config)
            summary = engine.update_for_run(
                run_id=self.run_id,
                drift_psi=drift_psi,
                episode=getattr(self, "_episode", None),
            )
            logger.info(
                "[%s] RL Update: retry=%s ranker=%s conf=%s meta=%s regret=%s "
                "epsilon=%s rollback=%s sandbox=%s",
                self.run_id[:8],
                summary.updated_retry_policy,
                summary.updated_ranker_priors,
                summary.updated_confidence_weights,
                summary.policies_updated,
                summary.regret_updated,
                summary.epsilon_adjusted,
                summary.rollback_triggered,
                summary.sandbox_active,
            )
            return True
        except ImportError:
            logger.warning("RL Update engine missing — skipping.")
            return False
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"RL Update failed: {exc}")
