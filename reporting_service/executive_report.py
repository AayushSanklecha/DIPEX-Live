"""
reporting_service/executive_report.py
---------------------------------------
Auto-generates HTML executive summary reports from pipeline run data.

Uses Jinja2 to render a print-ready HTML document with:
  - KPI scorecard
  - Data quality section
  - Model performance table
  - Risk flag summary
  - Narrative insights
  - Confidence breakdown
  - Audit metadata

Output: reports/{run_id}_executive_report.html
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("dipex.reporting.executive_report")


REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DIPEX Executive Report — {{ run_id }}</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', Arial, sans-serif; background: #0d1117; color: #e6edf3; padding: 0; }
    .cover { background: linear-gradient(135deg, #1a1f35 0%, #0d1117 100%); padding: 60px 80px; border-bottom: 2px solid #21d4fd; }
    .cover h1 { font-size: 2.6rem; color: #fff; font-weight: 700; letter-spacing: -0.5px; }
    .cover h1 span { color: #21d4fd; }
    .cover .subtitle { color: #8b949e; font-size: 1rem; margin-top: 8px; }
    .cover .meta { display: flex; gap: 40px; margin-top: 32px; flex-wrap: wrap; }
    .cover .meta-item { background: rgba(255,255,255,0.05); padding: 12px 20px; border-radius: 8px; border: 1px solid #21262d; }
    .cover .meta-item strong { display: block; color: #21d4fd; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; }
    .cover .meta-item span { font-size: 1rem; color: #e6edf3; }
    .container { max-width: 1100px; margin: 0 auto; padding: 40px 80px; }
    section { margin-bottom: 50px; }
    h2 { font-size: 1.4rem; color: #21d4fd; border-bottom: 1px solid #21262d; padding-bottom: 10px; margin-bottom: 24px; text-transform: uppercase; letter-spacing: 1px; }
    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }
    .kpi-card { background: #161b22; border: 1px solid #21262d; border-radius: 10px; padding: 20px; }
    .kpi-card .label { color: #8b949e; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; }
    .kpi-card .value { font-size: 2rem; font-weight: 700; color: {% if badge == 'QA_APPROVED' %}#3fb950{% elif badge == 'QA_CONDITIONAL' %}#d29922{% else %}#f85149{% endif %}; margin-top: 6px; }
    .kpi-card .sub { color: #8b949e; font-size: 0.8rem; margin-top: 4px; }
    table { width: 100%; border-collapse: collapse; background: #161b22; border-radius: 10px; overflow: hidden; }
    th { background: #1f2937; color: #8b949e; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; padding: 12px 16px; text-align: left; }
    td { padding: 12px 16px; border-bottom: 1px solid #21262d; color: #e6edf3; font-size: 0.9rem; }
    tr:last-child td { border-bottom: none; }
    .badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
    .badge.pass { background: rgba(63,185,80,0.15); color: #3fb950; border: 1px solid #3fb950; }
    .badge.warn { background: rgba(210,153,34,0.15); color: #d29922; border: 1px solid #d29922; }
    .badge.fail { background: rgba(248,81,73,0.15); color: #f85149; border: 1px solid #f85149; }
    .narrative-box { background: #161b22; border-left: 4px solid #21d4fd; border-radius: 0 8px 8px 0; padding: 24px; font-size: 0.95rem; line-height: 1.7; color: #e6edf3; white-space: pre-wrap; }
    .risk-item { display: flex; align-items: flex-start; gap: 12px; background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 16px; margin-bottom: 12px; }
    .risk-level { padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; white-space: nowrap; }
    .risk-HIGH { background: rgba(248,81,73,0.15); color: #f85149; }
    .risk-MEDIUM { background: rgba(210,153,34,0.15); color: #d29922; }
    .risk-LOW { background: rgba(63,185,80,0.15); color: #3fb950; }
    .conf-bar { background: #21262d; border-radius: 4px; height: 8px; width: 100%; margin-top: 4px; }
    .conf-fill { height: 8px; border-radius: 4px; background: linear-gradient(90deg, #21d4fd, #3fb950); }
    .footer { text-align: center; padding: 32px; color: #8b949e; font-size: 0.8rem; border-top: 1px solid #21262d; margin-top: 60px; }
    .conf-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
    .conf-label { color: #8b949e; font-size: 0.85rem; width: 160px; flex-shrink: 0; }
    .conf-score { color: #e6edf3; font-size: 0.85rem; width: 60px; text-align: right; }
    @media print {
      body { background: #fff; color: #000; }
      .cover { background: #1a1f35; -webkit-print-color-adjust: exact; }
    }
  </style>
</head>
<body>

<div class="cover">
  <h1>DIPEX <span>Executive Report</span></h1>
  <div class="subtitle">Cognitive Analytics Automation System — Verified Output</div>
  <div class="meta">
    <div class="meta-item"><strong>Run ID</strong><span>{{ run_id }}</span></div>
    <div class="meta-item"><strong>Generated</strong><span>{{ generated_at }}</span></div>
    <div class="meta-item"><strong>QA Status</strong><span>{{ qa_status }}</span></div>
    <div class="meta-item"><strong>Domain</strong><span>{{ domain | upper }}</span></div>
  </div>
</div>

<div class="container">

  {% for section in section_order %}
  {% if section == "kpi" %}
  <!-- KPI SCORECARD -->
  <section>
    <h2>Key Performance Indicators</h2>
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="label">Confidence Score</div>
        <div class="value" style="color: {{ '#3fb950' if confidence_score >= 0.8 else ('#d29922' if confidence_score >= 0.6 else '#f85149') }}">{{ '%.1f' | format(confidence_score * 100) }}%</div>
        <div class="sub">Aggregated gate confidence</div>
      </div>
      <div class="kpi-card">
        <div class="label">Gate 1 (Data Quality)</div>
        <div class="value" style="color: {{ '#3fb950' if gate1 == 'PASS' else '#f85149' }}">{{ gate1 }}</div>
        <div class="sub">Deterministic validation</div>
      </div>
      <div class="kpi-card">
        <div class="label">Gate 2 (Statistical)</div>
        <div class="value" style="color: {{ '#3fb950' if gate2 == 'PASS' else '#f85149' }}">{{ gate2 }}</div>
        <div class="sub">Model verification gate</div>
      </div>
      <div class="kpi-card">
        <div class="label">Dataset Rows</div>
        <div class="value">{{ row_count }}</div>
        <div class="sub">{{ col_count }} columns</div>
      </div>
      <div class="kpi-card">
        <div class="label">Quality Flags</div>
        <div class="value" style="color: {{ '#d29922' if flag_count > 0 else '#3fb950' }}">{{ flag_count }}</div>
        <div class="sub">Analyst flags raised</div>
      </div>
      <div class="kpi-card">
        <div class="label">Retry Attempts</div>
        <div class="value">{{ retry_count }}</div>
        <div class="sub">Pipeline retries</div>
      </div>
    </div>
  </section>

  {% elif section == "confidence" %}
  <!-- CONFIDENCE BREAKDOWN -->
  <section>
    <h2>Confidence Vector Breakdown</h2>
    {% for dim, score in confidence_vector.items() %}
    <div class="conf-row">
      <div class="conf-label">{{ dim | replace('_', ' ') | title }}</div>
      <div class="conf-bar" style="flex:1; margin: 0 16px;"><div class="conf-fill" style="width: {{ [score*100, 100]|min }}%;"></div></div>
      <div class="conf-score">{{ '%.0f' | format(score * 100) }}%</div>
    </div>
    {% endfor %}
  </section>

  {% elif section == "model_performance" %}
  <!-- MODEL PERFORMANCE -->
  {% if model_metrics %}
  <section>
    <h2>Model Performance</h2>
    <table>
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>
        {% for k, v in model_metrics.items() %}
        <tr><td>{{ k | replace('_', ' ') | title }}</td><td>{{ v }}</td></tr>
        {% endfor %}
      </tbody>
    </table>
  </section>
  {% endif %}

  {% elif section == "data_quality" %}
  <!-- DATA QUALITY -->
  {% if analyst_flags %}
  <section>
    <h2>Data Quality Flags</h2>
    <table>
      <thead><tr><th>Flag</th><th>Column</th><th>Detail</th></tr></thead>
      <tbody>
        {% for flag in analyst_flags %}
        <tr>
          <td><span class="badge {{ 'fail' if flag.flag in ['STRONG_CORRELATION', 'SIGNIFICANT_DRIFT', 'HIGH_NULL'] else 'warn' }}">{{ flag.flag }}</span></td>
          <td>{{ flag.column | default('—') }}</td>
          <td>{{ flag.detail | default('') | truncate(120) }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </section>
  {% endif %}

  {% elif section == "risk" %}
  <!-- RISK SUMMARY -->
  {% if risk_flags %}
  <section>
    <h2>Risk Communication</h2>
    {% for risk in risk_flags %}
    <div class="risk-item">
      <span class="risk-level risk-{{ risk.level }}">{{ risk.level }}</span>
      <div>
        <strong>{{ risk.category }}</strong>
        <div style="color:#8b949e; font-size:0.85rem; margin-top:4px;">{{ risk.message }}</div>
      </div>
    </div>
    {% endfor %}
  </section>
  {% endif %}

  {% elif section == "narrative" %}
  <!-- NARRATIVE -->
  {% if narrative %}
  <section>
    <h2>Verified Analytics Narrative</h2>
    <div class="narrative-box">{{ narrative }}</div>
  </section>
  {% endif %}

  {% elif section == "audit" %}
  <!-- AUDIT METADATA -->
  <section>
    <h2>Audit & Provenance</h2>
    <table>
      <thead><tr><th>Field</th><th>Value</th></tr></thead>
      <tbody>
        <tr><td>Run ID</td><td>{{ run_id }}</td></tr>
        <tr><td>Data Fingerprint</td><td>{{ fingerprint | default('N/A') }}</td></tr>
        <tr><td>Schema Version</td><td>{{ schema_version | default('1.0') }}</td></tr>
        <tr><td>Governance Decision</td><td><span class="badge {{ 'pass' if gov_decision == 'PASS' else ('warn' if gov_decision == 'WARN' else 'fail') }}">{{ gov_decision | default('N/A') }}</span></td></tr>
        <tr><td>Report Generated</td><td>{{ generated_at }}</td></tr>
      </tbody>
    </table>
  </section>
  {% endif %}
  {% endfor %}

</div>

<div class="footer">
  Generated by DIPEX Cognitive Analytics Automation System &mdash;
  Deterministic-first &bull; Statistically disciplined &bull; Governance-enforced
</div>

</body>
</html>
"""


class ExecutiveReportGenerator:
    """
    Generates HTML executive reports.

    Usage::

        reporter = ExecutiveReportGenerator(config)
        path = reporter.generate(run_id, approved_output, extra_data)
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self._report_dir = (config or {}).get("storage", {}).get("report_dir", "reports")
        self._domain = (config or {}).get("validation", {}).get("regulatory", {}).get("domain", "generic")
        os.makedirs(self._report_dir, exist_ok=True)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ExecutiveReportGenerator":
        return cls(config)

    def generate(
        self,
        run_id: str,
        confidence_vector: Dict[str, Any],
        gate1_decision: str,
        gate2_decision: str,
        narrative: str = "",
        analyst_flags: Optional[List[Dict]] = None,
        model_metrics: Optional[Dict[str, Any]] = None,
        risk_flags: Optional[List[Dict]] = None,
        fingerprint: str = "",
        schema_version: str = "1.0",
        row_count: int = 0,
        col_count: int = 0,
        flag_count: int = 0,
        retry_count: int = 0,
        gov_decision: str = "N/A",
    ) -> str:
        """Render and save an executive HTML report. Returns file path."""
        try:
            from jinja2 import Environment, BaseLoader
        except ImportError:
            logger.error("jinja2 not installed — executive report skipped.")
            return ""

        confidence_score = confidence_vector.get("confidence_score", 0.0)

        # Determine QA status
        if gate1_decision == "PASS" and gate2_decision == "PASS":
            if confidence_score >= 0.80:
                qa_status = "QA_APPROVED"
            else:
                qa_status = "QA_CONDITIONAL"
        else:
            qa_status = "QA_REJECTED"

        # Build confidence dimension display
        conf_dims = {
            k: v for k, v in confidence_vector.items()
            if isinstance(v, (int, float)) and k != "confidence_score"
        }

        # Fetch section order from RL Agent
        try:
            from explanation.rl_narrative import get_rl_agent
            rl_agent = get_rl_agent()
            section_order = rl_agent.get_section_order(self._domain)
            logger.debug("[RL] Executive report section order: %s", section_order)
        except Exception:
            section_order = ["kpi", "confidence", "model_performance", "data_quality", "risk", "narrative", "audit"]

        env = Environment(loader=BaseLoader())
        template = env.from_string(REPORT_TEMPLATE)
        html = template.render(
            run_id=run_id,
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            qa_status=qa_status,
            domain=self._domain,
            confidence_score=confidence_score,
            gate1=gate1_decision,
            gate2=gate2_decision,
            row_count=row_count,
            col_count=col_count,
            flag_count=flag_count,
            retry_count=retry_count,
            confidence_vector=conf_dims,
            model_metrics=model_metrics or {},
            analyst_flags=analyst_flags or [],
            risk_flags=risk_flags or [],
            narrative=narrative,
            fingerprint=fingerprint,
            schema_version=schema_version,
            gov_decision=gov_decision,
            badge=qa_status,
            section_order=section_order,
        )

        report_path = os.path.join(self._report_dir, f"{run_id}_executive_report.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info("Executive report saved: %s", report_path)
        return report_path
