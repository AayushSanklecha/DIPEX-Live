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
    /* EDA styles */
    .eda-stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 24px; }
    .eda-stat { background: #161b22; border: 1px solid #21262d; border-radius: 10px; padding: 16px; text-align: center; }
    .eda-stat .val { font-size: 1.8rem; font-weight: 700; color: #21d4fd; }
    .eda-stat .lbl { color: #8b949e; font-size: 0.75rem; margin-top: 4px; text-transform: uppercase; letter-spacing: .5px; }
    .eda-insight-list { list-style: none; padding: 0; }
    .eda-insight-list li { background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; font-size: 0.9rem; display: flex; gap: 10px; align-items: flex-start; }
    .eda-insight-list li::before { content: "›"; color: #21d4fd; font-size: 1.2em; flex-shrink: 0; margin-top: -1px; }
    .eda-chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 24px; }
    .eda-chart-card { background: #161b22; border: 1px solid #21262d; border-radius: 10px; padding: 16px; }
    .eda-chart-card .chart-title { color: #21d4fd; font-size: 0.85rem; font-weight: 600; margin-bottom: 4px; }
    .eda-chart-card .chart-sub { color: #8b949e; font-size: 0.78rem; margin-bottom: 10px; }
    .eda-chart-wrap { position: relative; height: 160px; }
    .eda-link { display: inline-block; margin-top: 20px; padding: 10px 20px; background: rgba(33,212,253,0.1); border: 1px solid #21d4fd; color: #21d4fd; border-radius: 8px; text-decoration: none; font-size: 0.85rem; font-weight: 600; }
    .eda-link:hover { background: rgba(33,212,253,0.2); }
    .eda-badge-ok   { background: rgba(63,185,80,0.15); color: #3fb950; border: 1px solid #3fb950; padding: 3px 8px; border-radius: 10px; font-size: 0.75rem; font-weight: 600; }
    .eda-badge-warn { background: rgba(210,153,34,0.15); color: #d29922; border: 1px solid #d29922; padding: 3px 8px; border-radius: 10px; font-size: 0.75rem; font-weight: 600; }
    .eda-badge-danger { background: rgba(248,81,73,0.15); color: #f85149; border: 1px solid #f85149; padding: 3px 8px; border-radius: 10px; font-size: 0.75rem; font-weight: 600; }
    /* Insight callout boxes — inline reasoning beneath each stat/chart */
    .insight-callout { background: rgba(33,212,253,0.04); border-left: 3px solid #21d4fd; border-radius: 0 6px 6px 0; padding: 10px 14px; margin-top: 8px; font-size: 0.82rem; color: #b0bec5; line-height: 1.6; font-style: italic; }
    .insight-callout .insight-label { color: #21d4fd; font-style: normal; font-weight: 600; font-size: 0.72rem; text-transform: uppercase; letter-spacing: .6px; display: block; margin-bottom: 4px; }
    .insight-callout .insight-rec  { color: #3fb950; font-style: normal; margin-top: 6px; font-size: 0.80rem; display: block; }
    .insight-callout .insight-warn { color: #d29922; font-style: normal; margin-top: 4px; font-size: 0.80rem; display: block; }
    .insight-callout .insight-flag { color: #f85149; font-style: normal; margin-top: 4px; font-size: 0.80rem; display: block; }
    .model-metric-card { background: #161b22; border: 1px solid #21262d; border-radius: 10px; padding: 14px 18px; margin-bottom: 10px; }
    @media print {
      body { background: #fff; color: #000; }
      .cover { background: #1a1f35; -webkit-print-color-adjust: exact; }
    }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
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

  {% elif section == "data_prep" %}
  <!-- DATA PREPARATION APPLIED -->
  {% if actions_log %}
  <section>
    <h2>Data Preparation Applied</h2>
    <table>
      <thead><tr><th>Feature</th><th>Action Applied</th><th>Reason</th></tr></thead>
      <tbody>
        {% for col, info in actions_log.items() %}
        <tr>
          <td><strong>{{ col }}</strong></td>
          <td><span class="badge pass">✅ {{ info.action }}</span></td>
          <td>{{ info.reason }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </section>
  {% endif %}

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
    <div id="raw-narrative" style="display:none;">{{ narrative }}</div>
    <div class="narrative-box" id="rendered-narrative"></div>
    <script>
      if (typeof marked !== 'undefined') {
        document.getElementById('rendered-narrative').innerHTML = marked.parse(document.getElementById('raw-narrative').innerText || document.getElementById('raw-narrative').textContent);
      } else {
        document.getElementById('rendered-narrative').innerText = document.getElementById('raw-narrative').innerText;
      }
    </script>
  </section>
  {% endif %}

  {% elif section == "eda" %}
  <!-- EDA REPORT SECTION -->
  {% if eda_summary %}
  <section>
    <h2>📊 Exploratory Data Analysis</h2>

    <!-- Stats row -->
    <div class="eda-stat-grid">
      <div class="eda-stat"><div class="val">{{ eda_summary.n_rows }}</div><div class="lbl">Rows</div></div>
      <div class="eda-stat"><div class="val">{{ eda_summary.n_cols }}</div><div class="lbl">Columns</div></div>
      <div class="eda-stat"><div class="val">{{ eda_summary.null_pct }}%</div><div class="lbl">Null Rate</div></div>
      <div class="eda-stat"><div class="val">{{ eda_summary.numeric_cols }}</div><div class="lbl">Numeric</div></div>
      <div class="eda-stat"><div class="val">{{ eda_summary.categorical_cols }}</div><div class="lbl">Categorical</div></div>
      <div class="eda-stat">
        <div class="val" style="color:{{ '#f85149' if eda_summary.anomaly_pct > 20 else ('#d29922' if eda_summary.anomaly_pct > 5 else '#3fb950') }}">{{ eda_summary.anomaly_pct }}%</div>
        <div class="lbl">Anomaly Rate</div>
      </div>
    </div>

    <!-- Auto Insights -->
    {% if eda_insights %}
    <h3 style="color:#8b949e; font-size:0.85rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px;">💡 Auto-Generated Insights</h3>
    <ul class="eda-insight-list" style="margin-bottom:28px;">
      {% for ins in eda_insights %}
      <li>{{ ins }}</li>
      {% endfor %}
    </ul>
    {% endif %}

    <!-- Distribution charts -->
    {% if eda_dist_charts %}
    <h3 style="color:#8b949e; font-size:0.85rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px;">📈 Numeric Distributions</h3>
    <div class="eda-chart-grid">
      {% for chart in eda_dist_charts %}
      <div class="eda-chart-card">
        <div class="chart-title">{{ chart.col }}</div>
        <div class="chart-sub">Mean: {{ chart.mean }} · Skew: {{ chart.skew }}</div>
        <div class="eda-chart-wrap"><canvas id="edadist_{{ chart.id }}"></canvas></div>
        {% if eda_col_interps and chart.col in eda_col_interps %}
        <div class="insight-callout">
          <span class="insight-label">📖 What this means</span>
          {{ eda_col_interps[chart.col].meaning }}
          <span class="insight-rec">{{ eda_col_interps[chart.col].recommendation }}</span>
          {% if eda_col_interps[chart.col].flag is defined %}
          <span class="insight-flag">{{ eda_col_interps[chart.col].flag }}</span>
          {% endif %}
        </div>
        {% endif %}
      </div>
      {% endfor %}
    </div>
    <script>
    {% for chart in eda_dist_charts %}
    (function(){
      var ctx = document.getElementById('edadist_{{ chart.id }}').getContext('2d');
      new Chart(ctx, {
        type: 'bar',
        data: { labels: {{ chart.labels | tojson }}, datasets: [{ data: {{ chart.counts | tojson }},
          backgroundColor: '#21d4fd33', borderColor: '#21d4fd', borderWidth: 1 }] },
        options: { responsive:true, maintainAspectRatio:false,
          plugins:{legend:{display:false}},
          scales:{ x:{ticks:{color:'#8b949e',maxRotation:45,maxTicksLimit:5},grid:{color:'#21262d'}},
                   y:{ticks:{color:'#8b949e'},grid:{color:'#21262d'},beginAtZero:true} } }
      });
    })();
    {% endfor %}
    </script>
    {% endif %}

    <!-- Correlations -->
    {% if eda_correlations %}
    <h3 style="color:#8b949e; font-size:0.85rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px; margin-top:24px;">🔗 Top Correlations</h3>
    <table style="margin-bottom:12px;">
      <thead><tr><th>Column A</th><th>Column B</th><th>Pearson r</th><th>Strength</th></tr></thead>
      <tbody>
      {% for c in eda_correlations %}
        <tr>
          <td>{{ c.a }}</td><td>{{ c.b }}</td><td>{{ c.r }}</td>
          <td><span class="eda-badge-{{ c.badge }}">{{ c.strength }}</span></td>
        </tr>
        {% if c.narration is defined %}
        <tr><td colspan="4" style="background:transparent;padding:0 6px 10px;">
          <div class="insight-callout"><span class="insight-label">📖 Interpretation</span>{{ c.narration }}</div>
        </td></tr>
        {% endif %}
      {% endfor %}
      </tbody>
    </table>
    {% if eda_multicollinearity_warn %}
    <div class="insight-callout" style="margin-bottom:20px;">
      <span class="insight-label">⚠️ Multicollinearity Warning</span>
      <span class="insight-warn">{{ eda_multicollinearity_warn }}</span>
    </div>
    {% endif %}
    {% endif %}

    <!-- Outliers -->
    {% if eda_outliers %}
    <h3 style="color:#8b949e; font-size:0.85rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px;">⚠️ Outlier Summary</h3>
    <table style="margin-bottom:12px;">
      <thead><tr><th>Column</th><th>Count</th><th>Pct</th><th>Method</th><th>Status</th></tr></thead>
      <tbody>
      {% for o in eda_outliers %}
        <tr>
          <td>{{ o.col }}</td><td>{{ o.count }}</td><td>{{ o.pct }}%</td><td>{{ o.method }}</td>
          <td><span class="eda-badge-{{ o.badge }}">{{ o.status }}</span></td>
        </tr>
        {% if o.explanation is defined %}
        <tr><td colspan="5" style="background:transparent;padding:0 6px 10px;">
          <div class="insight-callout"><span class="insight-label">📖 What this means</span>{{ o.explanation }}</div>
        </td></tr>
        {% endif %}
      {% endfor %}
      </tbody>
    </table>
    {% endif %}

    <!-- Anomaly context -->
    {% if eda_anomaly_narration %}
    <div class="insight-callout" style="margin-bottom:20px;">
      <span class="insight-label">🤖 Anomaly Interpretation (Isolation Forest)</span>
      {{ eda_anomaly_narration }}
    </div>
    {% endif %}

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

{% if eda_html_content %}
<div class="container" style="max-width: 1400px; margin-top: 40px; margin-bottom: 40px;">
  <h2>Full Interactive EDA Report</h2>
  <iframe srcdoc="{{ eda_html_content | e }}" width="100%" height="800px" style="border: 1px solid #e2e8f0; border-radius: 8px;"></iframe>
</div>
{% endif %}

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
        eda_report: Optional[Dict[str, Any]] = None,
        eda_html_path: Optional[str] = "",
        actions_log: Optional[Dict[str, Dict[str, str]]] = None,
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
            section_order = ["kpi", "confidence", "model_performance", "data_prep", "data_quality", "risk", "narrative", "audit", "eda"]

        import re as _re
        import json as _json

        # ── Build EDA template vars from eda_report dict ──────────────────────
        _eda = eda_report or {}
        _summary_raw = _eda.get("summary", {})
        _numeric_stats   = _eda.get("numeric_stats", {})
        _cat_stats       = _eda.get("categorical_stats", {})
        _missing_raw     = _eda.get("missing_values", {}) or {}
        _correlations_raw = _eda.get("correlations", []) or []
        _outliers_raw    = _eda.get("outliers", {}) or {}

        eda_summary = {
            "n_rows":          _summary_raw.get("n_rows", row_count),
            "n_cols":          _summary_raw.get("n_cols", col_count),
            "null_pct":        round(float(_summary_raw.get("overall_null_pct", 0)) * 100, 2),
            "numeric_cols":    len(_numeric_stats),
            "categorical_cols": len(_cat_stats),
            "anomaly_pct":     round(float(_summary_raw.get("anomaly_pct", 0)) * 100, 2),
        } if _eda else None

        eda_insights = (_eda.get("insights") or [])[:10]

        # Distribution charts — build histogram bins from numeric_stats if pre-computed
        eda_dist_charts: list = []
        for col, stats in list(_numeric_stats.items())[:12]:
            if not isinstance(stats, dict):
                continue
            bins = stats.get("histogram_bins") or stats.get("bins")
            counts = stats.get("histogram_counts") or stats.get("counts")
            if bins and counts:
                chart_id = _re.sub(r"[^a-zA-Z0-9]", "_", col)
                eda_dist_charts.append({
                    "col":    col,
                    "id":     chart_id,
                    "mean":   round(float(stats.get("mean", 0)), 4),
                    "skew":   round(float(stats.get("skewness", stats.get("skew", 0))), 3),
                    "labels": [str(round(float(b), 2)) for b in bins],
                    "counts": [int(c) for c in counts],
                })

        # Correlations
        _correlations_raw = _eda.get("correlations", [])
        if not isinstance(_correlations_raw, list):
            _correlations_raw = []
        eda_correlations: list = []
        for item in _correlations_raw[:10]:
            r = float(item.get("correlation", item.get("pearson", 0)))
            abs_r = abs(r)
            eda_correlations.append({
                "a":        item.get("col_a", item.get("feature_a", "")),
                "b":        item.get("col_b", item.get("feature_b", "")),
                "r":        round(r, 4),
                "badge":    "danger" if abs_r > 0.7 else "warn" if abs_r > 0.4 else "ok",
                "strength": "Strong" if abs_r > 0.7 else "Moderate" if abs_r > 0.4 else "Weak",
            })

        # Outliers
        eda_outliers: list = []
        for col, info in _outliers_raw.items():
            if not isinstance(info, dict):
                continue
            pct = round(float(info.get("pct", info.get("outlier_pct", 0))) * 100, 2)
            eda_outliers.append({
                "col":    col,
                "count":  int(info.get("count", info.get("outlier_count", 0))),
                "pct":    pct,
                "method": info.get("method", "IQR"),
                "badge":  "danger" if pct > 10 else "warn" if pct > 3 else "ok",
                "status": "High" if pct > 10 else "Watch" if pct > 3 else "OK",
            })

        # ── InsightNarrator — compute inline reasoning for all sections ──────
        eda_col_interps: dict = {}
        eda_multicollinearity_warn: str = ""
        eda_anomaly_narration: str = ""
        try:
            from reporting_service.insight_narrator import (
                ColumnInterpreter, CorrelationNarrator,
                OutlierExplainer, AnomalyExplainer,
            )
            _col_interp = ColumnInterpreter()
            _corr_narr  = CorrelationNarrator()
            _out_exp    = OutlierExplainer()
            _anom_exp   = AnomalyExplainer()

            # Column interpretations (for each chart)
            _missing_by_col = _eda.get("missing_values", {}) or {}
            for col, stats in list(_numeric_stats.items())[:16]:
                if not isinstance(stats, dict):
                    continue
                null_c = float(_missing_by_col.get(col, {}).get("null_pct", 0.0) if isinstance(_missing_by_col.get(col), dict) else 0.0)
                zero_c = float(stats.get("zero_pct", 0.0) or 0.0)
                eda_col_interps[col] = _col_interp.interpret(col, stats, null_pct=null_c, zero_pct=zero_c)

            # Add narration to each correlation entry
            for c in eda_correlations:
                c["narration"] = _corr_narr.narrate(c["a"], c["b"], c["r"])
            mc_pairs = [{"a": c["a"], "b": c["b"], "r": c["r"]} for c in eda_correlations]
            eda_multicollinearity_warn = _corr_narr.multicollinearity_warning(mc_pairs) or ""

            # Add explanation to each outlier entry
            for o in eda_outliers:
                o["explanation"] = _out_exp.explain(
                    o["col"], o["count"], o["pct"] / 100.0, o["method"]
                )

            # Anomaly narration
            if eda_summary and eda_summary.get("anomaly_pct", 0) > 0:
                anom_pct_val = float(eda_summary["anomaly_pct"]) / 100.0
                n_rows_val   = int(eda_summary.get("n_rows", row_count) or row_count)
                eda_anomaly_narration = _anom_exp.explain(anom_pct_val, n_rows_val)
        except Exception as _narr_exc:
            logger.debug("InsightNarrator wiring skipped (non-fatal): %s", _narr_exc)

        # Jinja2 env with tojson filter
        import json as _json2
        env = Environment(loader=BaseLoader())
        env.filters["tojson"] = lambda v: _json2.dumps(v)
        template = env.from_string(REPORT_TEMPLATE)
        eda_html_content = ""
        eda_default_path = os.path.join("reports_output", f"eda_profile_{run_id}.html")
        if os.path.exists(eda_default_path):
            try:
                with open(eda_default_path, "r", encoding="utf-8") as rf:
                    eda_html_content = rf.read()
            except Exception as ex:
                logger.warning(f"Could not read EDA HTML from {eda_default_path}: {ex}")
        elif eda_html_path and os.path.exists(eda_html_path):
            try:
                with open(eda_html_path, "r", encoding="utf-8") as rf:
                    eda_html_content = rf.read()
            except Exception as ex:
                logger.warning(f"Could not read EDA HTML: {ex}")

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
            eda_summary=eda_summary,
            eda_insights=eda_insights,
            eda_dist_charts=eda_dist_charts,
            eda_correlations=eda_correlations,
            eda_outliers=eda_outliers,
            eda_html_content=eda_html_content,
            eda_col_interps=eda_col_interps,
            eda_multicollinearity_warn=eda_multicollinearity_warn,
            eda_anomaly_narration=eda_anomaly_narration,
            actions_log=actions_log,
        )

        report_path = os.path.join(self._report_dir, f"{run_id}_executive_report.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info("Executive report saved: %s", report_path)
        return report_path
