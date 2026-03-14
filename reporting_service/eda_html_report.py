"""
reporting_service/eda_html_report.py
---------------------------------------
Upgrade 4 — Automated EDA Visual HTML Report Generator.

Converts the EDAReport dict (from eda/auto_eda.py) into a self-contained,
interactive HTML file with:
  - Summary stats table (rows, columns, nulls, duplicates)
  - Distribution histograms for numeric columns (Chart.js bar charts)
  - Correlation heatmap (top 15 pairs, colored table)
  - Missing value bar chart per column
  - Outlier summary table
  - Auto-generated insights from EDA (human-readable bullets)
  - Anomaly rate badge (from CleaningReport.anomaly_report)

All JS/CSS is inlined from CDN — one .html file, no external deps at read time.

Usage::

    gen  = EDAHTMLReportGenerator()
    path = gen.generate(eda_report_dict, run_id="abc123",
                        output_dir="reports/", df=df_cleaned)
    # → "reports/eda_abc123.html"
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger("dipex.reporting_service.eda_html_report")


# ─────────────────────────────────────────────────────────────────────────────
# HTML template (Jinja2-free — pure string.format / f-string)
# ─────────────────────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DIPEX EDA Report — {run_id}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0f1117; --card: #1a1d2e; --accent: #6c63ff; --accent2: #00d4aa;
    --text: #e0e0ff; --muted: #8888aa; --warn: #ffb347; --danger: #ff6b6b;
    --ok: #52fa7c; --border: #2a2d4a;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif;
         font-size: 14px; line-height: 1.6; padding: 20px; }}
  h1 {{ color: var(--accent); font-size: 1.8em; margin-bottom: 4px; }}
  h2 {{ color: var(--accent2); font-size: 1.15em; margin: 18px 0 10px; letter-spacing:.5px; text-transform:uppercase; }}
  .meta {{ color: var(--muted); font-size:.85em; margin-bottom:20px; }}
  .grid {{ display: grid; gap: 16px; }}
  .g2 {{ grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }}
  .g3 {{ grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px;
           padding: 18px; box-shadow: 0 4px 20px rgba(0,0,0,.4); }}
  .stat-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(140px,1fr)); gap:12px; margin-bottom:20px; }}
  .stat {{ background:var(--card); border:1px solid var(--border); border-radius:10px;
           padding:14px 18px; text-align:center; }}
  .stat .val {{ font-size:2em; font-weight:700; color:var(--accent); }}
  .stat .lbl {{ color:var(--muted); font-size:.8em; margin-top:2px; }}
  table {{ border-collapse: collapse; width: 100%; font-size:.85em; }}
  th {{ background:#252840; color:var(--accent2); padding:8px 10px; text-align:left;
        font-weight:600; border-bottom:2px solid var(--border); }}
  td {{ padding:7px 10px; border-bottom:1px solid var(--border); }}
  tr:hover td {{ background:rgba(108,99,255,.07); }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:20px; font-size:.78em;
             font-weight:600; letter-spacing:.3px; }}
  .ok {{ background:#052b14; color:var(--ok); border:1px solid #0d5c2b; }}
  .warn {{ background:#2b1900; color:var(--warn); border:1px solid #5c3600; }}
  .danger {{ background:#2b0000; color:var(--danger); border:1px solid #5c0000; }}
  .insight-list {{ list-style:none; padding:0; }}
  .insight-list li {{ padding: 7px 0; border-bottom:1px solid var(--border);
                       display:flex; gap:10px; align-items:center; }}
  .insight-list li::before {{ content:"●"; color:var(--accent); font-size:.7em; }}
  .chart-wrap {{ position:relative; height:200px; }}
  .heatmap-cell {{ text-align:center; font-size:.8em; font-weight:600; }}
  footer {{ margin-top:30px; color:var(--muted); font-size:.8em; text-align:center;
            border-top:1px solid var(--border); padding-top:12px; }}
  .section {{ margin-bottom: 28px; }}
</style>
</head>
<body>

<h1>📊 DIPEX EDA Report</h1>
<p class="meta">Run ID: <strong>{run_id}</strong> &nbsp;·&nbsp; Generated: {timestamp} &nbsp;·&nbsp; Dataset: {n_rows} rows × {n_cols} cols</p>

<!-- ── SUMMARY STATS ──────────────────────────────────────────────── -->
<div class="section">
<h2>📋 Dataset Overview</h2>
<div class="stat-grid">
  <div class="stat"><div class="val">{n_rows}</div><div class="lbl">Rows</div></div>
  <div class="stat"><div class="val">{n_cols}</div><div class="lbl">Columns</div></div>
  <div class="stat"><div class="val">{null_pct}%</div><div class="lbl">Null Rate</div></div>
  <div class="stat"><div class="val">{num_numeric}</div><div class="lbl">Numeric Cols</div></div>
  <div class="stat"><div class="val">{num_categorical}</div><div class="lbl">Categorical Cols</div></div>
  <div class="stat"><div class="val">{anomaly_pct}%</div><div class="lbl">Anomaly Rate</div></div>
</div>
</div>

<!-- ── AUTO INSIGHTS ──────────────────────────────────────────────── -->
<div class="section">
<h2>💡 Auto-Generated Insights</h2>
<div class="card">
<ul class="insight-list">{insights_html}</ul>
</div>
</div>

<!-- ── MISSING VALUES ─────────────────────────────────────────────── -->
<div class="section">
<h2>🕳️ Missing Value Profile</h2>
<div class="card">
  <div class="chart-wrap"><canvas id="missingChart"></canvas></div>
</div>
</div>

<!-- ── DISTRIBUTIONS ─────────────────────────────────────────────── -->
<div class="section">
<h2>📈 Numeric Distributions</h2>
<div class="grid g2">{dist_charts_html}</div>
</div>

<!-- ── CORRELATION ────────────────────────────────────────────────── -->
<div class="section">
<h2>🔗 Top Correlations</h2>
<div class="card">
<table><thead><tr><th>Column A</th><th>Column B</th><th>Pearson r</th><th>Strength</th></tr></thead>
<tbody>{corr_rows_html}</tbody>
</table>
</div>
</div>

<!-- ── OUTLIERS ───────────────────────────────────────────────────── -->
<div class="section">
<h2>⚠️ Outlier Summary</h2>
<div class="card">
<table><thead><tr><th>Column</th><th>Outlier Count</th><th>Outlier %</th><th>Method</th><th>Status</th></tr></thead>
<tbody>{outlier_rows_html}</tbody>
</table>
</div>
</div>

<footer>Generated by DIPEX Automated EDA Engine &nbsp;·&nbsp; {timestamp}</footer>

<script>
// ── Missing values chart ──────────────────────────────────────────────────
(function() {{
  const labels = {missing_labels};
  const values = {missing_values};
  if (!labels.length) return;
  const ctx = document.getElementById('missingChart').getContext('2d');
  new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels: labels,
      datasets: [{{ label: 'Null %', data: values,
        backgroundColor: values.map(v => v > 30 ? '#ff6b6b99' : v > 10 ? '#ffb34799' : '#6c63ff99'),
        borderColor: values.map(v => v > 30 ? '#ff6b6b' : v > 10 ? '#ffb347' : '#6c63ff'),
        borderWidth: 1 }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{ label: ctx => ctx.parsed.y.toFixed(2) + '%' }} }} }},
      scales: {{
        x: {{ ticks: {{ color: '#8888aa', maxRotation: 45 }}, grid: {{ color: '#2a2d4a' }} }},
        y: {{ ticks: {{ color: '#8888aa', callback: v => v + '%' }}, grid: {{ color: '#2a2d4a' }}, beginAtZero: true }}
      }}
    }}
  }});
}})();

// ── Distribution charts ───────────────────────────────────────────────────
{dist_chart_scripts}
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Generator class
# ─────────────────────────────────────────────────────────────────────────────

class EDAHTMLReportGenerator:
    """
    Generates a self-contained, interactive EDA HTML report from an EDAReport dict.

    The report contains Chart.js visualisations (distributions, missing values,
    correlations) and a full outlier + insight summary.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = (config or {}).get("reporting", {}).get("eda_html", {})
        self.output_dir: str = str(cfg.get("output_dir", "reports"))
        self.max_dist_cols: int = int(cfg.get("max_distribution_charts", 20))

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        eda_report: Dict[str, Any],
        run_id: str,
        output_dir: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        anomaly_report: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Render and write the EDA HTML report.

        Parameters
        ----------
        eda_report   : dict from AutoEDA.run()
        run_id       : pipeline run ID
        output_dir   : where to write the file (overrides config)
        df           : original cleaned DataFrame (used to compute distribution bins)
        anomaly_report : dict from AnomalyScorer.score() (optional)

        Returns
        -------
        Absolute path to the generated .html file
        """
        out_dir = output_dir or self.output_dir
        os.makedirs(out_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ── Extract data from EDA report ──────────────────────────────────────
        numeric_stats  = eda_report.get("numeric_stats", {})
        cat_stats      = eda_report.get("categorical_stats", {})
        correlations   = eda_report.get("correlations", [])
        outlier_info   = eda_report.get("outliers", {})
        missing_info   = eda_report.get("missing_values", {})
        insights       = eda_report.get("insights", [])
        summary        = eda_report.get("summary", {})

        n_rows = int(summary.get("n_rows", len(df) if df is not None else 0))
        n_cols = int(summary.get("n_cols", len(df.columns) if df is not None else 0))
        null_pct = round(float(summary.get("overall_null_pct", 0)) * 100, 2)
        num_numeric = len(numeric_stats)
        num_categorical = len(cat_stats)

        anomaly_pct = 0.0
        if anomaly_report:
            anomaly_pct = round(float(anomaly_report.get("anomaly_pct", 0)) * 100, 2)

        # ── Build insight bullets ─────────────────────────────────────────────
        if insights:
            insights_html = "\n".join(
                f"<li>{self._escape(str(ins))}</li>"
                for ins in insights[:30]
            )
        else:
            insights_html = "<li>No auto-insights generated for this dataset.</li>"

        # ── Missing value chart data ──────────────────────────────────────────
        missing_cols   = []
        missing_values_list: List[float] = []
        if isinstance(missing_info, dict):
            for col, info in missing_info.items():
                pct = info.get("null_pct", 0) if isinstance(info, dict) else float(info)
                if pct > 0:
                    missing_cols.append(col)
                    missing_values_list.append(round(float(pct) * 100, 2))

        if not missing_cols and df is not None:
            null_pcts = df.isnull().mean()
            for col in null_pcts[null_pcts > 0].index[:30]:
                missing_cols.append(col)
                missing_values_list.append(round(float(null_pcts[col]) * 100, 2))

        # ── Distribution charts (numeric columns) ─────────────────────────────
        dist_charts_html = ""
        dist_chart_scripts = ""
        if df is not None:
            num_cols_list = df.select_dtypes(include=["number"]).columns.tolist()
            for col in num_cols_list[:self.max_dist_cols]:
                series = df[col].dropna()
                if len(series) < 5:
                    continue
                chart_id = f"dist_{re.sub(r'[^a-zA-Z0-9]', '_', col)}"
                try:
                    import numpy as np
                    counts, bin_edges = np.histogram(series, bins=20)
                    bin_labels = [f"{b:.2f}" for b in bin_edges[:-1]]
                    counts_list = counts.tolist()
                    skew = round(float(series.skew()), 3)
                    mean = round(float(series.mean()), 4)
                    dist_charts_html += f"""
<div class="card">
  <div style="font-weight:600;margin-bottom:8px;color:var(--accent)">{self._escape(col)}</div>
  <div style="color:var(--muted);font-size:.8em;margin-bottom:6px">Mean: {mean} &nbsp;·&nbsp; Skew: {skew}</div>
  <div class="chart-wrap"><canvas id="{chart_id}"></canvas></div>
</div>"""
                    dist_chart_scripts += f"""
(function() {{
  const ctx = document.getElementById('{chart_id}').getContext('2d');
  new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels: {json.dumps(bin_labels)},
      datasets: [{{ label: '{self._escape(col)}', data: {json.dumps(counts_list)},
        backgroundColor: '#6c63ff55', borderColor: '#6c63ff', borderWidth: 1 }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ ticks: {{ color: '#8888aa', maxRotation: 45, maxTicksLimit: 6 }}, grid: {{ color: '#2a2d4a' }} }},
        y: {{ ticks: {{ color: '#8888aa' }}, grid: {{ color: '#2a2d4a' }}, beginAtZero: true }}
      }}
    }}
  }});
}})();"""
                except Exception:
                    continue

        # ── Correlation table ─────────────────────────────────────────────────
        corr_rows_html = ""
        corr_list = correlations if isinstance(correlations, list) else []
        for item in corr_list[:15]:
            a   = item.get("col_a", item.get("feature_a", ""))
            b   = item.get("col_b", item.get("feature_b", ""))
            r   = item.get("correlation", item.get("pearson", 0))
            abs_r = abs(float(r))
            badge = ("danger" if abs_r > 0.7 else "warn" if abs_r > 0.4 else "ok")
            strength = ("Strong" if abs_r > 0.7 else "Moderate" if abs_r > 0.4 else "Weak")
            corr_rows_html += (
                f"<tr><td>{self._escape(str(a))}</td><td>{self._escape(str(b))}</td>"
                f"<td>{round(float(r), 4)}</td>"
                f"<td><span class='badge {badge}'>{strength}</span></td></tr>\n"
            )
        if not corr_rows_html:
            corr_rows_html = "<tr><td colspan=4 style='color:var(--muted)'>No correlations computed.</td></tr>"

        # ── Outlier table ─────────────────────────────────────────────────────
        outlier_rows_html = ""
        if isinstance(outlier_info, dict):
            for col, info in outlier_info.items():
                if not isinstance(info, dict):
                    continue
                count = int(info.get("count", info.get("outlier_count", 0)))
                pct   = round(float(info.get("pct", info.get("outlier_pct", 0))) * 100, 2)
                method = str(info.get("method", "IQR"))
                badge = "danger" if pct > 10 else "warn" if pct > 3 else "ok"
                outlier_rows_html += (
                    f"<tr><td>{self._escape(col)}</td><td>{count}</td><td>{pct}%</td>"
                    f"<td>{method}</td>"
                    f"<td><span class='badge {badge}'>{'High' if pct>10 else 'Watch' if pct>3 else 'OK'}</span></td></tr>\n"
                )
        if not outlier_rows_html:
            outlier_rows_html = "<tr><td colspan=5 style='color:var(--muted)'>No outlier data available.</td></tr>"

        # ── Render ────────────────────────────────────────────────────────────
        html = _HTML_TEMPLATE.format(
            run_id           = self._escape(run_id),
            timestamp        = timestamp,
            n_rows           = n_rows,
            n_cols           = n_cols,
            null_pct         = null_pct,
            num_numeric      = num_numeric,
            num_categorical  = num_categorical,
            anomaly_pct      = anomaly_pct,
            insights_html    = insights_html,
            missing_labels   = json.dumps(missing_cols),
            missing_values   = json.dumps(missing_values_list),
            dist_charts_html = dist_charts_html,
            dist_chart_scripts = dist_chart_scripts,
            corr_rows_html   = corr_rows_html,
            outlier_rows_html= outlier_rows_html,
        )

        fname = f"eda_{run_id[:16].replace('/', '_').replace(':', '_')}.html"
        fpath = os.path.join(out_dir, fname)

        with open(fpath, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info("[EDAHTMLReport] Written to %s", fpath)
        return os.path.abspath(fpath)

    @staticmethod
    def _escape(s: str) -> str:
        """Minimal HTML-escape for embedding strings in HTML."""
        return (
            s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
             .replace("'", "&#39;")
        )
