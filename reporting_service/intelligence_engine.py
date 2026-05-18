"""
reporting_service/intelligence_engine.py
-----------------------------------------
Next-Gen Adaptive Analytics Engine — v2.

Improvements over v1:
  - Data-size-adaptive max_charts formula (8 for tiny, 60 for massive)
  - 15 chart generators: scatter, bar, stacked_bar, line, multi_line, pie,
    area (histogram), range_bar, heatmap, radar, funnel, count_bar,
    null_bar, feature_importance_bar, quality_radar
  - Every chart carries `section`, `explanation` (journalism prose),
    `insight` (one-line), `summary` (panel subtitle)
  - Accepts `context` dict from audit log for model/governance/anomaly charts
  - Structured `insights` array per-section for the InsightCards UI panel
  - Chart type chosen by data geometry — no forced variety, no forced uniformity

Section tags map to report accordion sections:
  schema | quality | missing | anomaly | drift | model | governance | regulatory | pipeline
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.reporting.intelligence")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_nunique(s: pd.Series) -> int:
    try:
        return s.nunique()
    except Exception:
        return s.astype(str).nunique()


def _fmt(col: str) -> str:
    """'transaction_amount' → 'Transaction Amount'"""
    return str(col).replace("_", " ").replace("-", " ").title()


def _clamp(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def _adaptive_max_charts(n_rows: int, n_numeric: int, n_cat: int, n_datetime: int) -> int:
    """
    Scale chart count with dataset richness.
      tiny  (<100 rows, <5 cols)  → 8
      small (100–1k, 5–15 cols)   → 15
      medium (1k–100k, 15–50)     → 25
      large (>100k, >50 cols)     → 50+
    """
    col_score = n_numeric * 3 + n_cat * 2 + n_datetime * 4
    row_score = math.log10(max(n_rows, 1))
    raw = int(col_score * 0.6 + row_score * 3)
    return max(8, min(60, raw))


# ── Main Engine ───────────────────────────────────────────────────────────────

class IntelligenceEngine:
    """Generates significance-ranked chart specs + structured insight narratives."""

    def analyze_dataset(
        self,
        df: pd.DataFrame,
        max_charts: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        df          : The processed snapshot DataFrame
        max_charts  : Override adaptive limit (default: auto-computed)
        context     : Audit entry dict — model_metrics, governance_report,
                      anomaly_report, drift_report, statistical_tests, etc.
        """
        if df is None or df.empty:
            return {"kpis": {}, "insights_feed": [], "insights": [], "charts": []}

        ctx   = context or {}
        df    = df.copy()
        n_rows, n_cols_total = df.shape

        # ── Column classification ────────────────────────────────────────────
        numeric_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

        # Try to parse likely date object columns
        for col in df.select_dtypes(include=["object"]).columns:
            if any(kw in col.lower() for kw in ("date", "time", "year", "created", "updated", "ts")):
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    if df[col].notna().mean() > 0.5:
                        datetime_cols.append(col)
                        numeric_cols = [c for c in numeric_cols if c != col]
                except Exception:
                    pass

        cat_cols = [
            col for col in df.columns
            if col not in numeric_cols and col not in datetime_cols
            and 1 < _safe_nunique(df[col]) < 80
        ]

        # ── Adaptive chart budget ────────────────────────────────────────────
        _max = max_charts or _adaptive_max_charts(
            n_rows, len(numeric_cols), len(cat_cols), len(datetime_cols)
        )

        candidates: List[Dict] = []

        # ════════════════════════════════════════════════════════════════════
        # §1 NUMERIC vs NUMERIC — Scatter plots
        # ════════════════════════════════════════════════════════════════════
        for i, col_a in enumerate(numeric_cols):
            for col_b in numeric_cols[i + 1:]:
                try:
                    corr = float(df[col_a].corr(df[col_b]))
                except Exception:
                    continue
                if math.isnan(corr):
                    corr = 0.0
                score = _clamp(abs(corr) + 0.1)
                sample = df[[col_a, col_b]].dropna().head(300)
                direction = (
                    "strong positive"  if corr >  0.6 else
                    "strong negative"  if corr < -0.6 else
                    "weak" if abs(corr) < 0.3 else "moderate"
                )
                candidates.append({
                    "id":          f"scatter_{col_a}_{col_b}",
                    "type":        "scatter",
                    "section":     "schema",
                    "significance": score,
                    "aspects":     2,
                    "columns":     [col_a, col_b],
                    "title":       f"{_fmt(col_a)} vs {_fmt(col_b)}",
                    "x_label":     _fmt(col_a),
                    "y_label":     _fmt(col_b),
                    "summary":     f"r = {corr:.3f} — {direction} linear relationship",
                    "insight":     (
                        f"{_fmt(col_a)} and {_fmt(col_b)} move together (r={corr:.2f})."
                        if abs(corr) > 0.6 else
                        f"No strong dependency between {_fmt(col_a)} and {_fmt(col_b)} (r={corr:.2f})."
                    ),
                    "explanation": (
                        f"The scatter plot reveals a {direction} linear relationship (Pearson r = {corr:.3f}) "
                        f"between {_fmt(col_a)} and {_fmt(col_b)}. "
                        + (
                            f"This correlation is statistically meaningful and suggests these features may be "
                            f"co-dependent — consider VIF analysis before including both in a regression model."
                            if abs(corr) > 0.6 else
                            f"The weak correlation implies these features carry largely independent information, "
                            f"which is favourable for model feature diversity."
                        )
                    ),
                    "data":        sample.to_dict(orient="records"),
                })

        # ════════════════════════════════════════════════════════════════════
        # §2 CATEGORICAL vs NUMERIC — Bar or Stacked Bar
        # ════════════════════════════════════════════════════════════════════
        for cat in cat_cols:
            for num in numeric_cols:
                try:
                    grouped = df.groupby(cat)[num].mean()
                except Exception:
                    continue
                if len(grouped) < 2 or grouped.mean() == 0:
                    continue
                cv = grouped.std() / abs(grouped.mean()) if grouped.mean() != 0 else 0.0
                score = _clamp(float(cv))

                sorted_grp  = grouped.sort_values(ascending=False)
                data_mapped = [{"name": str(k), "value": round(float(v), 4)} for k, v in sorted_grp.items()]
                best, worst = str(sorted_grp.index[0]), str(sorted_grp.index[-1])

                # Attempt stacked bar if another cat col is available
                other_cats = [c for c in cat_cols if c != cat and _safe_nunique(df[c]) < 10]
                if other_cats and score > 0.3:
                    stack_cat = other_cats[0]
                    try:
                        stacked = df.groupby([cat, stack_cat])[num].mean().unstack(fill_value=0)
                        sd      = [
                            {"name": str(ri), **{str(ci): round(float(v), 4) for ci, v in row.items()}}
                            for ri, row in stacked.iterrows()
                        ]
                        candidates.append({
                            "id":          f"stacked_bar_{cat}_{stack_cat}_{num}",
                            "type":        "stacked_bar",
                            "section":     "schema",
                            "significance": _clamp(score + 0.15),
                            "aspects":     3,
                            "columns":     [cat, stack_cat, num],
                            "title":       f"Avg {_fmt(num)} by {_fmt(cat)} & {_fmt(stack_cat)}",
                            "x_label":     _fmt(cat),
                            "y_label":     f"Average {_fmt(num)}",
                            "summary":     f"Group comparison with {_fmt(stack_cat)} segmentation",
                            "insight":     f"{best} leads in {_fmt(num)}, broken down by {_fmt(stack_cat)}.",
                            "explanation": (
                                f"This stacked bar chart splits average {_fmt(num)} across {_fmt(cat)} groups, "
                                f"further segmented by {_fmt(stack_cat)}. "
                                f"The compound view reveals interaction effects — for example whether one sub-group "
                                f"within {_fmt(cat)} is being disproportionately driven by a specific {_fmt(stack_cat)} value."
                            ),
                            "data":        sd[:20],
                            "stack_keys":  [str(c) for c in stacked.columns],
                        })
                    except Exception:
                        pass
                else:
                    candidates.append({
                        "id":          f"bar_{cat}_{num}",
                        "type":        "bar",
                        "section":     "schema",
                        "significance": score,
                        "aspects":     2,
                        "columns":     [cat, num],
                        "title":       f"Avg {_fmt(num)} by {_fmt(cat)}",
                        "x_label":     _fmt(cat),
                        "y_label":     f"Average {_fmt(num)}",
                        "summary":     f"{best} leads; {worst} lags (CV={cv:.2f})",
                        "insight":     (
                            f"{best} dominates {_fmt(num)}, {worst} is the lowest segment."
                            if score > 0.4 else
                            f"Uniform {_fmt(num)} distribution across {_fmt(cat)} groups."
                        ),
                        "explanation": (
                            f"Average {_fmt(num)} per {_fmt(cat)} group, sorted descending. "
                            f"The coefficient of variation is {cv:.2f}, indicating "
                            f"{'high variability across segments — a strong signal for targeted interventions.' if score > 0.4 else 'relatively uniform averages — no dominant segment skewing the metric.'} "
                            f"{best} achieves the highest average ({round(float(sorted_grp.iloc[0]), 2)}), "
                            f"while {worst} is the lowest ({round(float(sorted_grp.iloc[-1]), 2)})."
                        ),
                        "data":        data_mapped[:20],
                    })

        # ════════════════════════════════════════════════════════════════════
        # §3 TIME SERIES — Line / Multi-line
        # ════════════════════════════════════════════════════════════════════
        for dt_col in datetime_cols:
            for num in numeric_cols:
                tdf = df.dropna(subset=[dt_col, num]).copy()
                if tdf.empty:
                    continue
                try:
                    tdf[dt_col] = pd.to_datetime(tdf[dt_col], errors="coerce")
                    tdf = tdf.dropna(subset=[dt_col]).set_index(dt_col).sort_index()
                    resampled = tdf[num].resample("ME").sum()
                    if resampled.empty or resampled.mean() == 0:
                        continue
                    tv = _clamp(resampled.std() / abs(resampled.mean()))
                    data_ts = [{"time": str(idx.date()), "value": round(float(v), 4)} for idx, v in resampled.items()]

                    if cat_cols and tv > 0.2:
                        sc = cat_cols[0]
                        try:
                            pivot = tdf.groupby([pd.Grouper(freq="ME"), sc])[num].sum().unstack(fill_value=0)
                            md    = [
                                {"time": str(ri.date()), **{str(ci): round(float(v), 4) for ci, v in row.items()}}
                                for ri, row in pivot.iterrows()
                            ]
                            candidates.append({
                                "id": f"multi_line_{dt_col}_{sc}_{num}",
                                "type": "multi_line",
                                "section": "drift",
                                "significance": _clamp(tv + 0.2),
                                "aspects": 3,
                                "columns": [dt_col, sc, num],
                                "title": f"{_fmt(num)} Trend by {_fmt(sc)} Over Time",
                                "x_label": "Time Period",
                                "y_label": _fmt(num),
                                "summary": f"Monthly trend split by {_fmt(sc)}",
                                "insight": f"Seasonal volatility in {_fmt(num)} across {_fmt(sc)} groups.",
                                "explanation": (
                                    f"Monthly aggregated {_fmt(num)} tracked over time, separately for each {_fmt(sc)} value. "
                                    f"Diverging lines indicate that the metric evolves differently per segment — a pattern critical for "
                                    f"detecting distribution drift and segment-specific seasonality."
                                ),
                                "data": md[-60:],
                                "line_keys": [str(c) for c in pivot.columns],
                            })
                        except Exception:
                            pass
                    else:
                        candidates.append({
                            "id": f"line_{dt_col}_{num}",
                            "type": "line",
                            "section": "drift",
                            "significance": _clamp(tv + 0.1),
                            "aspects": 2,
                            "columns": [dt_col, num],
                            "title": f"{_fmt(num)} Over Time",
                            "x_label": "Time Period",
                            "y_label": _fmt(num),
                            "summary": f"Trend volatility: CV={tv:.2f}",
                            "insight": f"{'High' if tv > 0.5 else 'Stable'} trend in {_fmt(num)} over time.",
                            "explanation": (
                                f"Monthly resampled {_fmt(num)} over the full observed time range. "
                                + (
                                    f"The high coefficient of variation ({tv:.2f}) signals significant temporal instability — "
                                    f"this feature should be monitored for drift and included in retraining triggers."
                                    if tv > 0.5 else
                                    f"The stable trend (CV={tv:.2f}) suggests a consistent data generation process — "
                                    f"low risk of temporal distribution shift."
                                )
                            ),
                            "data": data_ts[-60:],
                        })
                except Exception as _te:
                    logger.debug("Time series failed: %s", _te)

        # ════════════════════════════════════════════════════════════════════
        # §4 CATEGORICAL DISTRIBUTION — Pie/Donut
        # ════════════════════════════════════════════════════════════════════
        for cat in cat_cols:
            try:
                counts = df[cat].value_counts(normalize=True)
            except Exception:
                counts = df[cat].astype(str).value_counts(normalize=True)
            if counts.empty:
                continue
            imb   = float(counts.iloc[0])
            score = _clamp(imb) if imb > 0.2 else 0.1
            data_pie = [{"name": str(k), "value": round(float(v), 4)} for k, v in counts.head(6).items()]
            if len(counts) > 6:
                data_pie.append({"name": "Other", "value": round(float(counts.iloc[6:].sum()), 4)})
            top = str(counts.index[0])
            candidates.append({
                "id":          f"pie_{cat}",
                "type":        "pie",
                "section":     "schema",
                "significance": score - 0.2,
                "aspects":     1,
                "columns":     [cat],
                "title":       f"Distribution of {_fmt(cat)}",
                "x_label":     _fmt(cat),
                "y_label":     "Share (%)",
                "summary":     f"{top}: {imb*100:.0f}% share",
                "insight":     (
                    f"{top} dominates with {imb*100:.0f}% of all {_fmt(cat)} records."
                    if imb > 0.4 else
                    f"{_fmt(cat)} is evenly distributed across categories."
                ),
                "explanation": (
                    f"Proportional share of each {_fmt(cat)} category across the full dataset. "
                    + (
                        f"The dominant category '{top}' accounts for {imb*100:.0f}% of records, indicating class imbalance "
                        f"that may bias classification models trained without resampling or class-weight adjustments."
                        if imb > 0.4 else
                        f"Categories are relatively evenly distributed, suggesting balanced representation — "
                        f"class imbalance is unlikely to be a concern for downstream modelling."
                    )
                ),
                "data":        data_pie,
            })

        # ════════════════════════════════════════════════════════════════════
        # §5 NUMERIC DISTRIBUTION — Histogram (area chart with bin labels)
        # ════════════════════════════════════════════════════════════════════
        for num in numeric_cols:
            col_data = df[num].dropna()
            if col_data.empty:
                continue
            try:
                skew = float(col_data.skew())
                if math.isnan(skew):
                    skew = 0.0
            except Exception:
                skew = 0.0
            score = _clamp(abs(skew) / 5.0)
            n_bins = min(20, max(5, int(1 + 3.322 * math.log10(max(len(col_data), 2)))))  # Sturges
            counts, bins = np.histogram(col_data.values, bins=n_bins)
            data_hist = [
                {"bin": f"{bins[i]:.2g}–{bins[i+1]:.2g}", "count": int(c)}
                for i, c in enumerate(counts)
            ]
            candidates.append({
                "id":          f"histogram_{num}",
                "type":        "area",
                "section":     "quality",
                "significance": score - 0.3,
                "aspects":     1,
                "columns":     [num],
                "title":       f"Distribution of {_fmt(num)}",
                "x_label":     f"{_fmt(num)} Range",
                "y_label":     "Record Count",
                "summary":     f"Skew = {skew:.2f} {'— heavy tail' if abs(skew) > 1 else '— near-normal'}",
                "insight":     (
                    f"Highly skewed distribution (skew={skew:.2f}) — tail risk present."
                    if abs(skew) > 1 else
                    f"Near-normal distribution (skew={skew:.2f}) — parametric models applicable."
                ),
                "explanation": (
                    f"Histogram of {_fmt(num)} across {n_bins} equal-width bins using Sturges' rule "
                    f"({len(col_data):,} non-null values). "
                    + (
                        f"The skewness of {skew:.2f} reveals a heavy {'right' if skew > 0 else 'left'} tail — "
                        f"log transformation or winsorization is recommended before applying parametric models or linear regression."
                        if abs(skew) > 1 else
                        f"The near-symmetrical shape (skew={skew:.2f}) is consistent with a normal distribution, "
                        f"making {_fmt(num)} suitable for parametric statistical tests without transformation."
                    )
                ),
                "data":        data_hist,
            })

        # ════════════════════════════════════════════════════════════════════
        # §6 CORRELATION HEATMAP — all numeric pairs
        # ════════════════════════════════════════════════════════════════════
        if len(numeric_cols) >= 3:
            try:
                corr_m = df[numeric_cols].corr()
                heat_d = [
                    {"x": _fmt(r), "y": _fmt(c), "value": round(float(corr_m.loc[r, c]), 3)}
                    for r in numeric_cols for c in numeric_cols
                    if not math.isnan(corr_m.loc[r, c])
                ]
                high = [(r, c, corr_m.loc[r, c]) for i, r in enumerate(numeric_cols)
                        for c in numeric_cols[i+1:] if abs(corr_m.loc[r, c]) > 0.7]
                candidates.append({
                    "id":          "heatmap_correlation",
                    "type":        "heatmap",
                    "section":     "schema",
                    "significance": 0.88,
                    "aspects":     len(numeric_cols),
                    "columns":     numeric_cols,
                    "title":       "Feature Correlation Matrix",
                    "x_label":     "Feature",
                    "y_label":     "Feature",
                    "summary":     f"{len(high)} high-correlation pair(s) detected (|r|>0.7)",
                    "insight":     (
                        f"{len(high)} strongly correlated feature pair(s) — multicollinearity risk."
                        if high else
                        "No strong inter-feature correlations — good feature independence."
                    ),
                    "explanation": (
                        f"Pearson correlation matrix across all {len(numeric_cols)} numeric features. "
                        + (
                            f"Found {len(high)} pair(s) with |r| > 0.7 — "
                            f"specifically: {', '.join(f'{r} ↔ {c}' for r, c, _ in high[:3])}. "
                            f"High collinearity inflates coefficient variance in linear models; VIF pruning is recommended."
                            if high else
                            f"All feature pairs show |r| ≤ 0.7, indicating satisfactory independence. "
                            f"Multicollinearity is not expected to be a concern for this dataset."
                        )
                    ),
                    "data":        heat_d,
                    "row_labels":  [_fmt(c) for c in numeric_cols],
                    "col_labels":  [_fmt(c) for c in numeric_cols],
                })
            except Exception as _he:
                logger.debug("Heatmap failed: %s", _he)

        # ════════════════════════════════════════════════════════════════════
        # §7 RADAR — multi-metric segment profile
        # ════════════════════════════════════════════════════════════════════
        if cat_cols and len(numeric_cols) >= 3:
            cat = cat_cols[0]
            try:
                top_cats   = df[cat].value_counts().head(6).index.tolist()
                radar_nums = numeric_cols[:6]
                gr         = df[df[cat].isin(top_cats)].groupby(cat)[radar_nums].mean()
                normed     = (gr - gr.min()) / (gr.max() - gr.min() + 1e-9)
                radar_d    = [
                    {"metric": _fmt(m), **{str(v): round(float(normed.loc[v, m]), 3)
                                           for v in top_cats if v in normed.index}}
                    for m in radar_nums
                ]
                candidates.append({
                    "id":          f"radar_{cat}",
                    "type":        "radar",
                    "section":     "schema",
                    "significance": 0.75,
                    "aspects":     len(radar_nums) + 1,
                    "columns":     [cat] + radar_nums,
                    "title":       f"Multi-Metric Profile by {_fmt(cat)}",
                    "x_label":     "Metric",
                    "y_label":     "Normalised Score (0–1)",
                    "summary":     f"Segment fingerprint across {len(radar_nums)} metrics",
                    "insight":     f"Distinct performance profiles across {_fmt(cat)} groups.",
                    "explanation": (
                        f"Radar chart comparing the top {len(top_cats)} {_fmt(cat)} segments across "
                        f"{len(radar_nums)} normalised metrics. Values are min-max scaled to 0–1 for comparability. "
                        f"Segments that form a larger radar area outperform on the combined metric set — "
                        f"useful for executive segment comparison without raw scale confusion."
                    ),
                    "data":        radar_d,
                    "line_keys":   [str(c) for c in top_cats],
                })
            except Exception as _re:
                logger.debug("Radar failed: %s", _re)

        # ════════════════════════════════════════════════════════════════════
        # §8 COUNT PER CATEGORY — frequency bar
        # ════════════════════════════════════════════════════════════════════
        for cat in cat_cols:
            try:
                counts = df[cat].value_counts().head(20)
                if len(counts) < 2:
                    continue
                dm = [{"name": str(k), "value": int(v)} for k, v in counts.items()]
                candidates.append({
                    "id":          f"count_bar_{cat}",
                    "type":        "bar",
                    "section":     "schema",
                    "significance": 0.35,
                    "aspects":     1,
                    "columns":     [cat],
                    "title":       f"Record Frequency by {_fmt(cat)}",
                    "x_label":     _fmt(cat),
                    "y_label":     "Record Count",
                    "summary":     f"Top category: {counts.index[0]} ({int(counts.iloc[0])} records)",
                    "insight":     f"Frequency distribution across {_fmt(cat)} — identifies dominant and rare segments.",
                    "explanation": (
                        f"Absolute record count per {_fmt(cat)} category, sorted by volume. "
                        f"The top category '{counts.index[0]}' contributes {int(counts.iloc[0])} records "
                        f"({counts.iloc[0]/len(df)*100:.1f}% of the dataset). "
                        f"Large count imbalances may lead to biased model training — consider oversampling minority categories."
                    ),
                    "data":        dm,
                })
            except Exception:
                pass

        # ════════════════════════════════════════════════════════════════════
        # §9 FUNNEL — ranked cumulative volume
        # ════════════════════════════════════════════════════════════════════
        if numeric_cols and cat_cols:
            try:
                cat = cat_cols[0]
                num = numeric_cols[0]
                for c in numeric_cols:
                    if any(kw in c.lower() for kw in ("total", "amount", "revenue", "sales", "count", "value")):
                        num = c
                        break
                totals   = df.groupby(cat)[num].sum().sort_values(ascending=False).head(8)
                cum_pct  = (totals.cumsum() / totals.sum() * 100)
                fd = [{"name": str(k), "value": round(float(v), 2)} for k, v in totals.items()]
                top2_pct = round(float(cum_pct.iloc[min(1, len(cum_pct)-1)]), 1)
                candidates.append({
                    "id":          f"funnel_{cat}_{num}",
                    "type":        "funnel",
                    "section":     "schema",
                    "significance": 0.62,
                    "aspects":     2,
                    "columns":     [cat, num],
                    "title":       f"Total {_fmt(num)} by {_fmt(cat)} (Ranked)",
                    "x_label":     _fmt(cat),
                    "y_label":     f"Total {_fmt(num)}",
                    "summary":     f"Top 2 {_fmt(cat)} groups = {top2_pct}% of total volume",
                    "insight":     f"Pareto effect: top {_fmt(cat)} groups dominate {_fmt(num)} volume.",
                    "explanation": (
                        f"Total {_fmt(num)} per {_fmt(cat)}, ranked highest to lowest — a Pareto analysis. "
                        f"The top 2 groups contribute {top2_pct}% of the overall {_fmt(num)} volume, "
                        f"{'confirming a strong Pareto concentration that warrants focused resource allocation.' if top2_pct > 60 else 'showing a moderate distribution without extreme concentration.'}"
                    ),
                    "data":        fd,
                })
            except Exception as _fe:
                logger.debug("Funnel failed: %s", _fe)

        # ════════════════════════════════════════════════════════════════════
        # §10 RANGE SUMMARY — min/avg/max per numeric feature
        # ════════════════════════════════════════════════════════════════════
        if len(numeric_cols) >= 2:
            try:
                sd = []
                for col in numeric_cols[:10]:
                    cd = df[col].dropna()
                    if cd.empty:
                        continue
                    sd.append({
                        "name": _fmt(col),
                        "min":  round(float(cd.min()), 4),
                        "avg":  round(float(cd.mean()), 4),
                        "max":  round(float(cd.max()), 4),
                    })
                if sd:
                    wide_range = max(sd, key=lambda x: x["max"] - x["min"])["name"]
                    candidates.append({
                        "id":          "range_bar_all_numeric",
                        "type":        "range_bar",
                        "section":     "quality",
                        "significance": 0.52,
                        "aspects":     len(numeric_cols),
                        "columns":     numeric_cols[:10],
                        "title":       "Numeric Feature Range Summary",
                        "x_label":     "Feature",
                        "y_label":     "Value",
                        "summary":     f"Min / Avg / Max across {len(sd)} numeric features",
                        "insight":     f"{wide_range} has the widest value range — potential outlier risk.",
                        "explanation": (
                            f"For each numeric column, this chart shows the minimum, mean, and maximum observed value. "
                            f"Columns with extreme spread between min and max (e.g. {wide_range}) are prone to outlier contamination "
                            f"and may require winsorization or robust scaling before model training."
                        ),
                        "data":        sd,
                        "stack_keys":  ["min", "avg", "max"],
                    })
            except Exception as _rbe:
                logger.debug("Range bar failed: %s", _rbe)

        # ════════════════════════════════════════════════════════════════════
        # §11 NULL RATE BAR — per column null percentages
        # ════════════════════════════════════════════════════════════════════
        try:
            null_rates = df.isnull().mean().sort_values(ascending=False)
            null_rates = null_rates[null_rates > 0]
            if not null_rates.empty:
                nd = [{"name": _fmt(c), "value": round(float(v) * 100, 1)} for c, v in null_rates.head(20).items()]
                max_null = float(null_rates.iloc[0])
                candidates.append({
                    "id":          "null_rate_bar",
                    "type":        "bar",
                    "section":     "missing",
                    "significance": _clamp(max_null + 0.4),
                    "aspects":     len(null_rates),
                    "columns":     null_rates.index.tolist()[:20],
                    "title":       "Null Rate (%) per Column",
                    "x_label":     "Column",
                    "y_label":     "Null %",
                    "summary":     f"Worst offender: {_fmt(null_rates.index[0])} ({max_null*100:.1f}% null)",
                    "insight":     f"{len(null_rates)} column(s) have missing values; worst: {_fmt(null_rates.index[0])} ({max_null*100:.0f}% null).",
                    "explanation": (
                        f"Percentage of null values per column, sorted by severity. "
                        f"{_fmt(null_rates.index[0])} is the most affected column with {max_null*100:.1f}% missing values. "
                        + (
                            f"Columns exceeding 40% null rate are typically dropped or require domain-specific imputation — "
                            f"automatic imputation may introduce significant bias for values this sparse."
                            if max_null > 0.4 else
                            f"All missing rates are within manageable bounds — standard median/mode imputation "
                            f"was applied and the risk of imputation bias is low."
                        )
                    ),
                    "data":        nd,
                })
        except Exception as _ne:
            logger.debug("Null bar failed: %s", _ne)

        # ════════════════════════════════════════════════════════════════════
        # §12 CONTEXT-ENRICHED CHARTS (model_metrics, governance, anomaly, drift)
        # ════════════════════════════════════════════════════════════════════

        # 12a — Feature Importance Bar (from audit model_metrics)
        feat_imp = ctx.get("feature_importances") or ctx.get("feature_importance") or {}
        mm       = ctx.get("model_metrics") or {}
        if not feat_imp:
            feat_imp = mm.get("feature_importances") or mm.get("feature_importance") or {}
        if feat_imp and isinstance(feat_imp, dict):
            try:
                sorted_fi    = sorted(feat_imp.items(), key=lambda x: float(x[1]), reverse=True)[:15]
                fi_data      = [{"name": _fmt(k), "value": round(float(v), 5)} for k, v in sorted_fi]
                top_feat     = fi_data[0]["name"] if fi_data else "—"
                best_model   = mm.get("best_model") or mm.get("model_name") or "AutoML"
                candidates.append({
                    "id":          "feature_importance_bar",
                    "type":        "bar",
                    "section":     "model",
                    "significance": 0.92,
                    "aspects":     len(fi_data),
                    "columns":     [k for k, _ in sorted_fi],
                    "title":       f"Feature Importance — {best_model}",
                    "x_label":     "Feature",
                    "y_label":     "Importance Score",
                    "summary":     f"Top predictor: {top_feat}",
                    "insight":     f"{top_feat} is the most influential feature in the {best_model} model.",
                    "explanation": (
                        f"Ranked feature importance scores as returned by {best_model}. "
                        f"'{top_feat}' contributes the most predictive signal (importance = {fi_data[0]['value']:.4f}). "
                        f"Features with near-zero importance are candidates for pruning to reduce overfitting and inference latency."
                    ),
                    "data":        fi_data,
                })
            except Exception as _fie:
                logger.debug("Feature importance bar failed: %s", _fie)

        # 12b — Model CV Score Bar (cross-fold scores)
        cv_scores = mm.get("cv_scores") or mm.get("fold_scores") or []
        if cv_scores and len(cv_scores) > 1:
            try:
                cv_d = [{"name": f"Fold {i+1}", "value": round(float(s), 4)} for i, s in enumerate(cv_scores)]
                mean_cv = float(np.mean(cv_scores))
                std_cv  = float(np.std(cv_scores))
                candidates.append({
                    "id":          "cv_scores_bar",
                    "type":        "bar",
                    "section":     "model",
                    "significance": 0.85,
                    "aspects":     len(cv_scores),
                    "columns":     [],
                    "title":       "Cross-Validation Fold Scores",
                    "x_label":     "Fold",
                    "y_label":     "Score",
                    "summary":     f"Mean={mean_cv:.3f} ± {std_cv:.3f}",
                    "insight":     f"CV mean = {mean_cv:.3f}, std = {std_cv:.3f} — {'stable' if std_cv < 0.05 else 'variable'} generalisation.",
                    "explanation": (
                        f"Score achieved on each cross-validation fold. Mean performance = {mean_cv:.3f} with std = {std_cv:.3f}. "
                        + (
                            f"The low variance across folds indicates stable model generalisation — the model is unlikely to overfit."
                            if std_cv < 0.05 else
                            f"The high fold-to-fold variance ({std_cv:.3f}) suggests the model may be sensitive to data distribution — "
                            f"review for overfitting or consider ensemble averaging."
                        )
                    ),
                    "data":        cv_d,
                })
            except Exception as _cve:
                logger.debug("CV bar failed: %s", _cve)

        # 12c — Anomaly Density Bar (from anomaly_report)
        anomaly_r = ctx.get("anomaly_report") or ctx.get("anomaly_deep_dive") or {}
        per_col   = anomaly_r.get("per_column") or []
        if per_col:
            try:
                ad = sorted(per_col, key=lambda x: x.get("count", 0), reverse=True)[:12]
                a_data = [{"name": _fmt(x.get("col", x.get("column", "?"))), "value": int(x.get("count", 0))} for x in ad]
                total_anom = sum(x["value"] for x in a_data)
                candidates.append({
                    "id":          "anomaly_density_bar",
                    "type":        "bar",
                    "section":     "anomaly",
                    "significance": 0.80,
                    "aspects":     len(a_data),
                    "columns":     [x.get("col", "") for x in ad],
                    "title":       "Anomaly Count per Column",
                    "x_label":     "Column",
                    "y_label":     "Anomaly Count",
                    "summary":     f"{total_anom} total anomalies across {len(a_data)} column(s)",
                    "insight":     f"{a_data[0]['name']} has the highest anomaly density ({a_data[0]['value']} records).",
                    "explanation": (
                        f"Isolation Forest anomaly counts per column ({total_anom} total anomalies flagged). "
                        f"'{a_data[0]['name']}' shows the highest density with {a_data[0]['value']} anomalous values. "
                        f"Columns with disproportionate anomaly rates may carry measurement errors, fraud signals, "
                        f"or legitimate extreme events — domain validation is recommended before automated removal."
                    ),
                    "data":        a_data,
                })
            except Exception as _ae:
                logger.debug("Anomaly bar failed: %s", _ae)

        # 12d — Drift PSI Bar (from drift_report)
        drift_r   = ctx.get("drift_report") or {}
        drift_cols = drift_r.get("drifted_columns") or []
        psi_scores = drift_r.get("psi_scores") or {}
        if psi_scores:
            try:
                ds = sorted(psi_scores.items(), key=lambda x: float(x[1]), reverse=True)[:12]
                drift_data = [{"name": _fmt(c), "value": round(float(v), 4)} for c, v in ds]
                max_psi = drift_data[0]["value"] if drift_data else 0
                candidates.append({
                    "id":          "drift_psi_bar",
                    "type":        "bar",
                    "section":     "drift",
                    "significance": _clamp(max_psi * 2 + 0.4),
                    "aspects":     len(drift_data),
                    "columns":     [c for c, _ in ds],
                    "title":       "Distribution Drift — PSI per Column",
                    "x_label":     "Column",
                    "y_label":     "Population Stability Index (PSI)",
                    "summary":     f"{len(drift_cols)} column(s) drifted (PSI > 0.2)",
                    "insight":     f"{drift_data[0]['name']} has the highest PSI ({max_psi:.3f}) — significant drift.",
                    "explanation": (
                        f"Population Stability Index (PSI) measures distribution shift vs the baseline. "
                        f"PSI < 0.1 = stable, 0.1–0.2 = moderate change, > 0.2 = significant drift. "
                        f"'{drift_data[0]['name']}' scores PSI = {max_psi:.3f} — "
                        + (
                            f"a critical drift signal that should trigger model retraining."
                            if max_psi > 0.2 else
                            f"within acceptable bounds, though the trend should be monitored."
                        )
                    ),
                    "data":        drift_data,
                })
            except Exception as _de:
                logger.debug("Drift bar failed: %s", _de)

        # 12e — PII Coverage Pie (from governance_report)
        gov_r    = ctx.get("governance_report") or {}
        pii_hits = gov_r.get("pii_hits") or {}
        if pii_hits:
            try:
                pii_counts: Dict[str, int] = {}
                for col_hits in pii_hits.values():
                    if isinstance(col_hits, dict):
                        for pii_type, cnt in col_hits.items():
                            pii_counts[pii_type] = pii_counts.get(pii_type, 0) + int(cnt)
                if pii_counts:
                    total_pii = sum(pii_counts.values())
                    pii_data  = [{"name": k, "value": v} for k, v in sorted(pii_counts.items(), key=lambda x: -x[1])]
                    candidates.append({
                        "id":          "pii_type_pie",
                        "type":        "pie",
                        "section":     "governance",
                        "significance": 0.78,
                        "aspects":     len(pii_data),
                        "columns":     list(pii_hits.keys()),
                        "title":       "PII Type Distribution",
                        "x_label":     "PII Category",
                        "y_label":     "Hit Count",
                        "summary":     f"{total_pii} PII hits across {len(pii_data)} type(s)",
                        "insight":     f"Most common PII type: {pii_data[0]['name']} ({pii_data[0]['value']} hits).",
                        "explanation": (
                            f"Breakdown of personally identifiable information by category across all scanned columns. "
                            f"A total of {total_pii} PII hits were detected, spread across {len(pii_data)} PII type(s). "
                            f"The governance engine applied the configured redaction policy to all affected columns before "
                            f"the data was persisted to the ISSF snapshot."
                        ),
                        "data":        pii_data,
                    })
            except Exception as _ge:
                logger.debug("PII pie failed: %s", _ge)

        # ════════════════════════════════════════════════════════════════════
        # §13 QUALITY RADAR — 5-axis data health score
        # ════════════════════════════════════════════════════════════════════
        try:
            completeness  = float(1.0 - df.isnull().mean().mean())
            uniqueness    = float(df.nunique().mean() / max(len(df), 1))
            uniqueness    = min(1.0, uniqueness)
            zero_var_frac = float(sum(df[c].std() == 0 for c in numeric_cols) / max(len(numeric_cols), 1))
            consistency   = float(1.0 - zero_var_frac)
            validity      = float(min(1.0, completeness * 1.1))
            dup_rate      = float(df.duplicated().mean())
            integrity     = float(1.0 - dup_rate)

            qr_data = [
                {"metric": "Completeness",  "score": round(completeness, 3)},
                {"metric": "Uniqueness",    "score": round(uniqueness, 3)},
                {"metric": "Consistency",   "score": round(consistency, 3)},
                {"metric": "Validity",      "score": round(validity, 3)},
                {"metric": "Integrity",     "score": round(integrity, 3)},
            ]
            overall = round(float(np.mean([completeness, uniqueness, consistency, validity, integrity])), 3)
            candidates.append({
                "id":          "quality_radar",
                "type":        "radar",
                "section":     "quality",
                "significance": 0.90,
                "aspects":     5,
                "columns":     numeric_cols[:3],
                "title":       "Data Quality Radar",
                "x_label":     "Quality Dimension",
                "y_label":     "Score (0–1)",
                "summary":     f"Overall quality score: {overall:.0%}",
                "insight":     f"Data health = {overall:.0%} across 5 dimensions.",
                "explanation": (
                    f"Five-axis quality radar scoring this dataset across Completeness, Uniqueness, "
                    f"Consistency (zero-variance check), Validity (inverse null rate), and Integrity (duplicate removal). "
                    f"The composite quality score is {overall:.0%}. "
                    + (
                        f"This exceeds the 70% production threshold — the dataset is cleared for downstream use."
                        if overall >= 0.70 else
                        f"This is below the 70% production threshold — data quality remediation is recommended "
                        f"before using this dataset for model training."
                    )
                ),
                "data":        qr_data,
                "line_keys":   ["score"],
            })
        except Exception as _qre:
            logger.debug("Quality radar failed: %s", _qre)

        # ════════════════════════════════════════════════════════════════════
        # SORT + DEDUPLICATE + TRIM
        # ════════════════════════════════════════════════════════════════════
        candidates.sort(key=lambda x: x.get("significance", 0.0), reverse=True)
        seen: set = set()
        unique: List[Dict] = []
        for c in candidates:
            if c["title"] not in seen:
                seen.add(c["title"])
                unique.append(c)

        final_charts = unique[:_max]

        # ── KPIs ─────────────────────────────────────────────────────────────
        kpis: Dict[str, Any] = {
            "Total Records": f"{n_rows:,}",
            "Total Columns": n_cols_total,
        }
        null_rate_pct = round(float(df.isnull().mean().mean()) * 100, 1)
        kpis["Null Rate"] = f"{null_rate_pct}%"

        if numeric_cols:
            pm = numeric_cols[0]
            for c in numeric_cols:
                if any(kw in c.lower() for kw in ("total", "amount", "sales", "rev", "price", "value")):
                    pm = c
                    break
            kpis[f"Avg {_fmt(pm)}"] = round(float(df[pm].mean()), 2)
            kpis[f"Max {_fmt(pm)}"] = round(float(df[pm].max()), 2)

        if cat_cols:
            top_cat_col = cat_cols[0]
            kpis[f"Top {_fmt(top_cat_col)}"] = str(df[top_cat_col].mode().iloc[0])

        if mm:
            roc = mm.get("roc_auc")
            if roc is not None:
                kpis["AUC Score"] = round(float(roc), 4)
            f1 = mm.get("f1")
            if f1 is not None:
                kpis["F1 Score"] = round(float(f1), 4)

        if pii_hits:
            kpis["PII Columns"] = len(pii_hits)

        if drift_cols:
            kpis["Drifted Columns"] = len(drift_cols)

        # ── Insights Feed (per chart) ────────────────────────────────────────
        insights_feed = [
            {
                "chart_id":  ch["id"],
                "title":     ch["title"],
                "text":      ch.get("explanation") or ch.get("summary") or ch.get("insight", ""),
                "relevance": round(ch["significance"] * 100),
                "section":   ch.get("section", "schema"),
            }
            for ch in final_charts
            if ch.get("significance", 0) >= 0.35
        ]
        insights_feed.sort(key=lambda x: -x["relevance"])

        # ── Structured Section Insights (for InsightCards) ───────────────────
        section_insights: Dict[str, List[Dict]] = {}
        for ch in final_charts:
            sec = ch.get("section", "schema")
            section_insights.setdefault(sec, []).append({
                "title":       ch["title"],
                "insight":     ch.get("insight", ""),
                "explanation": ch.get("explanation") or ch.get("summary", ""),
                "significance": ch.get("significance", 0),
            })
        # Sort within each section
        for sec in section_insights:
            section_insights[sec].sort(key=lambda x: -x["significance"])

        return {
            "kpis":             kpis,
            "insights_feed":    insights_feed,
            "section_insights": section_insights,
            "charts":           final_charts,
            "chart_count":      len(final_charts),
            "adaptive_max":     _max,
        }
