"""
reporting_service/intelligence_engine.py
----------------------------------------
Next-Gen Combinatorial Analytics Engine.

Instead of generating a generic chart for every column, this engine:
1. Iterates over combinatorial aspects (Num vs Num, Cat vs Num, Time vs Num).
2. Scores the significance of the relationship (Correlation, Variance, Trend).
3. Selects the best Chart Type using ChartRelevanceScorer.
4. Returns the Top 10+ most significant charts, dropping the noise to the bottom.
5. Generates data-journalism style insights.
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from reporting_service.chart_relevance_scorer import ChartRelevanceScorer

logger = logging.getLogger("dipex.reporting.intelligence")


class IntelligenceEngine:
    def __init__(self) -> None:
        self.scorer = ChartRelevanceScorer()

    def analyze_dataset(self, df: pd.DataFrame, max_charts: int = 20) -> Dict[str, Any]:
        """Runs the combinatorial analysis and returns prioritized charts and insights."""
        if df.empty:
            return {"kpis": {}, "insights_feed": [], "charts": []}

        # Avoid mutating the original dataframe
        df = df.copy()

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

        # Attempt to cast object cols that look like dates 
        for col in df.select_dtypes(include=["object"]):
            if "date" in col.lower() or "time" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                    datetime_cols.append(col)
                except Exception:
                    pass

        # Categorical columns (objects/category that aren't times, with cardinality < 50)
        cat_cols = []
        for col in df.columns:
            if col not in numeric_cols and col not in datetime_cols:
                if 1 < df[col].nunique() < 50:
                    cat_cols.append(col)

        candidates = []

        # ── 1. Numeric vs Numeric (Scatter plots) ──
        for i, num1 in enumerate(numeric_cols):
            for num2 in numeric_cols[i + 1 :]:
                corr = df[num1].corr(df[num2])
                if pd.isna(corr):
                    corr = 0.0
                score = min(abs(corr) + 0.1, 1.0) # Bonus for continuous vs continuous
                
                # We need safe data sampling for JSON
                sample_df = df[[num1, num2]].dropna().head(200)
                
                candidates.append({
                    "id": f"scatter_{num1}_{num2}",
                    "type": "scatter",
                    "significance": float(score),
                    "aspects": 2,
                    "columns": [num1, num2],
                    "title": f"Correlation: {num1} vs {num2}",
                    "insight": f"Strong correlation ({corr:.2f}) between {num1} and {num2}." if score > 0.6 else f"Weak relationship between {num1} and {num2}.",
                    "data": sample_df.to_dict(orient="records"),
                })

        # ── 2. Categorical vs Numeric (Bar / Stacked Bar) ──
        for cat in cat_cols:
            for num in numeric_cols:
                grouped = df.groupby(cat)[num].mean()
                if len(grouped) > 1 and grouped.mean() != 0:
                    cv = grouped.std() / abs(grouped.mean())
                    score = min(float(cv), 1.0)
                    
                    sorted_grp = grouped.sort_values(ascending=False)
                    best_cat = str(sorted_grp.index[0])
                    worst_cat = str(sorted_grp.index[-1])
                    
                    data_mapped = [{"name": str(idx), "value": float(val)} for idx, val in sorted_grp.items()]

                    other_cats = [c for c in cat_cols if c != cat and df[c].nunique() < 10]
                    if other_cats and score > 0.3:
                        stack_cat = other_cats[0]
                        stacked_grp = df.groupby([cat, stack_cat])[num].mean().unstack(fill_value=0)
                        
                        stacked_data = []
                        for row_idx, row in stacked_grp.iterrows():
                            record = {"name": str(row_idx)}
                            for col_idx, val in row.items():
                                record[str(col_idx)] = float(val)
                            stacked_data.append(record)

                        candidates.append({
                            "id": f"stacked_bar_{cat}_{stack_cat}_{num}",
                            "type": "stacked_bar",
                            "significance": float(score + 0.15), # Huge bonus for multivariate
                            "aspects": 3,
                            "columns": [cat, stack_cat, num],
                            "title": f"Average {num} by {cat} & {stack_cat}",
                            "insight": f"{best_cat} drives the highest {num}, segmented heavily by {stack_cat}.",
                            "data": stacked_data[:20],
                            "stack_keys": [str(c) for c in stacked_grp.columns],
                        })
                    else:
                        candidates.append({
                            "id": f"bar_{cat}_{num}",
                            "type": "bar",
                            "significance": float(score),
                            "aspects": 2,
                            "columns": [cat, num],
                            "title": f"Average {num} by {cat}",
                            "insight": f"{best_cat} dominates {num}, while {worst_cat} lags behind significantly." if score > 0.4 else f"Uniform {num} distribution across {cat}.",
                            "data": data_mapped[:20], # Top 20 bars
                        })

        # ── 3. Time Series (Line Chart / Multi-line) ──
        for dt_col in datetime_cols:
            for num in numeric_cols:
                time_df = df.dropna(subset=[dt_col, num]).copy()
                if time_df.empty:
                    continue
                
                time_df[dt_col] = pd.to_datetime(time_df[dt_col], errors='coerce')
                time_df = time_df.dropna(subset=[dt_col])
                time_df.set_index(dt_col, inplace=True)
                time_df.sort_index(inplace=True)
                
                try:
                    resampled = time_df[num].resample('ME').sum()
                    if resampled.empty or resampled.mean() == 0:
                        continue
                        
                    trend_score = min(float(resampled.std() / abs(resampled.mean())), 1.0)
                    
                    data_mapped = [{"time": str(idx.date()), "value": float(val)} for idx, val in resampled.items()]
                    
                    if cat_cols and trend_score > 0.2:
                        split_cat = cat_cols[0]
                        # Group by time and category
                        pivot = time_df.groupby([pd.Grouper(freq="ME"), split_cat])[num].sum().unstack(fill_value=0)
                        
                        multi_data = []
                        for row_idx, row in pivot.iterrows():
                            record = {"time": str(row_idx.date())}
                            for col_idx, val in row.items():
                                record[str(col_idx)] = float(val)
                            multi_data.append(record)
                            
                        candidates.append({
                            "id": f"multi_line_{dt_col}_{split_cat}_{num}",
                            "type": "multi_line",
                            "significance": float(trend_score + 0.2), # Bonus for temporal multivariate
                            "aspects": 3,
                            "columns": [dt_col, split_cat, num],
                            "title": f"{num} Trend by {split_cat}",
                            "insight": f"Significant seasonal volatility in {num}, heavily influenced by {split_cat}.",
                            "data": multi_data[-50:], # Last 50 periods
                            "line_keys": [str(c) for c in pivot.columns],
                        })
                    else:
                        candidates.append({
                            "id": f"line_{dt_col}_{num}",
                            "type": "line",
                            "significance": float(trend_score + 0.1),
                            "aspects": 2,
                            "columns": [dt_col, num],
                            "title": f"Trend: {num} over Time",
                            "insight": f"High trend volatility detected in {num}." if trend_score > 0.5 else f"Stable {num} over time.",
                            "data": data_mapped[-50:],
                        })
                except Exception as e:
                    logger.debug(f"Failed to generate time series: {e}")

        # ── 4. Pure Categorical (Pie/Donut) ──
        for cat in cat_cols:
            counts = df[cat].value_counts(normalize=True)
            if counts.empty:
                continue
            
            imbalance = float(counts.iloc[0])
            score = float(imbalance) if imbalance > 0.2 else 0.1
            
            # Map for Recharts Pie
            data_mapped = []
            for i, (idx, val) in enumerate(counts.items()):
                if i < 5:
                    data_mapped.append({"name": str(idx), "value": float(val)})
            
            if len(counts) > 5:
                other_sum = float(counts.iloc[5:].sum())
                data_mapped.append({"name": "Other", "value": other_sum})
                
            top_cat = str(counts.index[0])
            
            candidates.append({
                "id": f"pie_{cat}",
                "type": "pie",
                "significance": score - 0.2, # Punish pure pie charts
                "aspects": 1,
                "columns": [cat],
                "title": f"Distribution of {cat}",
                "insight": f"{top_cat} makes up a massive {imbalance*100:.0f}% of all {cat} records." if imbalance > 0.4 else f"{cat} is relatively evenly distributed.",
                "data": data_mapped,
            })

        # ── 5. Numeric Distribution (Area / Histogram) ──
        for num in numeric_cols:
            skew = float(df[num].skew())
            if pd.isna(skew): skew = 0.0
            score = min(abs(skew) / 5.0, 1.0)
            
            # Create rough histogram bins
            counts, bins = np.histogram(df[num].dropna(), bins=10)
            data_mapped = [{"bin": f"{bins[i]:.1f}-{bins[i+1]:.1f}", "count": int(c)} for i, c in enumerate(counts)]
            
            candidates.append({
                "id": f"area_{num}",
                "type": "area",
                "significance": float(score) - 0.3, # Demote simple 1D distributions
                "aspects": 1,
                "columns": [num],
                "title": f"Distribution curve: {num}",
                "insight": f"Highly skewed distribution curve for {num}." if score > 0.6 else f"Normal 'bell-curve' distribution for {num}.",
                "data": data_mapped,
            })

        # Sort all discovered charts by mathematical significance
        candidates.sort(key=lambda x: x["significance"], reverse=True)
        
        # Enforce uniqueness of titles
        seen_titles = set()
        unique_candidates = []
        for c in candidates:
            if c["title"] not in seen_titles:
                seen_titles.add(c["title"])
                unique_candidates.append(c)
                
        final_charts = unique_candidates[:max_charts]

        # ── Top Level KPIs ──
        kpis = {
            "Total Records": int(len(df)),
            "Total Columns": int(len(df.columns)),
        }
        if numeric_cols:
            prime_metric = numeric_cols[0]
            for c in numeric_cols:
                if 'total' in c.lower() or 'amount' in c.lower() or 'sales' in c.lower() or 'rev' in c.lower() or 'price' in c.lower():
                    prime_metric = c
                    break
            kpis[f"Avg {prime_metric}"] = round(float(df[prime_metric].mean()), 2)
            kpis[f"Max {prime_metric}"] = round(float(df[prime_metric].max()), 2)

        # ── Data Journalism Insights Feed ──
        insights_feed = []
        for chart in final_charts:
            # Only push the highly significant ones to the feed
            if chart["significance"] >= 0.35:
                insights_feed.append({
                    "chart_id": chart["id"],
                    "title": chart["title"],
                    "text": chart["insight"],
                    "relevance": round(chart["significance"] * 100)
                })

        return {
            "kpis": kpis,
            "insights_feed": insights_feed,
            "charts": final_charts
        }
