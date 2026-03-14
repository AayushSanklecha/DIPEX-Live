import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from analytics.orchestrator import AnalyticsOrchestrator

def create_messy_dataset():
    np.random.seed(42)
    n = 500
    
    # Normal distribution
    age = np.random.normal(40, 10, n)
    
    # Highly skewed
    income = np.random.lognormal(10, 2, n)
    
    # Categorical
    department = np.random.choice(["Engineering", "Sales", "HR", "Marketing"], n)
    
    # Target (with some correlation)
    target = (age * 0.5) + (np.log1p(income) * 10) + np.random.normal(0, 5, n)
    
    df = pd.DataFrame({
        "age": age,
        "income": income,
        "department": department,
        "constant_col": "all_same", # Should be dropped
        "high_null_col": np.random.randn(n) # Will make 40% null below
    })
    
    # Add target column (Binary class target)
    df["target"] = (target > np.median(target)).astype(int) 
    
    # Add missingness
    # > 20% missing for ML imputation
    df.loc[np.random.choice(n, int(n * 0.40), replace=False), "high_null_col"] = np.nan
    
    # < 20% missing for Median imputation
    df.loc[np.random.choice(n, int(n * 0.10), replace=False), "age"] = np.nan
    df.loc[np.random.choice(n, int(n * 0.10), replace=False), "department"] = np.nan
    
    # Zeros (e.g. 35% zeros to trigger zero-to-NaN-to-Median logic)
    zero_idx = np.random.choice(n, int(n * 0.35), replace=False)
    df.loc[zero_idx, "income"] = 0
    
    return df

def test_pipeline():
    print("Generating messy test dataset...")
    df = create_messy_dataset()
    
    # Save for reference
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/messy_test_data.csv", index=False)
    
    print("\nInitialize Analytics Orchestrator...")
    orchestrator = AnalyticsOrchestrator()
    
    print("\nRunning full pipeline...")
    df_loaded = pd.read_csv("data/messy_test_data.csv")
    result = orchestrator.run(df_loaded, target_col="target")
    print("\nGenerating Executive Report...")
    from reporting_service.executive_report import ExecutiveReportGenerator
    reporter = ExecutiveReportGenerator()
    report_path = reporter.generate(
        run_id=result.run_id,
        confidence_vector={"confidence_score": 0.85},
        gate1_decision="PASS",
        gate2_decision="PASS",
        narrative=result.llm_summary,
        actions_log=getattr(result, "actions_log", {}),
        eda_report=result.eda_report,
        row_count=len(df_loaded),
        col_count=len(df_loaded.columns)
    )
    print(f"\nReport generated at: {report_path}")
    
    print("\n=== PIPELINE EXECUTION COMPLETE ===")
    if hasattr(result, "actions_log"):
        print("\n=== ACTIONS APPLIED BY AUTOCORRECTOR ===")
        for col, info in result.actions_log.items():
            print(f"- {col}: {info.get('action')} ({info.get('reason')})")

if __name__ == "__main__":
    test_pipeline()
