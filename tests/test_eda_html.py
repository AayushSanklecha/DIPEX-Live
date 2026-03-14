import os
import pandas as pd
import pytest
from eda.auto_eda import AutoEDA

@pytest.fixture
def mock_eda_data():
    return pd.DataFrame({
        "age": [25, 30, 35, 40, None, 50, 60],
        "income": [50000, 60000, 75000, 100000, 120000, 0, 90000],
        "category": ["A", "B", "A", "C", "A", "B", "A"],
        "target": [0, 1, 0, 1, 1, 0, 1]
    })

def test_auto_eda_html_report_generation(mock_eda_data):
    eda = AutoEDA()
    
    # Run the EDA module
    report = eda.run(mock_eda_data, run_id="test_eda_html")
    
    # Assertions
    assert report.html_report_path is not None, "HTML report path should not be None."
    assert report.html_report_path.endswith(".html"), "Path should end with .html"
    assert "reports_output" in report.html_report_path
    
    # Verify the file actually exists on disk
    assert os.path.exists(report.html_report_path), f"File {report.html_report_path} was not created!"
    
    # Cleanup
    if os.path.exists(report.html_report_path):
        os.remove(report.html_report_path)
