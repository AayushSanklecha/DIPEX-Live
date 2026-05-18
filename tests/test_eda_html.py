# tests/test_eda_html.py
"""
EDA HTML report tests — Sprint 3, Issue H2.
Verifies that AutoEDA always produces a real HTML file (never returns None).
"""
import os
import pandas as pd
import pytest
from pathlib import Path
from eda.auto_eda import AutoEDA


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [25, 30, 35, 40, None, 50, 60],
        "income": [50000, 60000, 75000, 100000, 120000, 0, 90000],
        "category": ["A", "B", "A", "C", "A", "B", "A"],
        "target": [0, 1, 0, 1, 1, 0, 1]
    })


def test_html_report_returns_path_not_none(sample_df, tmp_path):
    """EDA HTML report must return a file path, never None."""
    eda = AutoEDA()
    report = eda.run(sample_df, run_id="test_eda_html")

    assert report.html_report_path is not None, (
        "HTML report path is None. Check _generate_html_report() "
        "for silent exception catches. The fallback must "
        "write an HTML file and return its path."
    )
    assert Path(report.html_report_path).exists(), (
        f"Reported path {report.html_report_path} does not exist on disk."
    )
    assert report.html_report_path.endswith(".html"), "Report file must be .html"

    # Cleanup
    if os.path.exists(report.html_report_path):
        os.remove(report.html_report_path)


def test_html_report_file_has_content(sample_df, tmp_path):
    """HTML file must contain actual content, not be empty."""
    eda = AutoEDA()
    report = eda.run(sample_df, run_id="test_eda_content")

    assert report.html_report_path is not None, "No HTML path returned"
    content = Path(report.html_report_path).read_text(encoding="utf-8")
    assert len(content) > 100, "HTML report file is suspiciously small."
    assert "<html" in content.lower() or "<!doctype" in content.lower()

    # Cleanup
    if os.path.exists(report.html_report_path):
        os.remove(report.html_report_path)
