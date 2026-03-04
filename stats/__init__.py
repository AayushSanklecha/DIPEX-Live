"""stats package — enterprise statistical analytics"""
from stats.descriptive import DescriptiveStats
from stats.hypothesis_tests import HypothesisTester
from stats.regression import RegressionEngine
from stats.time_series import TimeSeriesAnalyzer
from stats.correlation import CorrelationAnalyzer
from stats.permutation_tests import PermutationTester, MultipleTestingCorrector
from stats.residual_diagnostics import ResidualDiagnostics
from stats.drift_detection import DriftDetector

__all__ = [
    "DescriptiveStats", "HypothesisTester", "RegressionEngine", "TimeSeriesAnalyzer",
    "CorrelationAnalyzer", "PermutationTester", "MultipleTestingCorrector",
    "ResidualDiagnostics", "DriftDetector",
]
