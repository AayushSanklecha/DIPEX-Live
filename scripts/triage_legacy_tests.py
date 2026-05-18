# scripts/triage_legacy_tests.py
"""
Issue 09: Legacy test triage script.

Categorises all legacy test failures into:
  - STALE:   test references removed components (should delete or skip)
  - REGRESS: test was passing before a recent commit (needs investigation)
  - FLAKY:   test passes sometimes (isolate and fix)

Usage: python scripts/triage_legacy_tests.py
"""

import re
import subprocess
import sys


LEGACY_TEST_FILES = [
    "tests/legacy/test_analyst_intelligence.py",
    "tests/legacy/test_rl_enhancements.py",
    "tests/legacy/test_cognitive_engine.py",
    "tests/legacy/test_security.py",
    "tests/legacy/test_verifier_stubs.py",
    "tests/legacy/test_ml_components.py",
    "tests/legacy/test_property_based.py",
    "tests/legacy/test_rl_agents.py",
    "tests/legacy/test_insight_narrator.py",
    "tests/legacy/test_chaos.py",
    "tests/legacy/test_hard_gate.py",
    "tests/legacy/test_new_connectors.py",
    "tests/legacy/test_llm_governance.py",
    "tests/legacy/test_canary_datasets.py",
]

# Keywords that indicate a test references a removed component
STALE_KEYWORDS = [
    "RLUpdater", "CognitiveEngine", "AnalystBrain_v1",
    "RLOptimizer", "deprecated", "removed",
]

SECURITY_FILES = ["tests/legacy/test_security.py"]


def run_test_file(test_file: str) -> tuple[int, int, int, list[str]]:
    """Run a test file and return (passed, failed, errors, failure_lines)."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_file, "--tb=line", "-q"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = result.stdout + result.stderr

    # Parse summary line: "X failed, Y passed, Z errors..."
    passed = failed = errors = 0
    m = re.search(r"(\d+) passed", output)
    if m:
        passed = int(m.group(1))
    m = re.search(r"(\d+) failed", output)
    if m:
        failed = int(m.group(1))
    m = re.search(r"(\d+) error", output)
    if m:
        errors = int(m.group(1))

    # Collect FAILED lines
    failure_lines = [
        line.strip()
        for line in output.split("\n")
        if line.strip().startswith("FAILED")
    ]

    return passed, failed, errors, failure_lines


def classify_file(test_file: str) -> str:
    """Classify a file as STALE, SECURITY, or NEEDS_REVIEW."""
    if test_file in SECURITY_FILES:
        return "🔴 SECURITY — INVESTIGATE (do NOT skip)"
    # Check if file references removed components
    try:
        with open(test_file, encoding="utf-8") as f:
            content = f.read()
        for kw in STALE_KEYWORDS:
            if kw in content:
                return f"⚠️  STALE — references removed component: {kw}"
    except FileNotFoundError:
        return "❌ FILE NOT FOUND"
    return "🔍 NEEDS_REVIEW"


def main() -> None:
    print("=" * 70)
    print("DIPEX Legacy Test Triage Report")
    print("=" * 70)

    total_passed = total_failed = total_errors = 0

    for test_file in LEGACY_TEST_FILES:
        classification = classify_file(test_file)
        try:
            passed, failed, errors, failures = run_test_file(test_file)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"\n{'─' * 60}")
            print(f"FILE: {test_file}")
            print(f"STATUS: COULD NOT RUN — {e}")
            continue

        total_passed += passed
        total_failed += failed
        total_errors += errors

        print(f"\n{'─' * 60}")
        print(f"FILE: {test_file}")
        print(f"RESULT: {passed} passed, {failed} failed, {errors} errors")
        print(f"ACTION: {classification}")
        if failures:
            print("SAMPLES:")
            for f in failures[:3]:
                print(f"  → {f}")

    print(f"\n{'=' * 70}")
    print(f"TOTALS: {total_passed} passed, {total_failed} failed, {total_errors} errors")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
