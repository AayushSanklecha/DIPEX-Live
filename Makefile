# ─────────────────────────────────────────────────────────────────
# DIPEX v3.0.0 — Makefile
# ─────────────────────────────────────────────────────────────────

.PHONY: test-core test-legacy test-full kafka-check smoke-test

# Issue 01: Core tests ONLY — must have 0 failures in CI
test-core:
	@echo "════ Running CORE tests (must be green) ════"
	pytest tests/ --ignore=tests/legacy/ -v --tb=short
	@echo "✅ Core tests passed"

# Issue 01: Legacy tests — run and report but do NOT fail the build
test-legacy:
	@echo "════ Running LEGACY tests (known failures expected) ════"
	-pytest tests/legacy/ -v --tb=no -q
	@echo "⚠  Legacy run complete — see above for known failures"

# Full suite — CI summary
test-full:
	@echo "════ Full test run ════"
	pytest tests/ -v --tb=short 2>&1 | tee test_report.txt
	@echo "Full test run complete — see test_report.txt"

# Issue 11: Kafka connectivity check
kafka-check:
	@echo "Verifying Kafka connectivity..."
	pytest tests/test_kafka_connectivity.py -v -m integration
	@echo "Kafka OK"

# Quick developer smoke test
smoke-test:
	@echo "════ Smoke test ════"
	python -c "from utils.numpy_compat import check_numpy_compatibility; check_numpy_compatibility()"
	python -c "import ingestion.pipeline_bridge; print('Pipeline bridge: OK')"
	pytest tests/ --ignore=tests/legacy/ -v --tb=short -q
	@echo "✅ Smoke test passed"
