"""
tests/test_streaming_proposals.py
-----------------------------------
Tests for the ProposalRouter streaming window-size proposals (Layer 7)
and the new bandit_summary() method.

Covers:
  - Streaming proposals generated when is_streaming=True
  - No streaming proposals when is_streaming=False (or absent)
  - Tumbling window recommendation present and sensibly parameterised
  - Sliding window proposal present
  - Session window only when session_gap_ms provided
  - Watermark config only when has_late_data=True
  - bandit_summary() returns correct call-count dict
  - Layer 7 proposals have standard proposal keys and valid confidence
  - High event-rate produces large expected events/window
  - Low latency budget produces small tumbling window
"""
from __future__ import annotations

import pytest


@pytest.fixture
def router():
    from proposal.proposal_router import ProposalRouter
    return ProposalRouter(domain="default", max_proposals=20)


class TestStreamingProposals:

    def test_no_streaming_proposals_without_flag(self, router):
        """When is_streaming not in extra_signals, no streaming ops generated."""
        props = router.route(extra_signals={"is_streaming": False}, row_count=0)
        ops = [p["operation"] for p in props]
        assert "streaming_window_config" not in ops
        assert "streaming_sliding_window" not in ops
        assert "streaming_watermark_config" not in ops

    def test_streaming_flag_triggers_proposals(self, router):
        """When is_streaming=True, streaming proposals are always generated."""
        props = router.route(extra_signals={"is_streaming": True}, row_count=0)
        ops = [p["operation"] for p in props]
        # At minimum: tumbling + sliding windows
        assert "streaming_window_config" in ops
        assert "streaming_sliding_window" in ops

    def test_tumbling_window_confidence(self, router):
        props = router.route(extra_signals={"is_streaming": True, "target_latency_ms": 5000})
        tumbling = [p for p in props if p["operation"] == "streaming_window_config"]
        assert len(tumbling) > 0
        assert tumbling[0]["confidence"] > 0.80

    def test_sliding_window_confidence(self, router):
        props = router.route(extra_signals={"is_streaming": True, "target_latency_ms": 5000})
        sliding = [p for p in props if p["operation"] == "streaming_sliding_window"]
        assert len(sliding) > 0
        assert sliding[0]["confidence"] > 0.70

    def test_session_window_only_with_gap(self, router):
        """Session window should NOT appear without session_gap_ms."""
        props_no_gap  = router.route(extra_signals={"is_streaming": True})
        props_with_gap = router.route(extra_signals={"is_streaming": True, "session_gap_ms": 30000})

        ops_no_gap   = [p["operation"] for p in props_no_gap]
        ops_with_gap = [p["operation"] for p in props_with_gap]

        assert "streaming_session_window" not in ops_no_gap
        assert "streaming_session_window" in ops_with_gap

    def test_watermark_only_with_late_data(self, router):
        """Watermark proposal should NOT appear without has_late_data=True."""
        props_no_late  = router.route(extra_signals={"is_streaming": True, "has_late_data": False})
        props_with_late = router.route(extra_signals={"is_streaming": True, "has_late_data": True})

        ops_no_late   = [p["operation"] for p in props_no_late]
        ops_with_late = [p["operation"] for p in props_with_late]

        assert "streaming_watermark_config" not in ops_no_late
        assert "streaming_watermark_config"     in ops_with_late

    def test_watermark_confidence_high(self, router):
        """Watermark proposal should have very high confidence (safety-critical)."""
        props = router.route(extra_signals={"is_streaming": True, "has_late_data": True})
        wm = [p for p in props if p["operation"] == "streaming_watermark_config"]
        assert len(wm) > 0
        assert wm[0]["confidence"] >= 0.88

    def test_all_streaming_props_have_standard_keys(self, router):
        props = router.route(extra_signals={
            "is_streaming": True, "has_late_data": True, "session_gap_ms": 10000
        })
        streaming = [p for p in props if "streaming" in p["operation"]]
        for p in streaming:
            for key in ("operation", "tier", "rationale", "confidence", "priority", "estimated_ms"):
                assert key in p, f"Missing key '{key}' in: {p}"

    def test_streaming_proposals_tier_is_senior(self, router):
        props = router.route(extra_signals={"is_streaming": True, "has_late_data": True})
        streaming = [p for p in props if "streaming" in p["operation"]]
        assert all(p["tier"] == "senior" for p in streaming)

    def test_streaming_proposals_confidence_in_range(self, router):
        props = router.route(extra_signals={"is_streaming": True, "has_late_data": True})
        streaming = [p for p in props if "streaming" in p["operation"]]
        for p in streaming:
            assert 0.0 <= p["confidence"] <= 1.0

    def test_high_event_rate_reflected_in_rationale(self, router):
        props = router.route(extra_signals={
            "is_streaming": True, "event_rate_per_sec": 5000, "target_latency_ms": 2000
        })
        tumbling = [p for p in props if p["operation"] == "streaming_window_config"]
        assert len(tumbling) > 0
        # 5000 events/s × 2s = 10,000 events mentioned
        rationale = tumbling[0]["rationale"]
        assert "5000" in rationale or "5,000" in rationale

    def test_low_latency_produces_small_window(self, router):
        """A 1s latency budget → tumbling window rationale mentions 1s."""
        props = router.route(extra_signals={"is_streaming": True, "target_latency_ms": 1000})
        tumbling = [p for p in props if p["operation"] == "streaming_window_config"]
        assert len(tumbling) > 0
        assert "1s" in tumbling[0]["rationale"]

    def test_high_latency_budget_capped(self, router):
        """Very high latency budget is capped at 300s (5 min)."""
        props = router.route(extra_signals={
            "is_streaming": True, "target_latency_ms": 1_000_000  # 1000s, should be capped
        })
        tumbling = [p for p in props if p["operation"] == "streaming_window_config"]
        assert len(tumbling) > 0
        # Should not mention anything > 300s
        assert "1000s" not in tumbling[0]["rationale"]


class TestBanditSummary:

    def test_bandit_summary_returns_dict(self, router):
        summary = router.bandit_summary()
        assert isinstance(summary, dict)

    def test_bandit_summary_empty_before_routes(self):
        from proposal.proposal_router import ProposalRouter
        fresh = ProposalRouter()
        assert fresh.bandit_summary() == {}

    def test_bandit_summary_increments_on_route(self, router):
        before = router.bandit_summary().copy()
        router.route(row_count=200)
        after = router.bandit_summary()
        # At minimum one operation counter should have increased
        assert sum(after.values()) > sum(before.values())

    def test_bandit_summary_counts_all_returned_ops(self, router):
        props = router.route(row_count=200)
        summary = router.bandit_summary()
        returned_ops = {p["operation"] for p in props}
        for op in returned_ops:
            assert op in summary
            assert summary[op] >= 1

    def test_bandit_summary_increments_across_multiple_calls(self, router):
        for _ in range(5):
            router.route(row_count=200)
        summary = router.bandit_summary()
        # Some ops should have been called multiple times
        assert any(v >= 5 for v in summary.values())
