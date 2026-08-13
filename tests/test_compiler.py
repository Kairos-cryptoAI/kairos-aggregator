import json
from datetime import UTC, datetime, timedelta

from kairos_core.contracts import (
    DerivativesMetrics,
    MarketSnapshot,
    OrderBookSummary,
    SentimentSignal,
    TechnicalIndicators,
)
from kairos_core.enums import ImpactDirection, Side, SystemMode

from kairos_aggregator.compiler import compile_context


def _snap():
    return MarketSnapshot(
        source="q",
        symbol="BTCUSD",
        mid_price=65000.123,
        volume_usd=1e6,
        order_book=OrderBookSummary(
            best_bid=64999, best_ask=65001, spread_bps=0.3, imbalance=0.2, depth_usd=5e5
        ),
        derivatives=DerivativesMetrics(funding_rate=1e-4, open_interest=1e9, oi_change_pct_1h=1.2),
        indicators=TechnicalIndicators(rsi_14=61.7, macd=5, macd_signal=4, macd_hist=1.234567),
        quant_bias=Side.LONG,
    )


def test_context_is_compact_and_digested():
    ctx = json.loads(
        compile_context(
            _snap(),
            [
                SentimentSignal(
                    source="t",
                    topic="ETF",
                    sentiment=0.8,
                    impact=ImpactDirection.BULLISH,
                    summary="spot fund approved",
                    sources=["https://example.test/etf"],
                )
            ],
        )
    )
    assert ctx["price"] == 65000.12  # rounded, no raw precision
    assert ctx["quant"]["bias"] == "LONG"
    assert ctx["quant"]["macd_hist"] == 1.2346
    assert ctx["news"][0]["topic"] == "ETF"
    assert ctx["news"][0]["raw_sentiment"] == 0.8
    assert ctx["news"][0]["sentiment"] == 0.4
    assert ctx["news"][0]["summary"] == "spot fund approved"
    assert ctx["news"][0]["sources"] == ["https://example.test/etf"]
    assert ctx["text"]["calibrated_score"] == 0.4
    assert ctx["provenance"]["snapshot_id"] is not None


def test_text_local_filter_dampens_sentiment_by_confidence():
    context = json.loads(
        compile_context(
            _snap(),
            [
                SentimentSignal(
                    source="text-scouts:local",
                    topic="ETF",
                    sentiment=0.8,
                    confidence=0.25,
                    impact=ImpactDirection.BULLISH,
                )
            ],
            system_mode=SystemMode.TEXT_LOCAL_FILTER,
        )
    )

    assert context["system_mode"] == "TEXT_LOCAL_FILTER"
    assert context["text_reliability"] == "degraded_local_filter"
    assert context["news"][0]["sentiment"] == 0.2
    assert context["news"][0]["confidence"] == 0.25


def test_context_deduplicates_ids_and_orders_evidence_deterministically():
    now = datetime(2026, 8, 13, 12, tzinfo=UTC)
    older = SentimentSignal(
        message_id="sentiment-b",
        produced_at=now - timedelta(seconds=1),
        source="text-scouts",
        topic="rates",
        sentiment=-0.4,
        confidence=0.5,
        impact=ImpactDirection.BEARISH,
        sources=["https://b.test", "https://a.test", "https://a.test"],
    )
    newer = SentimentSignal(
        message_id="sentiment-a",
        produced_at=now,
        source="text-scouts",
        topic="ETF",
        sentiment=0.8,
        confidence=0.75,
        impact=ImpactDirection.BULLISH,
    )

    snapshot = _snap()
    first = compile_context(snapshot, [older, newer, older])
    second = compile_context(snapshot, [newer, older])
    first_context = json.loads(first)

    assert first == second
    assert first_context["provenance"]["sentiment_ids"] == ["sentiment-a", "sentiment-b"]
    assert first_context["news"][1]["sources"] == ["https://a.test", "https://b.test"]
    assert first_context["text"]["calibrated_score"] == 0.2


def test_neutral_impact_has_zero_direction_even_with_nonzero_raw_sentiment():
    context = json.loads(
        compile_context(
            _snap(),
            [
                SentimentSignal(
                    source="text-scouts",
                    topic="background",
                    sentiment=0.9,
                    confidence=0.9,
                    impact=ImpactDirection.NEUTRAL,
                )
            ],
        )
    )

    assert context["news"][0]["raw_sentiment"] == 0.9
    assert context["news"][0]["sentiment"] == 0.0
    assert context["text"]["calibrated_score"] == 0.0


def test_impact_score_contradiction_matches_router_zero_calibration():
    context = json.loads(
        compile_context(
            _snap(),
            [
                SentimentSignal(
                    source="text-scouts",
                    topic="contradictory",
                    sentiment=-0.9,
                    confidence=1.0,
                    impact=ImpactDirection.BULLISH,
                )
            ],
        )
    )

    assert context["news"][0]["raw_sentiment"] == -0.9
    assert context["news"][0]["sentiment"] == 0.0
    assert context["text"]["calibrated_score"] == 0.0
