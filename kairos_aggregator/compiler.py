"""Compile scout outputs into the compact JSON the LLM receives.

This is where "the LLM never sees raw numbers" is enforced: only digested,
rounded, decision-relevant fields make it into the context.
"""

from __future__ import annotations

import json

from kairos_core.contracts import MarketSnapshot, RouterDecision, SentimentSignal
from kairos_core.enums import ImpactDirection, Side, SystemMode


def _ordered_unique(
    sentiments: list[SentimentSignal],
    *,
    limit: int,
) -> list[SentimentSignal]:
    """Return one copy of every message in a replay-stable newest-first order."""
    by_id: dict[str, SentimentSignal] = {}
    for sentiment in sentiments:
        by_id.setdefault(sentiment.message_id, sentiment)
    return sorted(
        by_id.values(),
        key=lambda item: (-item.produced_at.timestamp(), item.message_id),
    )[:limit]


def calibrated_sentiment(signal: SentimentSignal) -> float:
    """Mirror Router calibration; neutral or contradictory evidence has zero direction."""
    if signal.impact is ImpactDirection.NEUTRAL:
        return 0.0
    if signal.impact is ImpactDirection.BULLISH and signal.sentiment <= 0.0:
        return 0.0
    if signal.impact is ImpactDirection.BEARISH and signal.sentiment >= 0.0:
        return 0.0
    return signal.sentiment * signal.confidence


def calibrated_text_score(sentiments: list[SentimentSignal]) -> float:
    """Use the same non-normalized confidence calibration as Router."""
    return sum(calibrated_sentiment(signal) for signal in sentiments) / len(sentiments) if sentiments else 0.0


def calibrated_text_bias(sentiments: list[SentimentSignal], *, deadband: float) -> Side:
    score = calibrated_text_score(sentiments)
    if score > deadband:
        return Side.LONG
    if score < -deadband:
        return Side.SHORT
    return Side.FLAT


def compile_context(
    snapshot: MarketSnapshot,
    sentiments: list[SentimentSignal],
    router: RouterDecision | None = None,
    *,
    system_mode: SystemMode = SystemMode.NORMAL,
    max_sentiments: int = 5,
) -> str:
    selected = _ordered_unique(sentiments, limit=max_sentiments)
    text_degraded = system_mode is SystemMode.TEXT_LOCAL_FILTER
    text_score = calibrated_text_score(selected)
    confidence_mean = sum(signal.confidence for signal in selected) / len(selected) if selected else 0.0
    ctx = {
        "symbol": snapshot.symbol,
        "price": round(snapshot.mid_price, 2),
        "system_mode": system_mode.value,
        "text_reliability": "degraded_local_filter" if text_degraded else "normal",
        "provenance": {
            "snapshot_id": snapshot.message_id,
            "snapshot_produced_at": snapshot.produced_at.isoformat(),
            "router_decision_id": router.message_id if router is not None else None,
            "sentiment_ids": [signal.message_id for signal in selected],
        },
        "quant": {
            "bias": snapshot.quant_bias.value,
            "rsi": round(snapshot.indicators.rsi_14, 1),
            "macd_hist": round(snapshot.indicators.macd_hist, 4),
            "ob_imbalance": round(snapshot.order_book.imbalance, 3),
            "funding": snapshot.derivatives.funding_rate,
            "oi_change_1h": snapshot.derivatives.oi_change_pct_1h,
        },
        "text": {
            "calibrated_score": round(text_score, 4),
            "confidence_mean": round(confidence_mean, 4),
            "evidence_count": len(selected),
        },
        "news": [
            {
                "message_id": signal.message_id,
                "produced_at": signal.produced_at.isoformat(),
                "topic": signal.topic,
                "raw_sentiment": round(signal.sentiment, 4),
                "sentiment": round(calibrated_sentiment(signal), 4),
                "confidence": round(signal.confidence, 4),
                "impact": signal.impact.value,
                "summary": signal.summary,
                "sources": sorted(set(signal.sources))[:3],
            }
            for signal in selected
        ],
    }
    if router is not None:
        ctx["router"] = {
            "mode": router.mode.value,
            "conflict": router.conflict_streak,
            "quant_bias": router.quant_bias.value,
            "text_bias": router.text_bias.value,
        }
    return json.dumps(ctx, separators=(",", ":"), sort_keys=True)
