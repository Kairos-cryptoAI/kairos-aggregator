"""Compile scout outputs into the compact JSON the LLM receives.

This is where "the LLM never sees raw numbers" is enforced: only digested,
rounded, decision-relevant fields make it into the context.
"""
from __future__ import annotations

import json
from typing import List, Optional

from kairos_core.contracts import MarketSnapshot, RouterDecision, SentimentSignal


def compile_context(snapshot: MarketSnapshot, sentiments: List[SentimentSignal],
                    router: Optional[RouterDecision] = None) -> str:
    ctx = {
        "symbol": snapshot.symbol,
        "price": round(snapshot.mid_price, 2),
        "quant": {
            "bias": snapshot.quant_bias.value,
            "rsi": round(snapshot.indicators.rsi_14, 1),
            "macd_hist": round(snapshot.indicators.macd_hist, 4),
            "ob_imbalance": round(snapshot.order_book.imbalance, 3),
            "funding": snapshot.derivatives.funding_rate,
            "oi_change_1h": snapshot.derivatives.oi_change_pct_1h,
        },
        "news": [
            {"topic": s.topic, "sentiment": round(s.sentiment, 2), "impact": s.impact.value}
            for s in sentiments[:5]
        ],
    }
    if router is not None:
        ctx["router"] = {"mode": router.mode.value, "conflict": router.conflict_streak,
                         "text_bias": router.text_bias.value}
    return json.dumps(ctx, separators=(",", ":"))
