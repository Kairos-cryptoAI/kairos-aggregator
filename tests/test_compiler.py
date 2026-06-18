import json
from kairos_aggregator.compiler import compile_context
from kairos_core.contracts import (MarketSnapshot, OrderBookSummary, DerivativesMetrics, TechnicalIndicators, SentimentSignal)
from kairos_core.enums import Side, ImpactDirection


def _snap():
    return MarketSnapshot(source="q", symbol="BTCUSD", mid_price=65000.123, volume_usd=1e6,
        order_book=OrderBookSummary(best_bid=64999, best_ask=65001, spread_bps=0.3, imbalance=0.2, depth_usd=5e5),
        derivatives=DerivativesMetrics(funding_rate=1e-4, open_interest=1e9, oi_change_pct_1h=1.2),
        indicators=TechnicalIndicators(rsi_14=61.7, macd=5, macd_signal=4, macd_hist=1.234567),
        quant_bias=Side.LONG)


def test_context_is_compact_and_digested():
    ctx = json.loads(compile_context(_snap(), [SentimentSignal(source="t", topic="ETF", sentiment=0.8, impact=ImpactDirection.BULLISH)]))
    assert ctx["price"] == 65000.12          # rounded, no raw precision
    assert ctx["quant"]["bias"] == "LONG"
    assert ctx["quant"]["macd_hist"] == 1.2346
    assert ctx["news"][0]["topic"] == "ETF"
