"""Aggregator service wiring and cross-layer price propagation."""
import asyncio

from kairos_core.bus import BusEnvelope
from kairos_core.contracts import (
    DerivativesMetrics,
    MarketSnapshot,
    OrderBookSummary,
    RouterDecision,
    TacticalCommand,
    TechnicalIndicators,
)
from kairos_core.enums import ReasonCode, ReasoningEffort, RouterMode, TacticalStatus
from kairos_core.topics import Topics

from kairos_aggregator.config import AggregatorSettings
from kairos_aggregator.service import AggregatorService


class _SpyBus:
    def __init__(self, envelope):
        self.envelope = envelope
        self.calls = []
        self.acks = []

    async def publish(self, topic, message):
        self.calls.append((topic, message))
        return "1"

    async def subscribe(self, topic, **kwargs):
        yield self.envelope

    async def ack(self, topic, envelope, **kwargs):
        self.acks.append((topic, envelope.id))


class _FakeBrain:
    async def decide(self, symbol, context_json, effort):
        return TacticalCommand(
            source="aggregator", symbol=symbol, status=TacticalStatus.STABLE_TREND_ENTRY,
            reason_code=ReasonCode.ENTER_LONG_TREND, effort_used=effort,
        )


class _FakeGateway:
    _on_health = object()


def _snapshot():
    return MarketSnapshot(
        source="quant", symbol="BTCUSDT", mid_price=65_000,
        order_book=OrderBookSummary(
            best_bid=64_999, best_ask=65_001, spread_bps=0.3, imbalance=0.2,
            depth_usd=500_000,
        ),
        volume_usd=1_000_000,
        derivatives=DerivativesMetrics(funding_rate=0.0001, open_interest=10_000_000),
        indicators=TechnicalIndicators(rsi_14=55, macd=1.0, macd_signal=0.8, macd_hist=0.2),
    )


def test_gateway_health_hook_is_wired():
    svc = AggregatorService(AggregatorSettings(bus_backend="memory"))
    assert svc.brain.gateway._on_health is not None


def test_router_decision_forwards_snapshot_price_to_risk_command():
    svc = AggregatorService(AggregatorSettings(bus_backend="memory"))
    svc.brain = _FakeBrain()
    snapshot = _snapshot()
    svc._snapshots[snapshot.symbol] = snapshot
    decision = RouterDecision(
        source="router", symbol=snapshot.symbol, mode=RouterMode.ROUTE_PRO,
        requested_effort=ReasoningEffort.MEDIUM,
    )
    envelope = BusEnvelope(
        id="router-1", topic=Topics.ROUTER_DECISION, payload=decision.to_payload(),
    )
    svc.bus = _SpyBus(envelope)

    asyncio.run(svc._on_router())

    assert svc.bus.acks == [(Topics.ROUTER_DECISION, "router-1")]
    assert len(svc.bus.calls) == 1
    topic, command = svc.bus.calls[0]
    assert topic == Topics.TACTICAL_COMMAND
    assert command.reference_price == snapshot.mid_price
