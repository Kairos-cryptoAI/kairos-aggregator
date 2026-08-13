"""Aggregator service wiring and cross-layer price propagation."""

import asyncio

from kairos_core.bus import BusEnvelope
from kairos_core.contracts import (
    DerivativesMetrics,
    MarketSnapshot,
    OrderBookSummary,
    RouterDecision,
    SentimentSignal,
    TacticalCommand,
    TechnicalIndicators,
)
from kairos_core.enums import (
    ImpactDirection,
    ReasonCode,
    ReasoningEffort,
    RouterMode,
    SystemMode,
    TacticalStatus,
)
from kairos_core.topics import Topics

from kairos_aggregator.config import AggregatorSettings
from kairos_aggregator.service import AggregatorService


class _SpyBus:
    def __init__(self, envelope, *, publish_error=None, ack_error=None):
        self.envelope = envelope
        self.publish_error = publish_error
        self.ack_error = ack_error
        self.calls = []
        self.acks = []
        self.closed = False

    async def publish(self, topic, message):
        if self.publish_error is not None:
            raise self.publish_error
        self.calls.append((topic, message))
        return "1"

    async def subscribe(self, topic, **kwargs):
        yield self.envelope

    async def ack(self, topic, envelope, **kwargs):
        self.acks.append((topic, envelope.id))
        if self.ack_error is not None:
            raise self.ack_error

    async def close(self):
        self.closed = True


class _FakeBrain:
    def __init__(self):
        self.calls = []

    async def decide(self, symbol, context_json, effort):
        self.calls.append((symbol, context_json, effort))
        return TacticalCommand(
            source="aggregator",
            symbol=symbol,
            status=TacticalStatus.STABLE_TREND_ENTRY,
            reason_code=ReasonCode.ENTER_LONG_TREND,
            effort_used=effort,
        )


class _FakeGateway:
    _on_health = object()

    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


def _snapshot():
    return MarketSnapshot(
        source="quant",
        symbol="BTCUSDT",
        mid_price=65_000,
        order_book=OrderBookSummary(
            best_bid=64_999,
            best_ask=65_001,
            spread_bps=0.3,
            imbalance=0.2,
            depth_usd=500_000,
        ),
        volume_usd=1_000_000,
        derivatives=DerivativesMetrics(funding_rate=0.0001, open_interest=10_000_000),
        indicators=TechnicalIndicators(rsi_14=55, macd=1.0, macd_signal=0.8, macd_hist=0.2),
    )


def _decision(*, snapshot_id: str, conflict: bool = False):
    return RouterDecision(
        source="router",
        symbol="BTCUSDT",
        mode=RouterMode.ROUTE_GPT if conflict else RouterMode.ROUTE_PRO,
        requested_effort=ReasoningEffort.HIGH if conflict else ReasoningEffort.MEDIUM,
        snapshot_id=snapshot_id,
    )


def test_gateway_health_hook_is_wired():
    svc = AggregatorService(AggregatorSettings(bus_backend="memory"))
    assert svc.brain.gateway._on_health is not None


def test_router_decision_forwards_snapshot_price_to_risk_command():
    svc = AggregatorService(AggregatorSettings(bus_backend="memory"))
    svc.brain = _FakeBrain()
    snapshot = _snapshot()
    svc._snapshots[snapshot.symbol] = snapshot
    svc._snapshots_by_id[snapshot.message_id] = snapshot
    decision = _decision(snapshot_id=snapshot.message_id)
    envelope = BusEnvelope(
        id="router-1",
        topic=Topics.ROUTER_DECISION,
        payload=decision.to_payload(),
    )
    svc.bus = _SpyBus(envelope)

    asyncio.run(svc._on_router())

    assert svc.bus.acks == [(Topics.ROUTER_DECISION, "router-1")]
    assert len(svc.bus.calls) == 1
    topic, command = svc.bus.calls[0]
    assert topic == Topics.TACTICAL_COMMAND
    assert command.reference_price == snapshot.mid_price


def test_router_message_is_not_acked_when_publish_fails():
    svc = AggregatorService(AggregatorSettings(bus_backend="memory"), gateway=_FakeGateway())
    brain = _FakeBrain()
    svc.brain = brain
    snapshot = _snapshot()
    svc._snapshots[snapshot.symbol] = snapshot
    svc._snapshots_by_id[snapshot.message_id] = snapshot
    envelope = BusEnvelope(
        id="router-1",
        topic=Topics.ROUTER_DECISION,
        payload=_decision(snapshot_id=snapshot.message_id).to_payload(),
    )
    svc.bus = _SpyBus(envelope, publish_error=RuntimeError("bus unavailable"))

    asyncio.run(svc._on_router())

    assert svc.bus.acks == []
    assert len(brain.calls) == 1

    svc.bus.publish_error = None
    asyncio.run(svc._on_router())

    assert len(brain.calls) == 1
    assert len(svc.bus.calls) == 1
    assert svc.bus.acks == [(Topics.ROUTER_DECISION, "router-1")]


def test_router_replay_after_failed_ack_does_not_repeat_llm_or_publish():
    svc = AggregatorService(AggregatorSettings(bus_backend="memory"), gateway=_FakeGateway())
    brain = _FakeBrain()
    svc.brain = brain
    snapshot = _snapshot()
    svc._snapshots[snapshot.symbol] = snapshot
    svc._snapshots_by_id[snapshot.message_id] = snapshot
    envelope = BusEnvelope(
        id="router-replay",
        topic=Topics.ROUTER_DECISION,
        payload=_decision(snapshot_id=snapshot.message_id).to_payload(),
    )
    bus = _SpyBus(envelope, ack_error=RuntimeError("redis unavailable"))
    svc.bus = bus

    asyncio.run(svc._on_router())
    bus.ack_error = None
    asyncio.run(svc._on_router())

    assert len(brain.calls) == 1
    assert len(bus.calls) == 1
    assert len(bus.acks) == 2


def test_router_decision_waits_for_its_exact_snapshot():
    svc = AggregatorService(AggregatorSettings(bus_backend="memory"), gateway=_FakeGateway())
    svc.brain = _FakeBrain()
    expected = _snapshot()
    newer = _snapshot()
    svc._snapshots[newer.symbol] = newer
    svc._snapshots_by_id[newer.message_id] = newer
    decision_envelope = BusEnvelope(
        id="router-deferred",
        topic=Topics.ROUTER_DECISION,
        payload=_decision(snapshot_id=expected.message_id).to_payload(),
    )
    bus = _SpyBus(decision_envelope)
    svc.bus = bus

    asyncio.run(svc._on_router())

    assert bus.calls == []
    assert bus.acks == []
    assert expected.message_id in svc._pending_decisions

    snapshot_envelope = BusEnvelope(
        id="snapshot-bus",
        topic=Topics.MARKET_SNAPSHOT,
        payload=expected.to_payload(),
    )
    asyncio.run(svc._handle_snapshot(snapshot_envelope))

    _, command = bus.calls[0]
    assert command.reference_price == expected.mid_price
    assert bus.acks == [(Topics.ROUTER_DECISION, "router-deferred")]
    assert expected.message_id not in svc._pending_decisions


def test_router_message_waits_unacked_for_missing_snapshot():
    svc = AggregatorService(AggregatorSettings(bus_backend="memory"), gateway=_FakeGateway())
    envelope = BusEnvelope(
        id="router-1",
        topic=Topics.ROUTER_DECISION,
        payload=_decision(snapshot_id="missing-snapshot").to_payload(),
    )
    svc.bus = _SpyBus(envelope)

    asyncio.run(svc._on_router())

    assert svc.bus.calls == []
    assert svc.bus.acks == []


def test_text_local_filter_clears_old_sentiment_and_acks_control():
    svc = AggregatorService(AggregatorSettings(bus_backend="memory"), gateway=_FakeGateway())
    svc._sentiments.append(
        SentimentSignal(
            source="text-scouts",
            topic="ETF",
            sentiment=0.9,
            impact=ImpactDirection.BULLISH,
            confidence=0.9,
        )
    )
    envelope = BusEnvelope(
        id="control-1",
        topic=Topics.SYSTEM_CONTROL,
        payload={"mode": SystemMode.TEXT_LOCAL_FILTER.value},
    )
    svc.bus = _SpyBus(envelope)

    asyncio.run(svc._track_control())

    assert svc.system_mode is SystemMode.TEXT_LOCAL_FILTER
    assert list(svc._sentiments) == []
    assert svc.bus.acks == [(Topics.SYSTEM_CONTROL, "control-1")]


def test_invalid_control_is_not_acked_or_applied():
    svc = AggregatorService(AggregatorSettings(bus_backend="memory"), gateway=_FakeGateway())
    envelope = BusEnvelope(
        id="control-1",
        topic=Topics.SYSTEM_CONTROL,
        payload={"mode": "UNKNOWN"},
    )
    svc.bus = _SpyBus(envelope)

    asyncio.run(svc._track_control())

    assert svc.system_mode is SystemMode.NORMAL
    assert svc.bus.acks == []


def test_recovery_from_text_local_filter_drops_degraded_sentiment():
    svc = AggregatorService(AggregatorSettings(bus_backend="memory"), gateway=_FakeGateway())
    svc.system_mode = SystemMode.TEXT_LOCAL_FILTER
    svc._sentiments.append(
        SentimentSignal(
            source="text-scouts:local",
            topic="ETF",
            sentiment=0.2,
            impact=ImpactDirection.BULLISH,
            confidence=0.2,
        )
    )
    envelope = BusEnvelope(
        id="control-1",
        topic=Topics.SYSTEM_CONTROL,
        payload={"mode": SystemMode.NORMAL.value},
    )
    svc.bus = _SpyBus(envelope)

    asyncio.run(svc._track_control())

    assert svc.system_mode is SystemMode.NORMAL
    assert list(svc._sentiments) == []


def test_conflict_safe_skips_high_effort_llm_and_publishes_no_trade():
    svc = AggregatorService(AggregatorSettings(bus_backend="memory"), gateway=_FakeGateway())
    brain = _FakeBrain()
    svc.brain = brain
    svc.system_mode = SystemMode.CONFLICT_SAFE
    snapshot = _snapshot()
    svc._snapshots[snapshot.symbol] = snapshot
    svc._snapshots_by_id[snapshot.message_id] = snapshot
    envelope = BusEnvelope(
        id="router-1",
        topic=Topics.ROUTER_DECISION,
        payload=_decision(snapshot_id=snapshot.message_id, conflict=True).to_payload(),
    )
    svc.bus = _SpyBus(envelope)

    asyncio.run(svc._on_router())

    assert brain.calls == []
    assert svc.bus.acks == [(Topics.ROUTER_DECISION, "router-1")]
    _, command = svc.bus.calls[0]
    assert command.status is TacticalStatus.WAIT_CONFIRMATION
    assert command.reason_code is ReasonCode.NO_TRADE
    assert command.reference_price == snapshot.mid_price


def test_conflict_safe_keeps_non_conflict_deepseek_route_available():
    svc = AggregatorService(AggregatorSettings(bus_backend="memory"), gateway=_FakeGateway())
    brain = _FakeBrain()
    svc.brain = brain
    svc.system_mode = SystemMode.CONFLICT_SAFE
    snapshot = _snapshot()
    svc._snapshots[snapshot.symbol] = snapshot
    svc._snapshots_by_id[snapshot.message_id] = snapshot
    envelope = BusEnvelope(
        id="router-1",
        topic=Topics.ROUTER_DECISION,
        payload=_decision(snapshot_id=snapshot.message_id).to_payload(),
    )
    svc.bus = _SpyBus(envelope)

    asyncio.run(svc._on_router())

    assert len(brain.calls) == 1
    assert svc.bus.acks == [(Topics.ROUTER_DECISION, "router-1")]


def test_run_uses_structured_tasks_and_closes_resources(monkeypatch):
    gateway = _FakeGateway()
    svc = AggregatorService(AggregatorSettings(bus_backend="memory"), gateway=gateway)
    bus = _SpyBus(None)
    svc.bus = bus
    completed = []

    def completed_task(name):
        async def task():
            completed.append(name)

        return task

    monkeypatch.setattr(svc, "_track_snapshots", completed_task("snapshots"))
    monkeypatch.setattr(svc, "_track_sentiment", completed_task("sentiment"))
    monkeypatch.setattr(svc, "_track_control", completed_task("control"))
    monkeypatch.setattr(svc, "_on_router", completed_task("router"))

    asyncio.run(svc.run())

    assert set(completed) == {"snapshots", "sentiment", "control", "router"}
    assert gateway.closed is True
    assert bus.closed is True
