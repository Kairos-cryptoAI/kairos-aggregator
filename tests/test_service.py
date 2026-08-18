"""Aggregator service wiring and cross-layer price propagation."""

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest
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
    Side,
    SystemMode,
    TacticalStatus,
)
from kairos_core.topics import Topics
from kairos_llm import BudgetedLLMGateway, DenyLLMUsageBudget
from kairos_persistence import DurableLLMUsageBudget

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


def _snapshot(*, produced_at: datetime | None = None, message_id: str | None = None):
    values = {
        "source": "quant",
        "symbol": "BTCUSDT",
        "mid_price": 65_000,
        "order_book": OrderBookSummary(
            best_bid=64_999,
            best_ask=65_001,
            spread_bps=0.3,
            imbalance=0.2,
            depth_usd=500_000,
        ),
        "volume_usd": 1_000_000,
        "derivatives": DerivativesMetrics(funding_rate=0.0001, open_interest=10_000_000),
        "indicators": TechnicalIndicators(rsi_14=55, macd=1.0, macd_signal=0.8, macd_hist=0.2),
    }
    if produced_at is not None:
        values["produced_at"] = produced_at
    if message_id is not None:
        values["message_id"] = message_id
    return MarketSnapshot(**values)


def _decision(
    *,
    snapshot_id: str,
    conflict: bool = False,
    produced_at: datetime | None = None,
    sentiment_ids: list[str] | None = None,
):
    values = {
        "source": "router",
        "symbol": "BTCUSDT",
        "mode": RouterMode.ROUTE_GPT if conflict else RouterMode.ROUTE_PRO,
        "requested_effort": ReasoningEffort.HIGH if conflict else ReasoningEffort.MEDIUM,
        "snapshot_id": snapshot_id,
        "sentiment_ids": sentiment_ids or [],
        "text_bias": Side.LONG if sentiment_ids else Side.FLAT,
    }
    if produced_at is not None:
        values["produced_at"] = produced_at
    return RouterDecision(**values)


def test_gateway_health_hook_is_wired():
    svc = AggregatorService(AggregatorSettings(bus_backend="memory"))
    assert isinstance(svc.brain.gateway, BudgetedLLMGateway)
    assert isinstance(svc.brain.gateway.budget, DenyLLMUsageBudget)
    assert svc.brain.gateway._on_health is not None


def test_durable_runtime_wires_shared_provider_budget():
    svc = AggregatorService(AggregatorSettings(bus_backend="redis"))
    assert isinstance(svc.brain.gateway.budget, DurableLLMUsageBudget)


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


def test_snapshot_replay_is_immutable_and_changed_payload_is_rejected():
    now = datetime(2026, 8, 13, 12, tzinfo=UTC)
    svc = AggregatorService(
        AggregatorSettings(bus_backend="memory"),
        gateway=_FakeGateway(),
        clock=lambda: now,
    )
    original = _snapshot(produced_at=now, message_id="immutable")
    envelope = BusEnvelope(
        id="original",
        topic=Topics.MARKET_SNAPSHOT,
        payload=original.to_payload(),
    )

    asyncio.run(svc._handle_snapshot(envelope))
    asyncio.run(svc._handle_snapshot(envelope))

    assert svc._snapshots_by_id[original.message_id].mid_price == original.mid_price

    changed_price = original.model_copy(update={"mid_price": original.mid_price + 1})
    with pytest.raises(ValueError, match="message_id .* was reused"):
        asyncio.run(
            svc._handle_snapshot(
                BusEnvelope(
                    id="changed-price",
                    topic=Topics.MARKET_SNAPSHOT,
                    payload=changed_price.to_payload(),
                )
            )
        )

    changed_symbol = original.model_copy(update={"symbol": "ETHUSDT"})
    with pytest.raises(ValueError, match="message_id .* was reused"):
        asyncio.run(
            svc._handle_snapshot(
                BusEnvelope(
                    id="changed-symbol",
                    topic=Topics.MARKET_SNAPSHOT,
                    payload=changed_symbol.to_payload(),
                )
            )
        )

    assert svc._snapshots_by_id[original.message_id] == original
    assert "ETHUSDT" not in svc._snapshots


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
    assert decision_envelope.payload["message_id"] in svc._pending_decisions

    snapshot_envelope = BusEnvelope(
        id="snapshot-bus",
        topic=Topics.MARKET_SNAPSHOT,
        payload=expected.to_payload(),
    )
    asyncio.run(svc._handle_snapshot(snapshot_envelope))

    _, command = bus.calls[0]
    assert command.reference_price == expected.mid_price
    assert bus.acks == [(Topics.ROUTER_DECISION, "router-deferred")]
    assert decision_envelope.payload["message_id"] not in svc._pending_decisions


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


def test_stale_router_decision_abstains_without_waiting_for_missing_evidence():
    now = datetime(2026, 8, 13, 12, tzinfo=UTC)
    svc = AggregatorService(
        AggregatorSettings(bus_backend="memory", router_decision_ttl_s=60),
        gateway=_FakeGateway(),
        clock=lambda: now,
    )
    brain = _FakeBrain()
    svc.brain = brain
    decision = _decision(
        snapshot_id="never-arrived",
        produced_at=now - timedelta(seconds=61),
        sentiment_ids=["also-missing"],
    )
    envelope = BusEnvelope(id="router", topic=Topics.ROUTER_DECISION, payload=decision.to_payload())
    svc.bus = _SpyBus(envelope)

    asyncio.run(svc._on_router())

    assert brain.calls == []
    assert svc.bus.acks == [(Topics.ROUTER_DECISION, "router")]
    assert decision.message_id not in svc._pending_decisions
    _, command = svc.bus.calls[0]
    assert command.reason_code is ReasonCode.NO_TRADE
    assert command.reference_price == 0.0


def test_only_router_referenced_sentiment_enters_context():
    now = datetime(2026, 8, 13, 12, tzinfo=UTC)
    svc = AggregatorService(
        AggregatorSettings(bus_backend="memory"),
        gateway=_FakeGateway(),
        clock=lambda: now,
    )
    brain = _FakeBrain()
    svc.brain = brain
    snapshot = _snapshot(produced_at=now, message_id="snapshot")
    svc._snapshots_by_id[snapshot.message_id] = snapshot
    referenced = SentimentSignal(
        message_id="sentiment-referenced",
        produced_at=now - timedelta(seconds=1),
        source="text-scouts",
        topic="ETF",
        sentiment=0.8,
        confidence=0.75,
        impact=ImpactDirection.BULLISH,
    )
    unrelated = SentimentSignal(
        message_id="sentiment-unrelated",
        produced_at=now,
        source="text-scouts",
        topic="rates",
        sentiment=-0.9,
        confidence=0.9,
        impact=ImpactDirection.BEARISH,
    )
    for signal in (referenced, unrelated):
        svc._sentiments_by_id[signal.message_id] = signal
        svc._remember(svc._processed_sentiment, signal.message_id)
    decision = _decision(
        snapshot_id=snapshot.message_id,
        produced_at=now,
        sentiment_ids=[referenced.message_id],
    )
    envelope = BusEnvelope(id="router", topic=Topics.ROUTER_DECISION, payload=decision.to_payload())
    svc.bus = _SpyBus(envelope)

    asyncio.run(svc._on_router())

    context = json.loads(brain.calls[0][1])
    assert context["provenance"]["sentiment_ids"] == [referenced.message_id]
    assert [item["message_id"] for item in context["news"]] == [referenced.message_id]


def test_router_text_bias_must_match_confidence_calibrated_evidence():
    now = datetime(2026, 8, 13, 12, tzinfo=UTC)
    svc = AggregatorService(
        AggregatorSettings(bus_backend="memory"),
        gateway=_FakeGateway(),
        clock=lambda: now,
    )
    brain = _FakeBrain()
    svc.brain = brain
    snapshot = _snapshot(produced_at=now, message_id="snapshot")
    svc._snapshots_by_id[snapshot.message_id] = snapshot
    bullish = SentimentSignal(
        message_id="bullish",
        produced_at=now,
        source="text-scouts",
        topic="ETF",
        sentiment=0.8,
        confidence=0.8,
        impact=ImpactDirection.BULLISH,
    )
    svc._sentiments_by_id[bullish.message_id] = bullish
    svc._remember(svc._processed_sentiment, bullish.message_id)
    decision = _decision(
        snapshot_id=snapshot.message_id,
        produced_at=now,
        sentiment_ids=[bullish.message_id],
    ).model_copy(update={"text_bias": Side.SHORT})
    envelope = BusEnvelope(id="router", topic=Topics.ROUTER_DECISION, payload=decision.to_payload())
    svc.bus = _SpyBus(envelope)

    asyncio.run(svc._on_router())

    assert brain.calls == []
    _, command = svc.bus.calls[0]
    assert command.reason_code is ReasonCode.NO_TRADE
    assert "does not match calibrated evidence LONG" in command.rationale


def test_router_decision_waits_for_exact_sentiment_then_retries():
    now = datetime(2026, 8, 13, 12, tzinfo=UTC)
    svc = AggregatorService(
        AggregatorSettings(bus_backend="memory"),
        gateway=_FakeGateway(),
        clock=lambda: now,
    )
    brain = _FakeBrain()
    svc.brain = brain
    snapshot = _snapshot(produced_at=now, message_id="snapshot")
    svc._snapshots_by_id[snapshot.message_id] = snapshot
    decision = _decision(
        snapshot_id=snapshot.message_id,
        produced_at=now,
        sentiment_ids=["sentiment-late"],
    )
    router_envelope = BusEnvelope(
        id="router-late",
        topic=Topics.ROUTER_DECISION,
        payload=decision.to_payload(),
    )
    svc.bus = _SpyBus(router_envelope)

    asyncio.run(svc._on_router())

    assert brain.calls == []
    assert svc.bus.acks == []
    assert decision.message_id in svc._pending_decisions

    sentiment = SentimentSignal(
        message_id="sentiment-late",
        produced_at=now - timedelta(seconds=1),
        source="text-scouts",
        topic="ETF",
        sentiment=0.7,
        confidence=0.8,
        impact=ImpactDirection.BULLISH,
    )
    sentiment_envelope = BusEnvelope(
        id="sentiment-envelope",
        topic=Topics.SENTIMENT_SIGNAL,
        payload=sentiment.to_payload(),
    )
    asyncio.run(svc._handle_sentiment(sentiment_envelope))

    assert len(brain.calls) == 1
    assert svc.bus.acks == [(Topics.ROUTER_DECISION, "router-late")]
    assert decision.message_id not in svc._pending_decisions


def test_oversized_router_evidence_abstains_instead_of_silently_truncating():
    now = datetime(2026, 8, 13, 12, tzinfo=UTC)
    svc = AggregatorService(
        AggregatorSettings(bus_backend="memory", max_sentiment_window=2),
        gateway=_FakeGateway(),
        clock=lambda: now,
    )
    brain = _FakeBrain()
    svc.brain = brain
    snapshot = _snapshot(produced_at=now, message_id="snapshot")
    svc._snapshots_by_id[snapshot.message_id] = snapshot
    decision = _decision(
        snapshot_id=snapshot.message_id,
        produced_at=now,
        sentiment_ids=["one", "two", "three"],
    )
    envelope = BusEnvelope(id="router", topic=Topics.ROUTER_DECISION, payload=decision.to_payload())
    svc.bus = _SpyBus(envelope)

    asyncio.run(svc._on_router())

    assert brain.calls == []
    assert decision.message_id not in svc._pending_decisions
    _, command = svc.bus.calls[0]
    assert command.reason_code is ReasonCode.NO_TRADE
    assert "referenced 3 sentiment signals; maximum is 2" in command.rationale


def test_stale_snapshot_abstains_without_calling_model():
    now = datetime(2026, 8, 13, 12, tzinfo=UTC)
    settings = AggregatorSettings(bus_backend="memory", snapshot_ttl_s=60)
    svc = AggregatorService(settings, gateway=_FakeGateway(), clock=lambda: now)
    brain = _FakeBrain()
    svc.brain = brain
    snapshot = _snapshot(produced_at=now - timedelta(seconds=61), message_id="stale")
    svc._snapshots_by_id[snapshot.message_id] = snapshot
    decision = _decision(snapshot_id=snapshot.message_id, produced_at=now)
    envelope = BusEnvelope(id="router", topic=Topics.ROUTER_DECISION, payload=decision.to_payload())
    svc.bus = _SpyBus(envelope)

    asyncio.run(svc._on_router())

    assert brain.calls == []
    _, command = svc.bus.calls[0]
    assert command.reason_code is ReasonCode.NO_TRADE
    assert command.confidence == 0.0
    assert command.effort_used is ReasoningEffort.MEDIUM
    assert "market snapshot is stale" in command.rationale


def test_future_sentiment_abstains_without_calling_model():
    now = datetime(2026, 8, 13, 12, tzinfo=UTC)
    svc = AggregatorService(
        AggregatorSettings(bus_backend="memory", max_future_skew_s=1),
        gateway=_FakeGateway(),
        clock=lambda: now,
    )
    brain = _FakeBrain()
    svc.brain = brain
    snapshot = _snapshot(produced_at=now, message_id="snapshot")
    svc._snapshots_by_id[snapshot.message_id] = snapshot
    future = SentimentSignal(
        message_id="future",
        produced_at=now + timedelta(seconds=2),
        source="text-scouts",
        topic="ETF",
        sentiment=0.8,
        confidence=0.8,
        impact=ImpactDirection.BULLISH,
    )
    svc._sentiments_by_id[future.message_id] = future
    svc._remember(svc._processed_sentiment, future.message_id)
    decision = _decision(
        snapshot_id=snapshot.message_id,
        produced_at=now,
        sentiment_ids=[future.message_id],
    )
    envelope = BusEnvelope(id="router", topic=Topics.ROUTER_DECISION, payload=decision.to_payload())
    svc.bus = _SpyBus(envelope)

    asyncio.run(svc._on_router())

    assert brain.calls == []
    _, command = svc.bus.calls[0]
    assert command.reason_code is ReasonCode.NO_TRADE
    assert "postdates snapshot" in command.rationale


def test_sentiment_one_second_after_snapshot_only_applies_to_later_snapshot():
    snapshot_at = datetime(2026, 8, 13, 12, tzinfo=UTC)
    signal_at = snapshot_at + timedelta(seconds=1)
    svc = AggregatorService(
        AggregatorSettings(bus_backend="memory"),
        gateway=_FakeGateway(),
        clock=lambda: signal_at,
    )
    brain = _FakeBrain()
    svc.brain = brain
    current = _snapshot(produced_at=snapshot_at, message_id="snapshot-current")
    later = _snapshot(produced_at=signal_at, message_id="snapshot-later")
    svc._snapshots_by_id[current.message_id] = current
    svc._snapshots_by_id[later.message_id] = later
    signal = SentimentSignal(
        message_id="sentiment-plus-one",
        produced_at=signal_at,
        source="text-scouts",
        topic="ETF",
        sentiment=0.8,
        confidence=0.8,
        impact=ImpactDirection.BULLISH,
    )
    svc._sentiments_by_id[signal.message_id] = signal
    svc._remember(svc._processed_sentiment, signal.message_id)
    current_decision = _decision(
        snapshot_id=current.message_id,
        produced_at=signal_at,
        sentiment_ids=[signal.message_id],
    )
    later_decision = _decision(
        snapshot_id=later.message_id,
        produced_at=signal_at,
        sentiment_ids=[signal.message_id],
    )
    svc.bus = _SpyBus(None)

    asyncio.run(
        svc._handle_router(
            BusEnvelope(
                id="current-decision",
                topic=Topics.ROUTER_DECISION,
                payload=current_decision.to_payload(),
            )
        )
    )
    asyncio.run(
        svc._handle_router(
            BusEnvelope(
                id="later-decision",
                topic=Topics.ROUTER_DECISION,
                payload=later_decision.to_payload(),
            )
        )
    )

    assert len(brain.calls) == 1
    assert json.loads(brain.calls[0][1])["provenance"]["snapshot_id"] == later.message_id
    assert svc.bus.calls[0][1].reason_code is ReasonCode.NO_TRADE


def test_snapshot_one_second_after_decision_is_strictly_noncausal():
    decision_at = datetime(2026, 8, 13, 12, tzinfo=UTC)
    snapshot_at = decision_at + timedelta(seconds=1)
    svc = AggregatorService(
        AggregatorSettings(bus_backend="memory"),
        gateway=_FakeGateway(),
        clock=lambda: snapshot_at,
    )
    brain = _FakeBrain()
    svc.brain = brain
    snapshot = _snapshot(produced_at=snapshot_at, message_id="snapshot-plus-one")
    svc._snapshots_by_id[snapshot.message_id] = snapshot
    early = _decision(snapshot_id=snapshot.message_id, produced_at=decision_at)
    causal = _decision(snapshot_id=snapshot.message_id, produced_at=snapshot_at)
    svc.bus = _SpyBus(None)

    asyncio.run(
        svc._handle_router(BusEnvelope(id="early", topic=Topics.ROUTER_DECISION, payload=early.to_payload()))
    )
    asyncio.run(
        svc._handle_router(
            BusEnvelope(id="causal", topic=Topics.ROUTER_DECISION, payload=causal.to_payload())
        )
    )

    assert svc.bus.calls[0][1].reason_code is ReasonCode.NO_TRADE
    assert "postdates router decision" in svc.bus.calls[0][1].rationale
    assert len(brain.calls) == 1


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
