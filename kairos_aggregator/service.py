"""Aggregator service: router decision -> compile -> LLM -> tactical command."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from kairos_core.bus import BusEnvelope, build_bus
from kairos_core.contracts import (
    LLMHealthEvent,
    MarketSnapshot,
    RouterDecision,
    SentimentSignal,
    TacticalCommand,
)
from kairos_core.enums import ReasoningEffort, RouterMode, SystemMode
from kairos_core.logging import configure_logging, get_logger
from kairos_core.topics import Topics

from .compiler import calibrated_text_bias, compile_context
from .config import AggregatorSettings
from .decider import AggregatorBrain, safe_command

log = get_logger("aggregator")


class _DeferredDecision(RuntimeError):
    """A router decision whose exact source evidence has not arrived yet."""


class AggregatorService:
    def __init__(
        self,
        settings: AggregatorSettings | None = None,
        *,
        gateway=None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings or AggregatorSettings()
        self.bus = build_bus(self.settings)
        if gateway is None:
            from kairos_llm import LLMGateway

            gateway = LLMGateway(on_health=self._publish_health)
        self.gateway = gateway
        self.brain = AggregatorBrain(
            gateway,
            source=self.settings.service_name,
            min_entry_confidence=self.settings.min_entry_confidence,
        )
        self._clock = clock or (lambda: datetime.now(UTC))
        self.system_mode = SystemMode.NORMAL
        self._snapshots: dict[str, MarketSnapshot] = {}
        self._snapshots_by_id: OrderedDict[str, MarketSnapshot] = OrderedDict()
        self._seen_snapshot_ids: OrderedDict[str, str] = OrderedDict()
        self._pending_decisions: OrderedDict[str, BusEnvelope] = OrderedDict()
        self._processed_router: OrderedDict[str, None] = OrderedDict()
        self._command_cache: OrderedDict[str, TacticalCommand] = OrderedDict()
        self._processed_sentiment: OrderedDict[str, None] = OrderedDict()
        self._sentiments: deque[SentimentSignal] = deque(maxlen=self.settings.max_sentiment_window)
        self._sentiments_by_id: OrderedDict[str, SentimentSignal] = OrderedDict()
        self._closed = False

    def _remember(self, cache: OrderedDict[str, None], message_id: str) -> None:
        cache[message_id] = None
        cache.move_to_end(message_id)
        while len(cache) > self.settings.processed_cache_size:
            cache.popitem(last=False)

    def _remember_snapshot_identity(self, message_id: str, digest: str) -> None:
        self._seen_snapshot_ids[message_id] = digest
        self._seen_snapshot_ids.move_to_end(message_id)
        while len(self._seen_snapshot_ids) > self.settings.processed_cache_size:
            self._seen_snapshot_ids.popitem(last=False)

    @staticmethod
    def _snapshot_digest(snapshot: MarketSnapshot) -> str:
        payload = json.dumps(snapshot.to_payload(), separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def _require_aware(value: datetime, *, field: str) -> None:
        if value.utcoffset() is None:
            raise ValueError(f"{field} must be timezone-aware")

    def _now(self) -> datetime:
        now = self._clock()
        self._require_aware(now, field="clock")
        return now.astimezone(UTC)

    def _freshness_issue(
        self,
        *,
        label: str,
        produced_at: datetime,
        reference: datetime,
        ttl_s: float,
    ) -> str | None:
        age_s = (reference - produced_at).total_seconds()
        if age_s < -self.settings.max_future_skew_s:
            return f"{label} is {abs(age_s):.3f}s in the future"
        if age_s > ttl_s:
            return f"{label} is stale by age {age_s:.3f}s (ttl {ttl_s:.3f}s)"
        return None

    def _defer(self, decision: RouterDecision, envelope: BusEnvelope, detail: str) -> None:
        self._pending_decisions[decision.message_id] = envelope
        self._pending_decisions.move_to_end(decision.message_id)
        while len(self._pending_decisions) > self.settings.snapshot_cache_size:
            self._pending_decisions.popitem(last=False)
        raise _DeferredDecision(detail)

    async def _retry_pending_decisions(self) -> None:
        for decision_id, pending in list(self._pending_decisions.items()):
            try:
                await self._handle_router(pending)
            except _DeferredDecision:
                continue
            await self.bus.ack(Topics.ROUTER_DECISION, pending, group="aggregator")
            self._pending_decisions.pop(decision_id, None)

    async def _publish_health(self, model: str, provider: str, ok: bool, kind: str, latency_s: float) -> None:
        await self.bus.publish(
            Topics.LLM_HEALTH,
            LLMHealthEvent(
                source=self.settings.service_name,
                provider=provider,
                model=model,
                ok=ok,
                kind=kind,
                latency_s=latency_s,
            ),
        )

    async def _consume(
        self,
        topic: str,
        *,
        consumer: str,
        handler: Callable[[BusEnvelope], Awaitable[None]],
    ) -> None:
        """ACK only messages whose validation and side effects completed successfully."""
        async for envelope in self.bus.subscribe(topic, group="aggregator", consumer=consumer):
            try:
                await handler(envelope)
                await self.bus.ack(topic, envelope, group="aggregator")
            except _DeferredDecision as exc:
                log.info("aggregator.decision_deferred", detail=str(exc))
                continue
            except Exception:
                log.exception(
                    "aggregator.message_processing_failed",
                    topic=topic,
                    envelope_id=envelope.id,
                    attempt=envelope.attempt,
                )
                continue

    async def _handle_snapshot(self, envelope: BusEnvelope) -> None:
        snapshot = MarketSnapshot.model_validate(envelope.payload)
        self._require_aware(snapshot.produced_at, field="market snapshot produced_at")
        digest = self._snapshot_digest(snapshot)
        prior_digest = self._seen_snapshot_ids.get(snapshot.message_id)
        if prior_digest is not None:
            if prior_digest != digest:
                raise ValueError(f"snapshot message_id {snapshot.message_id!r} was reused")
            return
        future_issue = self._freshness_issue(
            label="market snapshot",
            produced_at=snapshot.produced_at,
            reference=self._now(),
            ttl_s=float("inf"),
        )
        if future_issue is not None:
            raise ValueError(future_issue)
        if not self.settings.symbol_allowed(snapshot.symbol):
            log.warning("aggregator.symbol_rejected", symbol=snapshot.symbol)
            return
        current = self._snapshots.get(snapshot.symbol)
        if current is None or snapshot.produced_at >= current.produced_at:
            self._snapshots[snapshot.symbol] = snapshot
        self._snapshots_by_id[snapshot.message_id] = snapshot
        self._snapshots_by_id.move_to_end(snapshot.message_id)
        self._remember_snapshot_identity(snapshot.message_id, digest)
        while len(self._snapshots_by_id) > self.settings.snapshot_cache_size:
            self._snapshots_by_id.popitem(last=False)
        await self._retry_pending_decisions()

    async def _track_snapshots(self) -> None:
        await self._consume(
            Topics.MARKET_SNAPSHOT,
            consumer="snap",
            handler=self._handle_snapshot,
        )

    async def _handle_sentiment(self, envelope: BusEnvelope) -> None:
        sentiment = SentimentSignal.model_validate(envelope.payload)
        self._require_aware(sentiment.produced_at, field="sentiment produced_at")
        future_issue = self._freshness_issue(
            label="sentiment",
            produced_at=sentiment.produced_at,
            reference=self._now(),
            ttl_s=float("inf"),
        )
        if future_issue is not None:
            raise ValueError(future_issue)
        if sentiment.message_id in self._processed_sentiment:
            return
        self._sentiments.append(sentiment)
        self._sentiments_by_id[sentiment.message_id] = sentiment
        self._sentiments_by_id.move_to_end(sentiment.message_id)
        while len(self._sentiments_by_id) > self.settings.processed_cache_size:
            self._sentiments_by_id.popitem(last=False)
        self._remember(self._processed_sentiment, sentiment.message_id)
        await self._retry_pending_decisions()

    async def _track_sentiment(self) -> None:
        await self._consume(
            Topics.SENTIMENT_SIGNAL,
            consumer="sent",
            handler=self._handle_sentiment,
        )

    async def _handle_control(self, envelope: BusEnvelope) -> None:
        mode_value = envelope.payload.get("mode")
        if not isinstance(mode_value, str):
            raise ValueError("system control mode must be a string")
        mode = SystemMode(mode_value)
        previous = self.system_mode
        self.system_mode = mode
        if (previous is SystemMode.TEXT_LOCAL_FILTER) != (mode is SystemMode.TEXT_LOCAL_FILTER):
            # Never mix model-derived and low-confidence local sentiment across recovery edges.
            self._sentiments.clear()
            self._sentiments_by_id.clear()
        if mode is not previous:
            log.warning("aggregator.mode_change", previous=previous.value, mode=mode.value)
        await self._retry_pending_decisions()

    async def _track_control(self) -> None:
        await self._consume(
            Topics.SYSTEM_CONTROL,
            consumer="control",
            handler=self._handle_control,
        )

    def _must_degrade(self, decision: RouterDecision) -> bool:
        if self.system_mode is SystemMode.LOCAL_QUANT_MODE:
            return True
        conflict = decision.mode is RouterMode.ROUTE_GPT or decision.requested_effort is ReasoningEffort.HIGH
        return self.system_mode is SystemMode.CONFLICT_SAFE and conflict

    async def _handle_router(self, envelope: BusEnvelope) -> None:
        decision = RouterDecision.model_validate(envelope.payload)
        self._require_aware(decision.produced_at, field="router decision produced_at")
        if decision.message_id in self._processed_router:
            return
        if not self.settings.symbol_allowed(decision.symbol):
            log.warning("aggregator.decision_symbol_rejected", symbol=decision.symbol)
            return

        command = self._command_cache.get(decision.message_id)
        if command is not None:
            await self.bus.publish(Topics.TACTICAL_COMMAND, command)
            self._remember(self._processed_router, decision.message_id)
            self._command_cache.pop(decision.message_id, None)
            self._pending_decisions.pop(decision.message_id, None)
            return

        now = self._now()
        evidence_issue = self._freshness_issue(
            label="router decision",
            produced_at=decision.produced_at,
            reference=now,
            ttl_s=self.settings.router_decision_ttl_s,
        )
        if evidence_issue is None and self._must_degrade(decision):
            evidence_issue = f"system mode {self.system_mode.value}: LLM route suppressed"

        snapshot = (
            self._snapshots_by_id.get(decision.snapshot_id) if decision.snapshot_id is not None else None
        )
        if snapshot is None:
            if decision.snapshot_id is None:
                evidence_issue = evidence_issue or "router decision is missing snapshot_id"
            elif evidence_issue is None and decision.snapshot_id not in self._seen_snapshot_ids:
                self._defer(
                    decision,
                    envelope,
                    f"snapshot {decision.snapshot_id} is not available for {decision.symbol}",
                )
            elif evidence_issue is None:
                evidence_issue = f"snapshot {decision.snapshot_id} was evicted before aggregation"
            sentiments: list[SentimentSignal] = []
        else:
            if evidence_issue is None and snapshot.symbol != decision.symbol:
                evidence_issue = (
                    f"snapshot symbol {snapshot.symbol} does not match decision {decision.symbol}"
                )
            elif evidence_issue is None and snapshot.quant_bias is not decision.quant_bias:
                evidence_issue = "router quant bias does not match the referenced snapshot"
            if evidence_issue is None and snapshot.produced_at > decision.produced_at:
                evidence_issue = (
                    f"market snapshot {snapshot.message_id} postdates router decision {decision.message_id}"
                )
            if evidence_issue is None:
                evidence_issue = self._freshness_issue(
                    label="market snapshot",
                    produced_at=snapshot.produced_at,
                    reference=now,
                    ttl_s=self.settings.snapshot_ttl_s,
                )
            if evidence_issue is None:
                evidence_issue = self._freshness_issue(
                    label="market snapshot relative to router decision",
                    produced_at=snapshot.produced_at,
                    reference=decision.produced_at,
                    ttl_s=self.settings.snapshot_ttl_s,
                )

            sentiments = []
            if evidence_issue is None:
                requested_sentiment_ids = list(dict.fromkeys(decision.sentiment_ids))
                if len(requested_sentiment_ids) > self.settings.max_sentiment_window:
                    evidence_issue = (
                        f"router referenced {len(requested_sentiment_ids)} sentiment signals; "
                        f"maximum is {self.settings.max_sentiment_window}"
                    )
                else:
                    missing_ids = [
                        message_id
                        for message_id in requested_sentiment_ids
                        if message_id not in self._sentiments_by_id
                    ]
                    unknown_ids = [
                        message_id
                        for message_id in missing_ids
                        if message_id not in self._processed_sentiment
                    ]
                    if unknown_ids:
                        self._defer(
                            decision,
                            envelope,
                            "sentiment evidence is not available: " + ",".join(sorted(unknown_ids)),
                        )
                    if missing_ids:
                        evidence_issue = "sentiment evidence was evicted or invalidated: " + ",".join(
                            sorted(missing_ids)
                        )
                    sentiments = [
                        self._sentiments_by_id[message_id]
                        for message_id in requested_sentiment_ids
                        if message_id in self._sentiments_by_id
                    ]

        if evidence_issue is None:
            if snapshot is None:
                raise RuntimeError("fresh evidence invariant violated: snapshot is unavailable")
            for sentiment in sentiments:
                if sentiment.produced_at > snapshot.produced_at:
                    evidence_issue = (
                        f"sentiment {sentiment.message_id} postdates snapshot {snapshot.message_id}"
                    )
                else:
                    evidence_issue = self._freshness_issue(
                        label=f"sentiment {sentiment.message_id} relative to market snapshot",
                        produced_at=sentiment.produced_at,
                        reference=snapshot.produced_at,
                        ttl_s=self.settings.sentiment_ttl_s,
                    )
                if evidence_issue is not None:
                    break
        if evidence_issue is None and sentiments:
            below_threshold = [
                sentiment.message_id
                for sentiment in sentiments
                if sentiment.confidence < self.settings.sentiment_min_confidence
            ]
            if below_threshold:
                evidence_issue = "router referenced confidence-ineligible sentiment: " + ",".join(
                    sorted(below_threshold)
                )
            else:
                derived_text_bias = calibrated_text_bias(
                    sentiments,
                    deadband=self.settings.sentiment_deadband,
                )
                if derived_text_bias is not decision.text_bias:
                    evidence_issue = (
                        f"router text bias {decision.text_bias.value} does not match "
                        f"calibrated evidence {derived_text_bias.value}"
                    )

        if evidence_issue is not None:
            command = safe_command(
                decision.symbol,
                decision.requested_effort,
                source=self.settings.service_name,
                rationale=f"abstention: {evidence_issue}",
            )
        else:
            if snapshot is None:
                raise RuntimeError("aggregation invariant violated: snapshot is unavailable")
            context = compile_context(
                snapshot,
                sentiments,
                decision,
                system_mode=self.system_mode,
                max_sentiments=self.settings.max_sentiment_window,
            )
            command = await self.brain.decide(
                decision.symbol,
                context,
                decision.requested_effort,
            )

        command = command.model_copy(
            update={
                "message_id": f"aggregator:{decision.message_id}",
                "correlation_id": decision.correlation_id or decision.message_id,
                "causation_id": decision.message_id,
                "reference_price": snapshot.mid_price if snapshot is not None else 0.0,
            }
        )
        self._command_cache[decision.message_id] = command
        self._command_cache.move_to_end(decision.message_id)
        while len(self._command_cache) > self.settings.processed_cache_size:
            self._command_cache.popitem(last=False)

        await self.bus.publish(Topics.TACTICAL_COMMAND, command)
        self._remember(self._processed_router, decision.message_id)
        self._command_cache.pop(decision.message_id, None)
        self._pending_decisions.pop(decision.message_id, None)
        log.info(
            "aggregator.command",
            symbol=command.symbol,
            status=command.status.value,
            reason=command.reason_code.value,
            effort=command.effort_used.value,
            system_mode=self.system_mode.value,
        )

    async def _on_router(self) -> None:
        await self._consume(
            Topics.ROUTER_DECISION,
            consumer="router",
            handler=self._handle_router,
        )

    async def close(self) -> None:
        """Close the gateway and bus together; both get a chance even if one fails."""
        if self._closed:
            return
        self._closed = True
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(self.bus.close(), name="close-bus")
            gateway_close = getattr(self.gateway, "close", None)
            if gateway_close is not None:
                tasks.create_task(gateway_close(), name="close-llm-gateway")

    async def run(self) -> None:  # pragma: no cover - network
        try:
            configure_logging(
                self.settings.log_level,
                json_logs=self.settings.log_json,
                service=self.settings.service_name,
            )
            log.info("aggregator.start")
            async with asyncio.TaskGroup() as tasks:
                tasks.create_task(self._track_snapshots(), name="market-snapshots")
                tasks.create_task(self._track_sentiment(), name="sentiment-signals")
                tasks.create_task(self._track_control(), name="system-control")
                tasks.create_task(self._on_router(), name="router-decisions")
        finally:
            await self.close()


def main() -> None:  # pragma: no cover
    asyncio.run(AggregatorService().run())


if __name__ == "__main__":
    main()
