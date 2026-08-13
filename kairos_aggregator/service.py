"""Aggregator service: router decision -> compile -> LLM -> tactical command."""

from __future__ import annotations

import asyncio
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

from .compiler import compile_context
from .config import AggregatorSettings
from .decider import AggregatorBrain, safe_command

log = get_logger("aggregator")


class _DeferredDecision(RuntimeError):
    """A router decision whose exact source snapshot has not arrived yet."""


class AggregatorService:
    def __init__(self, settings: AggregatorSettings | None = None, *, gateway=None) -> None:
        self.settings = settings or AggregatorSettings()
        self.bus = build_bus(self.settings)
        if gateway is None:
            from kairos_llm import LLMGateway

            gateway = LLMGateway(on_health=self._publish_health)
        self.gateway = gateway
        self.brain = AggregatorBrain(gateway, source=self.settings.service_name)
        self.system_mode = SystemMode.NORMAL
        self._snapshots: dict[str, MarketSnapshot] = {}
        self._snapshots_by_id: OrderedDict[str, MarketSnapshot] = OrderedDict()
        self._pending_decisions: OrderedDict[str, BusEnvelope] = OrderedDict()
        self._processed_router: OrderedDict[str, None] = OrderedDict()
        self._command_cache: OrderedDict[str, TacticalCommand] = OrderedDict()
        self._processed_sentiment: OrderedDict[str, None] = OrderedDict()
        self._sentiments: deque[SentimentSignal] = deque(maxlen=self.settings.max_sentiment_window)
        self._closed = False

    def _remember(self, cache: OrderedDict[str, None], message_id: str) -> None:
        cache[message_id] = None
        cache.move_to_end(message_id)
        while len(cache) > self.settings.processed_cache_size:
            cache.popitem(last=False)

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
        if not self.settings.symbol_allowed(snapshot.symbol):
            log.warning("aggregator.symbol_rejected", symbol=snapshot.symbol)
            return
        self._snapshots[snapshot.symbol] = snapshot
        self._snapshots_by_id[snapshot.message_id] = snapshot
        self._snapshots_by_id.move_to_end(snapshot.message_id)
        while len(self._snapshots_by_id) > self.settings.snapshot_cache_size:
            self._snapshots_by_id.popitem(last=False)

        pending = self._pending_decisions.get(snapshot.message_id)
        if pending is not None:
            await self._handle_router(pending)
            await self.bus.ack(Topics.ROUTER_DECISION, pending, group="aggregator")
            self._pending_decisions.pop(snapshot.message_id, None)

    async def _track_snapshots(self) -> None:
        await self._consume(
            Topics.MARKET_SNAPSHOT,
            consumer="snap",
            handler=self._handle_snapshot,
        )

    async def _handle_sentiment(self, envelope: BusEnvelope) -> None:
        sentiment = SentimentSignal.model_validate(envelope.payload)
        if sentiment.message_id in self._processed_sentiment:
            return
        self._sentiments.append(sentiment)
        self._remember(self._processed_sentiment, sentiment.message_id)

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
        if mode is not previous:
            log.warning("aggregator.mode_change", previous=previous.value, mode=mode.value)

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
        if decision.message_id in self._processed_router:
            return
        if not self.settings.symbol_allowed(decision.symbol):
            log.warning("aggregator.decision_symbol_rejected", symbol=decision.symbol)
            return
        if decision.snapshot_id is None:
            raise ValueError("router decision is missing snapshot_id")
        snapshot = self._snapshots_by_id.get(decision.snapshot_id)
        if snapshot is None:
            self._pending_decisions[decision.snapshot_id] = envelope
            self._pending_decisions.move_to_end(decision.snapshot_id)
            while len(self._pending_decisions) > self.settings.snapshot_cache_size:
                self._pending_decisions.popitem(last=False)
            raise _DeferredDecision(f"snapshot {decision.snapshot_id} is not available for {decision.symbol}")

        cutoff = datetime.now(UTC).timestamp() - self.settings.sentiment_ttl_s
        sentiments = [
            sentiment for sentiment in self._sentiments if sentiment.produced_at.timestamp() >= cutoff
        ]

        command = self._command_cache.get(decision.message_id)
        if command is None:
            context = compile_context(
                snapshot,
                sentiments,
                decision,
                system_mode=self.system_mode,
            )
            if self._must_degrade(decision):
                command = safe_command(
                    decision.symbol,
                    decision.requested_effort,
                    source=self.settings.service_name,
                    rationale=f"system mode {self.system_mode.value}: LLM route suppressed",
                )
            else:
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
                    "reference_price": snapshot.mid_price,
                }
            )
            self._command_cache[decision.message_id] = command
            self._command_cache.move_to_end(decision.message_id)
            while len(self._command_cache) > self.settings.processed_cache_size:
                self._command_cache.popitem(last=False)

        await self.bus.publish(Topics.TACTICAL_COMMAND, command)
        self._remember(self._processed_router, decision.message_id)
        self._command_cache.pop(decision.message_id, None)
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
