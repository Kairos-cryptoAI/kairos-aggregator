"""PAPER/shadow candidate-review runtime, isolated from the legacy tactical route."""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable

from kairos_core import canonical_sha256
from kairos_core.bus import BusEnvelope, build_bus
from kairos_core.contracts import CandidateReviewV1, CandidateRouteV1, LLMHealthEvent, SentimentSignal
from kairos_core.enums import CandidateReviewTier, SystemMode
from kairos_core.logging import configure_logging, get_logger
from kairos_core.topics import Topics
from kairos_persistence import DurableLLMUsageBudget, DurableMessageBus

from .candidate_review import CandidateReviewBrain, deterministic_defer
from .config import AggregatorSettings

log = get_logger("candidate-review")

OPENAI_SHADOW_BUDGET_MICROUSD = 12_000_000
DEEPSEEK_SHADOW_BUDGET_MICROUSD = 1_000_000


class CandidateReviewService:
    """Consume immutable routes and emit one terminal review for each intent."""

    def __init__(
        self,
        settings: AggregatorSettings | None = None,
        *,
        gateway=None,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self.settings = settings or AggregatorSettings()
        transport = build_bus(self.settings)
        self.bus = (
            transport
            if self.settings.bus_backend == "memory"
            else DurableMessageBus(transport, service_name=f"{self.settings.service_name}-candidate")
        )
        if gateway is None:
            from kairos_llm import (
                BudgetedLLMGateway,
                DenyLLMUsageBudget,
                LLMGateway,
                LLMSettings,
                Provider,
            )

            budget = (
                DurableLLMUsageBudget(self.bus)
                if isinstance(self.bus, DurableMessageBus)
                else DenyLLMUsageBudget()
            )
            gateway = BudgetedLLMGateway(
                LLMGateway(settings=LLMSettings(max_retries=0), on_health=self._publish_health),
                budget,
                monthly_budgets_microusd={
                    Provider.OPENAI: OPENAI_SHADOW_BUDGET_MICROUSD,
                    Provider.DEEPSEEK: DEEPSEEK_SHADOW_BUDGET_MICROUSD,
                },
            )
        self.gateway = gateway
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1_000))
        self.brain = CandidateReviewBrain(
            gateway,
            source=self.settings.service_name,
            clock_ms=self._clock_ms,
        )
        self.system_mode = SystemMode.NORMAL
        self._sentiments: OrderedDict[str, SentimentSignal] = OrderedDict()
        self._sentiment_digests: OrderedDict[str, str] = OrderedDict()
        self._processed_routes: OrderedDict[str, None] = OrderedDict()
        self._review_cache: OrderedDict[str, CandidateReviewV1] = OrderedDict()
        self._closed = False

    def _remember(self, cache: OrderedDict, key: str, value) -> None:
        cache[key] = value
        cache.move_to_end(key)
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
        async for envelope in self.bus.subscribe(topic, group="candidate-review", consumer=consumer):
            try:
                await handler(envelope)
                await self.bus.ack(topic, envelope, group="candidate-review")
            except Exception:
                log.exception(
                    "candidate_review.message_processing_failed",
                    topic=topic,
                    envelope_id=envelope.id,
                    attempt=envelope.attempt,
                )

    async def _handle_sentiment(self, envelope: BusEnvelope) -> None:
        signal = SentimentSignal.model_validate(envelope.payload)
        observed_at_ms = int(signal.produced_at.timestamp() * 1_000)
        if observed_at_ms > self._clock_ms() + int(self.settings.max_future_skew_s * 1_000):
            log.warning("candidate_review.sentiment_from_future", message_id=signal.message_id)
            return
        digest = canonical_sha256(signal)
        existing = self._sentiment_digests.get(signal.message_id)
        if existing is not None:
            if existing != digest:
                raise ValueError(f"sentiment message_id {signal.message_id!r} was reused")
            return
        self._remember(self._sentiments, signal.message_id, signal)
        self._remember(self._sentiment_digests, signal.message_id, digest)

    async def _handle_control(self, envelope: BusEnvelope) -> None:
        mode_value = envelope.payload.get("mode")
        if not isinstance(mode_value, str):
            raise ValueError("system control mode must be a string")
        self.system_mode = SystemMode(mode_value)

    async def _handle_route(self, envelope: BusEnvelope) -> None:
        route = CandidateRouteV1.model_validate(envelope.payload)
        route_id = route.route_id
        if route_id is None:  # impossible after strict contract validation
            raise ValueError("candidate route has no canonical identity")
        if route_id in self._processed_routes:
            return

        cached = self._review_cache.get(route_id)
        if cached is not None:
            await self.bus.publish(Topics.CANDIDATE_REVIEW, cached)
            self._remember(self._processed_routes, route_id, None)
            self._review_cache.pop(route_id, None)
            return

        selected = tuple(
            self._sentiments[message_id]
            for message_id in route.evidence_ids
            if message_id in self._sentiments
        )
        missing_ids = tuple(
            message_id for message_id in route.evidence_ids if message_id not in self._sentiments
        )
        reason: str | None = None
        if not self.settings.symbol_allowed(route.intent.symbol):
            reason = "SYMBOL_NOT_ALLOWED"
        elif self.system_mode is SystemMode.LOCAL_QUANT_MODE:
            reason = "LLM_ROUTE_DISABLED"
        elif (
            self.system_mode is SystemMode.CONFLICT_SAFE and route.review_tier is CandidateReviewTier.CONFLICT
        ):
            reason = "CONFLICT_REVIEW_DISABLED"
        elif missing_ids:
            reason = "EVIDENCE_UNAVAILABLE"
        else:
            routed_at_ms = route.routed_at_ms
            for signal in selected:
                observed_at_ms = int(signal.produced_at.timestamp() * 1_000)
                if observed_at_ms > routed_at_ms:
                    reason = "EVIDENCE_POSTDATES_ROUTE"
                    break
                if routed_at_ms - observed_at_ms > int(self.settings.sentiment_ttl_s * 1_000):
                    reason = "EVIDENCE_STALE"
                    break

        review = (
            deterministic_defer(
                route,
                reviewed_at_ms=self._clock_ms(),
                reason_code=reason,
                source=self.settings.service_name,
                evidence=selected,
            )
            if reason is not None
            else await self.brain.review(route, selected)
        )
        if review.intent.intent_id != route.intent.intent_id:
            raise ValueError("candidate review mutated the immutable strategy intent")
        self._remember(self._review_cache, route_id, review)
        await self.bus.publish(Topics.CANDIDATE_REVIEW, review)
        self._remember(self._processed_routes, route_id, None)
        self._review_cache.pop(route_id, None)
        log.info(
            "candidate_review.completed",
            route_id=route_id,
            intent_id=route.intent.intent_id,
            decision=review.decision.value,
            priority=review.priority,
            reviewer=review.reviewer,
        )

    async def _track_sentiments(self) -> None:
        await self._consume(
            Topics.SENTIMENT_SIGNAL,
            consumer="sentiment",
            handler=self._handle_sentiment,
        )

    async def _track_control(self) -> None:
        await self._consume(
            Topics.SYSTEM_CONTROL,
            consumer="control",
            handler=self._handle_control,
        )

    async def _review_routes(self) -> None:
        await self._consume(
            Topics.STRATEGY_ROUTE,
            consumer="route",
            handler=self._handle_route,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(self.bus.close(), name="close-candidate-bus")
            gateway_close = getattr(self.gateway, "close", None)
            if gateway_close is not None:
                tasks.create_task(gateway_close(), name="close-candidate-gateway")

    async def run(self) -> None:  # pragma: no cover - requires services
        try:
            configure_logging(
                self.settings.log_level,
                json_logs=self.settings.log_json,
                service=f"{self.settings.service_name}-candidate",
            )
            log.info("candidate_review.start")
            async with asyncio.TaskGroup() as tasks:
                tasks.create_task(self._track_sentiments(), name="candidate-sentiments")
                tasks.create_task(self._track_control(), name="candidate-control")
                tasks.create_task(self._review_routes(), name="candidate-routes")
        finally:
            await self.close()


def main() -> None:  # pragma: no cover
    asyncio.run(CandidateReviewService().run())


if __name__ == "__main__":
    main()
