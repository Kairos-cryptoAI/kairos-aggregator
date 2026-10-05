"""PAPER/shadow candidate-review runtime, isolated from the legacy tactical route."""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable

from kairos_core import canonical_sha256
from kairos_core.bus import BusEnvelope, MessageBus, build_bus
from kairos_core.contracts import CandidateReviewV1, CandidateRouteV1, LLMHealthEvent
from kairos_core.contracts.decision_context import ContextSourceKind, DecisionContextV1
from kairos_core.enums import CandidateReviewTier, SystemMode
from kairos_core.logging import configure_logging, get_logger
from kairos_core.topics import Topics
from kairos_persistence import DurableLLMUsageBudget, DurableMessageBus

from .candidate_review import CandidateReviewBrain, deterministic_defer
from .config import AggregatorSettings
from .decision_context import CandidateContextStore

log = get_logger("candidate-review")

OPENAI_SHADOW_BUDGET_MICROUSD = 12_000_000
DEEPSEEK_SHADOW_BUDGET_MICROUSD = 1_000_000


class CandidateReviewService:
    """Freeze causal evidence before model dispatch; never invent a trading intent."""

    def __init__(
        self,
        settings: AggregatorSettings | None = None,
        *,
        gateway=None,
        bus: MessageBus | None = None,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self.settings = settings or AggregatorSettings()
        if bus is not None:
            self.bus = bus
        else:
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
        self.context_store = CandidateContextStore(self.settings)
        self._route_digests: OrderedDict[str, str] = OrderedDict()
        self._route_messages: OrderedDict[str, str] = OrderedDict()
        self._route_cutoffs: OrderedDict[str, int] = OrderedDict()
        self._context_cache: OrderedDict[str, DecisionContextV1] = OrderedDict()
        self._processed_routes: OrderedDict[str, None] = OrderedDict()
        self._review_cache: OrderedDict[str, CandidateReviewV1] = OrderedDict()
        self._route_lock = asyncio.Lock()
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

    def _receive_source(self, kind: ContextSourceKind, envelope: BusEnvelope, expected_topic: str) -> None:
        if envelope.topic != expected_topic:
            raise ValueError("context source arrived on the wrong topic")
        # Envelope metadata and producer timestamps cannot attest past availability.
        self.context_store.ingest(kind, envelope.payload, received_at_ms=self._clock_ms())

    async def _handle_sentiment(self, envelope: BusEnvelope) -> None:
        self._receive_source("text", envelope, Topics.SENTIMENT_SIGNAL)

    async def _handle_market(self, envelope: BusEnvelope) -> None:
        self._receive_source("market", envelope, Topics.MARKET_SNAPSHOT)

    async def _handle_closed_bar(self, envelope: BusEnvelope) -> None:
        self._receive_source("closed_bars", envelope, Topics.CLOSED_BAR)

    async def _handle_macro(self, envelope: BusEnvelope) -> None:
        self._receive_source("macro", envelope, Topics.STRATEGIC_ALLOCATION)

    async def _handle_control(self, envelope: BusEnvelope) -> None:
        mode_value = envelope.payload.get("mode")
        if not isinstance(mode_value, str):
            raise ValueError("system control mode must be a string")
        self.system_mode = SystemMode(mode_value)

    async def _handle_route(self, envelope: BusEnvelope) -> None:
        received_at_ms = self._clock_ms()
        async with self._route_lock:
            await self._review_route(envelope, received_at_ms=received_at_ms)

    async def _review_route(self, envelope: BusEnvelope, *, received_at_ms: int) -> None:
        if envelope.topic != Topics.STRATEGY_ROUTE:
            raise ValueError("candidate route arrived on the wrong topic")
        route = CandidateRouteV1.model_validate(envelope.payload)
        route_id = route.route_id
        if route_id is None:  # impossible after strict contract validation
            raise ValueError("candidate route has no canonical identity")
        digest = canonical_sha256(route)
        for existing in (self._route_digests.get(route_id), self._route_messages.get(route.message_id)):
            if existing is not None and existing != digest:
                raise ValueError("candidate route identity was reused with conflicting content")
        self._remember(self._route_digests, route_id, digest)
        self._remember(self._route_messages, route.message_id, digest)
        if route_id in self._processed_routes:
            return

        context = self._context_cache.get(route_id)
        cached = self._review_cache.get(route_id)
        if cached is not None:
            if context is None and any(item.kind == "decision_context" for item in cached.evidence):
                raise ValueError("cached review lost its original immutable context")
            if context is not None and self.context_store.integrity_error(context):
                raise ValueError("cached review context has a conflicting source identity")
            if context is not None:
                await self.bus.publish(Topics.DECISION_CONTEXT, context)
            await self.bus.publish(Topics.CANDIDATE_REVIEW, cached)
            self._remember(self._processed_routes, route_id, None)
            self._review_cache.pop(route_id, None)
            return

        captured_at_ms = self._clock_ms()
        reason: str | None = None
        if not self.settings.symbol_allowed(route.intent.symbol):
            reason = "SYMBOL_NOT_ALLOWED"
        elif self.system_mode is SystemMode.LOCAL_QUANT_MODE:
            reason = "LLM_ROUTE_DISABLED"
        elif (
            self.system_mode is SystemMode.CONFLICT_SAFE and route.review_tier is CandidateReviewTier.CONFLICT
        ):
            reason = "CONFLICT_REVIEW_DISABLED"
        elif received_at_ms < route.routed_at_ms:
            reason = "ROUTE_FROM_FUTURE"
        elif captured_at_ms > route.review_deadline_ms:
            reason = "REVIEW_DEADLINE_EXCEEDED"

        if reason is None and context is None:
            if route_id in self._route_cutoffs:
                # Cache eviction cannot silently create a new evidence cut on retry.
                reason = "DECISION_CONTEXT_EVICTED"
            else:
                self._remember(self._route_cutoffs, route_id, received_at_ms)
                try:
                    context = self.context_store.build(
                        route,
                        cutoff_ms=received_at_ms,
                        captured_at_ms=captured_at_ms,
                    )
                except (ValueError, KeyError, TypeError):
                    reason = "DECISION_CONTEXT_INVALID"
                else:
                    self._remember(self._context_cache, route_id, context)

        if context is not None:
            # An unsuccessful context publication must never spend model budget.
            # The exact frozen context survives a publish retry; late evidence cannot upgrade it.
            await self.bus.publish(Topics.DECISION_CONTEXT, context)
            if self.context_store.integrity_error(context):
                reason = "DECISION_CONTEXT_IDENTITY_CONFLICT"
        review = (
            deterministic_defer(
                route,
                reviewed_at_ms=received_at_ms,
                reason_code=reason,
                source=self.settings.service_name,
                decision_context=context,
            )
            if reason is not None
            else await self.brain.review_with_context(route, context)
        )
        if context is not None and self.context_store.integrity_error(context):
            review = deterministic_defer(
                route,
                reviewed_at_ms=self._clock_ms(),
                reason_code="DECISION_CONTEXT_IDENTITY_CONFLICT",
                source=self.settings.service_name,
                decision_context=context,
            )
        if review.intent.model_dump(mode="json") != route.intent.model_dump(mode="json"):
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

    async def _track_market(self) -> None:
        await self._consume(Topics.MARKET_SNAPSHOT, consumer="market", handler=self._handle_market)

    async def _track_closed_bars(self) -> None:
        await self._consume(Topics.CLOSED_BAR, consumer="closed-bars", handler=self._handle_closed_bar)

    async def _track_macro(self) -> None:
        await self._consume(Topics.STRATEGIC_ALLOCATION, consumer="macro", handler=self._handle_macro)

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
                tasks.create_task(self._track_market(), name="candidate-market")
                tasks.create_task(self._track_closed_bars(), name="candidate-closed-bars")
                tasks.create_task(self._track_macro(), name="candidate-macro")
                tasks.create_task(self._track_control(), name="candidate-control")
                tasks.create_task(self._review_routes(), name="candidate-routes")
        finally:
            await self.close()


def main() -> None:  # pragma: no cover
    asyncio.run(CandidateReviewService().run())


if __name__ == "__main__":
    main()
