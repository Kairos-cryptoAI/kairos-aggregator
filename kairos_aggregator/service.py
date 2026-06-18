"""Aggregator service: router decision -> compile -> LLM -> tactical command."""
from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from typing import Deque, Dict

from kairos_core.bus import build_bus
from kairos_core.contracts import MarketSnapshot, RouterDecision, SentimentSignal
from kairos_core.logging import configure_logging, get_logger
from kairos_core.topics import Topics

from .compiler import compile_context
from .config import AggregatorSettings
from .decider import AggregatorBrain

log = get_logger("aggregator")


class AggregatorService:
    def __init__(self, settings: AggregatorSettings | None = None, *, gateway=None) -> None:
        self.settings = settings or AggregatorSettings()
        self.bus = build_bus(self.settings)
        if gateway is None:
            from kairos_llm import LLMGateway
            gateway = LLMGateway()
        self.brain = AggregatorBrain(gateway, source=self.settings.service_name)
        self._snapshots: Dict[str, MarketSnapshot] = {}
        self._sentiments: Deque[SentimentSignal] = deque(maxlen=self.settings.max_sentiment_window)

    async def _track_snapshots(self) -> None:
        async for env in self.bus.subscribe(Topics.MARKET_SNAPSHOT, group="aggregator", consumer="snap"):
            try:
                snap = MarketSnapshot.model_validate(env.payload)
                self._snapshots[snap.symbol] = snap
            finally:
                await self.bus.ack(Topics.MARKET_SNAPSHOT, env, group="aggregator")

    async def _track_sentiment(self) -> None:
        async for env in self.bus.subscribe(Topics.SENTIMENT_SIGNAL, group="aggregator", consumer="sent"):
            try:
                self._sentiments.append(SentimentSignal.model_validate(env.payload))
            finally:
                await self.bus.ack(Topics.SENTIMENT_SIGNAL, env, group="aggregator")

    async def _on_router(self) -> None:
        async for env in self.bus.subscribe(Topics.ROUTER_DECISION, group="aggregator", consumer="router"):
            try:
                decision = RouterDecision.model_validate(env.payload)
                snap = self._snapshots.get(decision.symbol)
                if snap is None:
                    continue
                ctx = compile_context(snap, list(self._sentiments), decision)
                cmd = await self.brain.decide(decision.symbol, ctx, decision.requested_effort)
                await self.bus.publish(Topics.TACTICAL_COMMAND, cmd)
                log.info("aggregator.command", symbol=cmd.symbol, status=cmd.status.value,
                        reason=cmd.reason_code.value, effort=cmd.effort_used.value)
            finally:
                await self.bus.ack(Topics.ROUTER_DECISION, env, group="aggregator")

    async def run(self) -> None:  # pragma: no cover - network
        configure_logging(self.settings.log_level, json_logs=self.settings.log_json, service=self.settings.service_name)
        log.info("aggregator.start")
        await asyncio.gather(self._track_snapshots(), self._track_sentiment(), self._on_router())


def main() -> None:  # pragma: no cover
    asyncio.run(AggregatorService().run())


if __name__ == "__main__":
    main()
