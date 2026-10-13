"""Offline V2 engineering qualification against the real candidate service boundary.

All source records and model responses are synthetic. A PASS means only that this
local harness exercised the current DecisionContext path; it is not source
authenticity evidence, paid/live qualification, production A/B, or trading approval.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

from kairos_core.bus import BusEnvelope
from kairos_core.config import DEFAULT_TRADING_SYMBOLS
from kairos_core.contracts import (
    CandidateRouteV1,
    ClosedBarEventV1,
    DerivativesMetrics,
    ExitPlanV1,
    MarketSnapshot,
    OrderBookSummary,
    SentimentSignal,
    StrategicAllocation,
    StrategyIntentV1,
    StrategyProvenanceV1,
    TechnicalIndicators,
)
from kairos_core.contracts.base import datetime_from_unix_ms
from kairos_core.enums import (
    CandidateReviewTier,
    ImpactDirection,
    MarketRegime,
    ReviewDecision,
    Side,
    StrategicTrigger,
)
from kairos_core.topics import Topics
from kairos_llm import LLMResult, TokenUsage
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .candidate_review import CandidateReviewOutput
from .candidate_service import CandidateReviewService
from .config import AggregatorSettings

CORPUS_RESOURCE = "candidate_review_v2_context.json"
ROUTED_AT_MS = 1_900_000_019_999  # exact close of a UTC 1m bar
NOW_MS = ROUTED_AT_MS + 1_000
DEADLINE_MS = ROUTED_AT_MS + 30_000
MAX_CASE_COST_USD = 0.10


class ContextCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(pattern=r"^[a-z0-9_]+$")
    category: Literal[
        "normal_allow",
        "conflict_veto",
        "conflict_allow_guard",
        "prompt_injection",
        "forbidden",
        "stale_text",
        "future_text",
        "missing_market",
        "missing_macro",
        "missing_bar",
        "missing_text",
    ]
    symbol: str = Field(pattern=r"^[A-Z0-9]+USDT$")
    review_tier: CandidateReviewTier
    sentiment: float = Field(ge=-1, le=1)
    summary: str = Field(min_length=1, max_length=500)
    model_decision: ReviewDecision
    expected_decisions: tuple[ReviewDecision, ...] = Field(min_length=1)
    expected_reviewer: Literal["LLM", "DETERMINISTIC"]
    expected_reason_contains: str | None = None
    fault: Literal[
        "none",
        "forbidden",
        "stale_text",
        "future_text",
        "missing_market",
        "missing_macro",
        "missing_bar",
        "missing_text",
    ]

    @model_validator(mode="after")
    def validate_case(self) -> ContextCase:
        if self.review_tier is CandidateReviewTier.CONFLICT and self.category not in {
            "conflict_veto",
            "conflict_allow_guard",
        }:
            raise ValueError("only conflict cases may use the conflict workload")
        if self.expected_reviewer == "LLM" and self.fault != "none":
            raise ValueError("only complete, fresh causal cases may call the model")
        if self.expected_reviewer == "DETERMINISTIC" and not self.expected_reason_contains:
            raise ValueError("deterministic cases require an expected fail-closed reason")
        return self


class ContextCorpus(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[2]
    qualification_kind: Literal["synthetic_local_engineering_only"]
    cases: tuple[ContextCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_corpus(self) -> ContextCorpus:
        ids = [case.case_id for case in self.cases]
        required = {
            "normal_allow",
            "conflict_veto",
            "prompt_injection",
            "forbidden",
            "stale_text",
            "future_text",
            "missing_market",
            "missing_macro",
            "missing_bar",
            "missing_text",
            "conflict_allow_guard",
        }
        if len(ids) != len(set(ids)):
            raise ValueError("case IDs must be unique")
        if not required.issubset({case.category for case in self.cases}):
            raise ValueError("V2 corpus is missing a required safety category")
        return self


@dataclass(frozen=True)
class ContextObservation:
    case_id: str
    category: str
    passed: bool
    decision: str | None
    reviewer: str | None
    reason_codes: tuple[str, ...]
    model_called: bool
    context_complete_at_model_call: bool
    intent_preserved: bool
    deadline_met: bool
    provenance_complete: bool
    model_call_count: int
    cost_usd: float
    note: str | None


@dataclass(frozen=True)
class ContextQualificationReport:
    schema_version: int
    qualification_kind: str
    generated_at: str
    corpus_sha256: str
    status: str
    observations: tuple[ContextObservation, ...]
    actual_cost_usd: float
    paid_qualification: bool = False
    source_authenticity_verified: bool = False
    production_ab: bool = False
    live_orders_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "qualification_kind": self.qualification_kind,
            "generated_at": self.generated_at,
            "corpus_sha256": self.corpus_sha256,
            "status": self.status,
            "observations": [asdict(item) for item in self.observations],
            "actual_cost_usd": self.actual_cost_usd,
            "paid_qualification": False,
            "source_authenticity_verified": False,
            "production_ab": False,
            "live_orders_allowed": False,
        }


class _CaptureBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, Any]] = []

    async def publish(self, topic: str, message: Any) -> str:
        self.published.append((topic, message))
        return message.message_id

    async def close(self) -> None:
        return None


class _SyntheticLocalGateway:
    """Explicit, offline, zero-cost gateway; never contacts a provider."""

    def __init__(self, corpus: ContextCorpus) -> None:
        self.calls: list[str] = []
        self.decisions = {case.case_id: case.model_decision for case in corpus.cases}

    async def complete(self, **kwargs: Any) -> LLMResult:
        payload = json.loads(kwargs["user"])
        context = payload["decision_context"]
        intent = json.loads(context["intent_identity_json"])
        case_id = intent["strategy_id"].removeprefix("qualification_v2_")
        self.calls.append(case_id)
        # The local oracle's behavior is explicit per case, not derived from untrusted text.
        decision = self.decisions[case_id].value
        reason = {
            "ALLOW": "SYNTHETIC_SUPPORT",
            "VETO": "SYNTHETIC_INVALIDATION",
            "DEFER": "UNTRUSTED_INSTRUCTIONS",
        }[decision]
        priority = {"ALLOW": 70, "VETO": 90, "DEFER": 0}[decision]
        parsed = CandidateReviewOutput.model_validate(
            {"decision": decision, "priority": priority, "reason_codes": [reason]}
        )
        return LLMResult(
            content=parsed.model_dump_json(),
            parsed=parsed,
            model="synthetic-local-gateway",
            effort="none",
            usage=TokenUsage(input_tokens=1, output_tokens=1),
            cost_usd=0.0,
            latency_s=0.001,
            provider="synthetic-local",
            request_id=f"synthetic:{case_id}",
            resolved_model="synthetic-local-v2",
            budget_reservation_id=f"synthetic:not-billable:{case_id}",
        )

    async def close(self) -> None:
        return None


def load_context_corpus() -> tuple[ContextCorpus, str]:
    raw = files("kairos_aggregator.corpora").joinpath(CORPUS_RESOURCE).read_bytes()
    return ContextCorpus.model_validate_json(raw), hashlib.sha256(raw).hexdigest()


def _materialize(case: ContextCase, index: int) -> tuple[CandidateRouteV1, dict[str, Any]]:
    # Keep every case's clocks inside a narrow deterministic window while ensuring
    # its one exact declared bar closes at decision_ts_ms.
    routed_at = ROUTED_AT_MS + index * 60_000
    now = routed_at + 1_000
    symbol = case.symbol
    bar_open = routed_at - 59_999
    bar = ClosedBarEventV1(
        source="qualification-v2:synthetic-bars",
        symbol=symbol,
        open_time_ms=bar_open,
        close_time_ms=routed_at,
        open=100,
        high=101,
        low=99,
        close=100.5,
        base_volume=10,
        quote_volume=1_000,
        taker_buy_base_volume=5,
        taker_buy_quote_volume=500,
        produced_at=datetime_from_unix_ms(routed_at),
    )
    market = MarketSnapshot(
        source="qualification-v2:synthetic-market",
        message_id=f"market-{index}",
        produced_at=datetime_from_unix_ms(routed_at - 2_000),
        symbol=symbol,
        mid_price=100.5,
        volume_usd=1_000_000,
        order_book=OrderBookSummary(
            best_bid=100.4,
            best_ask=100.6,
            spread_bps=19.9,
            imbalance=0.1,
            depth_usd=500_000,
        ),
        derivatives=DerivativesMetrics(funding_rate=0.0001, open_interest=1_000_000),
        indicators=TechnicalIndicators(rsi_14=55, macd=0.2, macd_signal=0.1, macd_hist=0.1, atr_pct=0.01),
        quant_bias=Side.LONG,
    )
    event_at = routed_at - 1_000
    if case.fault == "stale_text":
        event_at -= 601_000
    if case.fault == "future_text":
        event_at = routed_at + 1_000
    sentiment = SentimentSignal(
        source="qualification-v2:synthetic-text",
        message_id=f"text-{index}",
        produced_at=datetime_from_unix_ms(event_at),
        topic=symbol,
        sentiment=case.sentiment,
        impact=ImpactDirection.BULLISH if case.sentiment >= 0 else ImpactDirection.BEARISH,
        confidence=0.95,
        sources=["synthetic://local-fixture"],
        summary=case.summary,
    )
    macro = StrategicAllocation(
        source="qualification-v2:synthetic-macro",
        message_id=f"macro-{index}",
        produced_at=datetime_from_unix_ms(routed_at - 3_000),
        regime=MarketRegime.BULL,
        stable_reserve_pct=0.2,
        strategy_weights={"synthetic-trend": 0.5},
        max_gross_leverage=2,
        triggered_by=StrategicTrigger.SCHEDULE,
        rationale="synthetic local qualification evidence",
    )
    side = Side.LONG
    entry = ((routed_at // 60_000) + 1) * 60_000
    intent = StrategyIntentV1(
        source="strategy-engine:qualification-v2",
        strategy_id=f"qualification_v2_{case.case_id}",
        strategy_revision="synthetic-context-v2",
        symbol=symbol,
        side=side,
        decision_ts_ms=routed_at,
        entry_eligible_ts_ms=entry,
        entry_expires_ts_ms=entry + 60_000,
        reference_price=100,
        signal_strength=0.8,
        gross_reward_bps=500,
        exit_plan=ExitPlanV1(stop_price=95, target_price=105, max_holding_ms=180_000),
        provenance=StrategyProvenanceV1(
            strategy_code_sha256="a" * 64,
            config_sha256="b" * 64,
            input_window_sha256="c" * 64,
            features_sha256="d" * 64,
            input_bar_sha256s=(bar.bar_sha256,),
        ),
    )
    route = CandidateRouteV1(
        source="router:qualification-v2",
        intent=intent,
        review_tier=case.review_tier,
        requested_reasoning_effort=("high" if case.review_tier is CandidateReviewTier.CONFLICT else "medium"),
        routed_at_ms=routed_at,
        review_deadline_ms=routed_at + 30_000,
        evidence_ids=(sentiment.message_id,),
        conflict_rationale=(
            "synthetic official evidence conflicts"
            if case.review_tier is CandidateReviewTier.CONFLICT
            else None
        ),
    )
    return route, {"now": now, "bar": bar, "market": market, "text": sentiment, "macro": macro}


async def qualify_context_corpus(corpus: ContextCorpus, corpus_sha256: str) -> ContextQualificationReport:
    gateway = _SyntheticLocalGateway(corpus)
    observations: list[ContextObservation] = []
    for index, case in enumerate(corpus.cases):
        route, sources = _materialize(case, index)
        original_intent = route.intent.model_dump(mode="json")
        # model_construct avoids consulting process environment or .env while retaining
        # validated service contract defaults and an explicitly deterministic universe.
        settings = AggregatorSettings.model_construct(
            bus_backend="memory",
            service_name="kairos-aggregator-qualification-v2",
            trading_symbols=DEFAULT_TRADING_SYMBOLS.copy(),
            candidate_require_macro_context=True,
            candidate_require_text_context=True,
        )
        service = CandidateReviewService(
            settings,
            gateway=gateway,
            clock_ms=lambda now=sources["now"]: now,
        )
        capture = _CaptureBus()
        service.bus = capture  # type: ignore[assignment]
        omitted = {
            "forbidden": {"market", "bar", "text", "macro"},
            "missing_market": {"market"},
            "missing_macro": {"macro"},
            "missing_bar": {"bar"},
            "missing_text": {"text"},
        }.get(case.fault, set())
        for name, handler, topic in (
            ("market", service._handle_market, Topics.MARKET_SNAPSHOT),
            ("bar", service._handle_closed_bar, Topics.CLOSED_BAR),
            ("text", service._handle_sentiment, Topics.SENTIMENT_SIGNAL),
            ("macro", service._handle_macro, Topics.STRATEGIC_ALLOCATION),
        ):
            if name in omitted:
                continue
            message = sources[name]
            try:
                await handler(
                    BusEnvelope(id=f"{case.case_id}:{name}", topic=topic, payload=message.to_payload())
                )
            except (TypeError, ValueError):
                # Future source timestamp is rejected at ingestion, leaving an explicit
                # unavailable slot in the context; the service then deterministically defers.
                if case.fault != "future_text" or name != "text":
                    raise
        before = len(gateway.calls)
        await service._handle_route(
            BusEnvelope(id=f"route:{case.case_id}", topic=Topics.STRATEGY_ROUTE, payload=route.to_payload())
        )
        review = next(message for topic, message in capture.published if topic == Topics.CANDIDATE_REVIEW)
        context = next(
            (message for topic, message in capture.published if topic == Topics.DECISION_CONTEXT), None
        )
        call_count = len(gateway.calls) - before
        called = call_count > 0
        complete = bool(
            context
            and all(
                source.availability == "AVAILABLE"
                for source in context.sources
                if source.kind in context.required_sources
            )
        )
        provenance = review.model_provenance
        provenance_ok = not called or bool(
            provenance
            and provenance.provider
            and provenance.model
            and provenance.request_id
            and provenance.prompt_sha256
            and provenance.response_sha256
            and provenance.budget_reservation_id
            and provenance.cost_usd == 0.0
        )
        reasons = review.reason_codes
        reason_ok = case.expected_reason_contains is None or any(
            case.expected_reason_contains in code for code in reasons
        )
        passed = (
            review.decision in case.expected_decisions
            and review.reviewer == case.expected_reviewer
            and (called == (case.expected_reviewer == "LLM"))
            and (case.expected_reviewer != "LLM" or complete)
            and reason_ok
            and provenance_ok
            and call_count <= 1
            and review.intent.model_dump(mode="json") == original_intent
            and review.reviewed_at_ms <= route.review_deadline_ms
        )
        observations.append(
            ContextObservation(
                case_id=case.case_id,
                category=case.category,
                passed=passed,
                decision=review.decision.value,
                reviewer=review.reviewer,
                reason_codes=reasons,
                model_called=called,
                context_complete_at_model_call=complete,
                intent_preserved=review.intent.model_dump(mode="json") == original_intent,
                deadline_met=review.reviewed_at_ms <= route.review_deadline_ms,
                provenance_complete=provenance_ok,
                model_call_count=call_count,
                cost_usd=provenance.cost_usd if provenance is not None else 0.0,
                note=None if passed else "expectation_or_invariant_mismatch",
            )
        )
    total_cost = math.fsum(item.cost_usd for item in observations)
    passed = all(item.passed for item in observations) and total_cost <= MAX_CASE_COST_USD
    return ContextQualificationReport(
        schema_version=2,
        qualification_kind=corpus.qualification_kind,
        generated_at=datetime.now(UTC).isoformat(),
        corpus_sha256=corpus_sha256,
        status="PASS" if passed else "FAIL",
        observations=tuple(observations),
        actual_cost_usd=total_cost,
    )


def _write_report(path: Path, report: ContextQualificationReport, overwrite: bool) -> None:
    resolved = path.resolve()
    if resolved.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite qualification report: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"
    descriptor, name = tempfile.mkstemp(prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        corpus, digest = load_context_corpus()
        report = asyncio.run(qualify_context_corpus(corpus, digest))
        _write_report(args.output, report, args.overwrite)
    except (OSError, ValueError) as exc:
        print(f"context qualification failed: {exc}")
        return 2
    print(f"Context qualification V2: {report.status}; synthetic/offline; live_orders_allowed=false")
    return 0 if report.status == "PASS" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
