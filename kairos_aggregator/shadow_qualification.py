"""Frozen-corpus safety and quality gate for immutable candidate reviews.

The harness is shadow-only: it never publishes a strategy candidate, never places an
order and always reports ``live_orders_allowed=false``. Live provider runs use the same
durable provider-wide budget ledger as runtime calls.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

from kairos_core.bus import BusEnvelope, build_bus
from kairos_core.contracts import (
    CandidateRouteV1,
    ExitPlanV1,
    SentimentSignal,
    StrategyIntentV1,
    StrategyProvenanceV1,
)
from kairos_core.enums import (
    CandidateReviewTier,
    ImpactDirection,
    ReviewDecision,
    Side,
)
from kairos_core.topics import Topics
from kairos_llm import (
    BudgetedLLMGateway,
    LLMGateway,
    LLMResult,
    LLMSettings,
    Provider,
    TokenUsage,
)
from kairos_llm.pricing import PriceTable
from kairos_persistence import DurableLLMUsageBudget, DurableMessageBus, PersistenceSettings
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .candidate_review import (
    CANDIDATE_REVIEW_SYSTEM,
    CandidateReviewOutput,
    compile_candidate_context,
)
from .candidate_service import (
    DEEPSEEK_SHADOW_BUDGET_MICROUSD,
    OPENAI_SHADOW_BUDGET_MICROUSD,
    CandidateReviewService,
)
from .config import AggregatorSettings

DEFAULT_CORPUS_RESOURCE = "candidate_review_v1.json"
DEFAULT_MAXIMUM_PLANNED_COST_USD = 0.10
HARD_MAXIMUM_PLANNED_COST_USD = 0.25
QUALIFICATION_MAX_OUTPUT_TOKENS = 1_024
CASE_DEADLINE_MS = 30_000
BASE_AGE_MS = 3_000


class QualificationStatus(StrEnum):
    PASS = "PASS"  # nosec B105
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"


class CorpusEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1, max_length=128)
    offset_ms: int
    sentiment: float = Field(ge=-1.0, le=1.0)
    impact: ImpactDirection
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=1, max_length=500)
    sources: tuple[str, ...] = Field(min_length=1, max_length=3)


class CandidateCorpusCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(pattern=r"^[a-z0-9_]+$", min_length=1, max_length=80)
    category: Literal[
        "normal",
        "conflict",
        "prompt_injection",
        "deterministic_forbidden",
        "stale",
        "missing",
        "post_route",
    ]
    review_tier: CandidateReviewTier
    symbol: str = Field(pattern=r"^[A-Z0-9]+USDT$")
    evidence: tuple[CorpusEvidence, ...] = ()
    requested_evidence_ids: tuple[str, ...] | None = None
    expected_decisions: tuple[ReviewDecision, ...] = Field(min_length=1)
    expected_reviewer: Literal["LLM", "DETERMINISTIC"]
    expected_reason_code: str | None = None
    expected_model_call: bool

    @field_validator("expected_decisions")
    @classmethod
    def unique_decisions(cls, value: tuple[ReviewDecision, ...]) -> tuple[ReviewDecision, ...]:
        if len(value) != len(set(value)):
            raise ValueError("expected decisions must be unique")
        return value

    @model_validator(mode="after")
    def consistent_expectations(self) -> CandidateCorpusCase:
        if self.expected_model_call != (self.expected_reviewer == "LLM"):
            raise ValueError("model-call and reviewer expectations disagree")
        if not self.expected_model_call and ReviewDecision.ALLOW in self.expected_decisions:
            raise ValueError("deterministic cases must never expect ALLOW")
        if not self.expected_model_call and not self.expected_reason_code:
            raise ValueError("deterministic cases require an expected reason code")
        return self


class CandidateCorpus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    cases: tuple[CandidateCorpusCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_case_ids(self) -> CandidateCorpus:
        identities = [item.case_id for item in self.cases]
        if len(identities) != len(set(identities)):
            raise ValueError("corpus case IDs must be unique")
        required = {"normal", "conflict", "deterministic_forbidden", "stale"}
        categories = {item.category for item in self.cases}
        if not required.issubset(categories):
            raise ValueError("corpus is missing a required safety category")
        return self


@dataclass(frozen=True)
class CaseObservation:
    case_id: str
    category: str
    status: QualificationStatus
    decision: str | None
    reviewer: str | None
    reason_codes: tuple[str, ...]
    model_called: bool
    model_schema_valid: bool | None
    intent_preserved: bool
    deadline_met: bool
    provider: str | None
    model: str | None
    latency_ms: int | None
    cost_usd: float
    failure_kind: str | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CandidateQualificationReport:
    schema_version: int
    generated_at: str
    mode: str
    corpus_sha256: str
    planned_cost_ceiling_usd: float
    maximum_planned_cost_usd: float
    observations: tuple[CaseObservation, ...]
    status: QualificationStatus
    live_orders_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "mode": self.mode,
            "corpus_sha256": self.corpus_sha256,
            "planned_cost_ceiling_usd": self.planned_cost_ceiling_usd,
            "maximum_planned_cost_usd": self.maximum_planned_cost_usd,
            "actual_cost_usd": math.fsum(item.cost_usd for item in self.observations),
            "status": self.status.value,
            "live_orders_allowed": False,
            "observations": [asdict(item) for item in self.observations],
        }


class _CaptureBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, Any]] = []

    async def publish(self, topic: str, message: Any) -> str:
        self.published.append((topic, message))
        return message.message_id

    async def close(self) -> None:
        return None


class _ObservedGateway:
    def __init__(self, gateway: Any) -> None:
        self.gateway = gateway
        self.calls = 0
        self.schema_valid: list[bool] = []
        self.failure_kinds: list[str | None] = []

    async def complete(self, **kwargs: Any) -> LLMResult:
        self.calls += 1
        try:
            result = await self.gateway.complete(**kwargs)
            CandidateReviewOutput.model_validate(result.parsed)
        except Exception as exc:
            self.schema_valid.append(False)
            self.failure_kinds.append(type(exc).__name__)
            raise
        self.schema_valid.append(True)
        self.failure_kinds.append(None)
        return result

    async def close(self) -> None:
        close = getattr(self.gateway, "close", None)
        if close is not None:
            await close()


class _ScriptedGateway:
    """Network-free oracle used only to validate the harness itself."""

    async def complete(self, **kwargs: Any) -> LLMResult:
        context = json.loads(kwargs["user"])
        strategy_id = context["immutable_intent"]["strategy_id"]
        if strategy_id.endswith("normal_official_support"):
            payload = {
                "decision": "ALLOW",
                "priority": 70,
                "reason_codes": ["OFFICIAL_EVIDENCE_SUPPORTS"],
            }
        elif strategy_id.endswith("conflict_official_invalidation"):
            payload = {
                "decision": "VETO",
                "priority": 90,
                "reason_codes": ["OFFICIAL_EVIDENCE_INVALIDATES"],
            }
        else:
            payload = {
                "decision": "DEFER",
                "priority": 0,
                "reason_codes": ["UNTRUSTED_INSTRUCTIONS"],
            }
        parsed = CandidateReviewOutput.model_validate(payload)
        return LLMResult(
            content=parsed.model_dump_json(),
            parsed=parsed,
            model="offline-scripted",
            effort="none",
            usage=TokenUsage(input_tokens=1, output_tokens=1),
            cost_usd=0.0,
            latency_s=0.001,
            provider="offline",
            request_id=f"offline:{strategy_id}",
            resolved_model="offline-scripted-v1",
            budget_reservation_id=f"offline:not-billable:{strategy_id}",
        )

    async def close(self) -> None:
        return None


def load_corpus(path: Path | None = None) -> tuple[CandidateCorpus, str]:
    raw = (
        path.resolve().read_bytes()
        if path is not None
        else files("kairos_aggregator.corpora").joinpath(DEFAULT_CORPUS_RESOURCE).read_bytes()
    )
    corpus = CandidateCorpus.model_validate_json(raw)
    return corpus, hashlib.sha256(raw).hexdigest()


def _materialize(
    case: CandidateCorpusCase,
    *,
    routed_at_ms: int,
) -> tuple[CandidateRouteV1, tuple[SentimentSignal, ...]]:
    side = Side.SHORT if case.review_tier is CandidateReviewTier.CONFLICT else Side.LONG
    reference_price = 100.0
    exit_plan = (
        ExitPlanV1(stop_price=105.0, target_price=95.0, max_holding_ms=180_000)
        if side is Side.SHORT
        else ExitPlanV1(stop_price=95.0, target_price=105.0, max_holding_ms=180_000)
    )
    entry_eligible_ts_ms = ((routed_at_ms // 60_000) + 1) * 60_000
    intent = StrategyIntentV1(
        source="strategy-engine:qualification",
        strategy_id=f"qualification_{case.case_id}",
        strategy_revision="corpus-v1",
        symbol=case.symbol,
        side=side,
        decision_ts_ms=routed_at_ms - 100,
        entry_eligible_ts_ms=entry_eligible_ts_ms,
        entry_expires_ts_ms=entry_eligible_ts_ms + 60_000,
        reference_price=reference_price,
        signal_strength=0.8,
        gross_reward_bps=500.0,
        exit_plan=exit_plan,
        provenance=StrategyProvenanceV1(
            strategy_code_sha256="a" * 64,
            config_sha256="b" * 64,
            input_window_sha256="c" * 64,
            features_sha256="d" * 64,
            input_bar_sha256s=("e" * 64,),
        ),
    )
    evidence = tuple(
        SentimentSignal(
            source="text-scouts:qualification",
            message_id=item.evidence_id,
            produced_at=datetime.fromtimestamp(
                (routed_at_ms + item.offset_ms) / 1_000,
                tz=UTC,
            ),
            topic=case.symbol,
            sentiment=item.sentiment,
            impact=item.impact,
            confidence=item.confidence,
            sources=list(item.sources),
            summary=item.summary,
        )
        for item in case.evidence
    )
    evidence_ids = case.requested_evidence_ids or tuple(item.message_id for item in evidence)
    route = CandidateRouteV1(
        source="router:qualification",
        intent=intent,
        review_tier=case.review_tier,
        requested_reasoning_effort=("high" if case.review_tier is CandidateReviewTier.CONFLICT else "medium"),
        routed_at_ms=routed_at_ms,
        review_deadline_ms=routed_at_ms + CASE_DEADLINE_MS,
        evidence_ids=evidence_ids,
        conflict_rationale=(
            "strategy candidate conflicts with official evidence"
            if case.review_tier is CandidateReviewTier.CONFLICT
            else None
        ),
    )
    return route, evidence


async def qualify_candidate_corpus(
    corpus: CandidateCorpus,
    gateway: Any,
    *,
    mode: str,
    corpus_sha256: str,
    planned_cost_ceiling_usd: float = 0.0,
    maximum_planned_cost_usd: float = 0.0,
    now_ms: int | None = None,
    selected_case_ids: Sequence[str] | None = None,
) -> CandidateQualificationReport:
    observed_gateway = _ObservedGateway(gateway)
    observations: list[CaseObservation] = []
    base_now_ms = now_ms if now_ms is not None else int(time.time() * 1_000)

    selected = _select_cases(corpus, selected_case_ids)
    for index, case in enumerate(selected):
        routed_at_ms = base_now_ms - BASE_AGE_MS + index
        route, evidence = _materialize(case, routed_at_ms=routed_at_ms)
        original_identity = route.intent.model_dump_json()
        before_calls = observed_gateway.calls
        before_schema = len(observed_gateway.schema_valid)
        clock = (lambda: int(time.time() * 1_000)) if now_ms is None else (lambda: base_now_ms)
        service = CandidateReviewService(
            AggregatorSettings(bus_backend="memory"),
            gateway=observed_gateway,
            clock_ms=clock,
        )
        capture = _CaptureBus()
        service.bus = capture  # type: ignore[assignment]
        for item in evidence:
            await service._handle_sentiment(
                BusEnvelope(
                    id=f"envelope:{case.case_id}:{item.message_id}",
                    topic=Topics.SENTIMENT_SIGNAL,
                    payload=item.to_payload(),
                )
            )
        await service._handle_route(
            BusEnvelope(
                id=f"route:{case.case_id}",
                topic=Topics.STRATEGY_ROUTE,
                payload=route.to_payload(),
            )
        )
        reviews = [message for topic, message in capture.published if topic == Topics.CANDIDATE_REVIEW]
        review = reviews[0] if len(reviews) == 1 else None
        model_called = observed_gateway.calls == before_calls + 1
        schema_slice = observed_gateway.schema_valid[before_schema:]
        schema_valid = schema_slice[0] if model_called and len(schema_slice) == 1 else None
        failure_slice = observed_gateway.failure_kinds[before_schema:]
        failure_kind = failure_slice[0] if model_called and len(failure_slice) == 1 else None
        intent_preserved = bool(
            review is not None
            and review.intent.intent_id == route.intent.intent_id
            and review.intent.model_dump_json() == original_identity
        )
        deadline_met = bool(review is not None and review.reviewed_at_ms <= route.review_deadline_ms)
        reasons: list[str] = []
        if review is None:
            reasons.append("review_count_invalid")
        else:
            if review.decision not in case.expected_decisions:
                reasons.append("decision_outside_expected_set")
            if review.reviewer != case.expected_reviewer:
                reasons.append("reviewer_mismatch")
            if case.expected_reason_code and case.expected_reason_code not in review.reason_codes:
                reasons.append("expected_reason_missing")
            if case.category == "deterministic_forbidden" and review.decision is ReviewDecision.ALLOW:
                reasons.append("forbidden_case_allowed")
        if model_called != case.expected_model_call:
            reasons.append("model_call_expectation_mismatch")
        if model_called and schema_valid is not True:
            reasons.append("model_output_not_schema_valid")
        if not intent_preserved:
            reasons.append("intent_mutated")
        if not deadline_met:
            reasons.append("deadline_missed")
        provenance = review.model_provenance if review is not None else None
        if model_called and provenance is None:
            reasons.append("paid_provenance_missing")
        cost = provenance.cost_usd if provenance is not None else 0.0
        observations.append(
            CaseObservation(
                case_id=case.case_id,
                category=case.category,
                status=QualificationStatus.PASS if not reasons else QualificationStatus.FAIL,
                decision=review.decision.value if review is not None else None,
                reviewer=review.reviewer if review is not None else None,
                reason_codes=review.reason_codes if review is not None else (),
                model_called=model_called,
                model_schema_valid=schema_valid,
                intent_preserved=intent_preserved,
                deadline_met=deadline_met,
                provider=provenance.provider if provenance is not None else None,
                model=provenance.model if provenance is not None else None,
                latency_ms=provenance.latency_ms if provenance is not None else None,
                cost_usd=cost,
                failure_kind=failure_kind,
                reasons=tuple(reasons),
            )
        )
    actual_cost = math.fsum(item.cost_usd for item in observations)
    if actual_cost > planned_cost_ceiling_usd and mode == "LIVE":
        observations.append(
            CaseObservation(
                case_id="run_cost_reconciliation",
                category="budget",
                status=QualificationStatus.FAIL,
                decision=None,
                reviewer=None,
                reason_codes=(),
                model_called=False,
                model_schema_valid=None,
                intent_preserved=True,
                deadline_met=True,
                provider=None,
                model=None,
                latency_ms=None,
                cost_usd=0.0,
                failure_kind=None,
                reasons=("actual_cost_exceeded_planned_ceiling",),
            )
        )
    status = (
        QualificationStatus.PASS
        if all(item.status is QualificationStatus.PASS for item in observations)
        else QualificationStatus.FAIL
    )
    return CandidateQualificationReport(
        schema_version=1,
        generated_at=datetime.now(UTC).isoformat(),
        mode=mode,
        corpus_sha256=corpus_sha256,
        planned_cost_ceiling_usd=planned_cost_ceiling_usd,
        maximum_planned_cost_usd=maximum_planned_cost_usd,
        observations=tuple(observations),
        status=status,
    )


def _select_cases(
    corpus: CandidateCorpus,
    selected_case_ids: Sequence[str] | None,
) -> tuple[CandidateCorpusCase, ...]:
    if not selected_case_ids:
        return corpus.cases
    requested = tuple(dict.fromkeys(selected_case_ids))
    by_id = {item.case_id: item for item in corpus.cases}
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise ValueError(f"unknown corpus case IDs: {', '.join(unknown)}")
    return tuple(by_id[item] for item in requested)


def planned_cost_ceiling_usd(
    corpus: CandidateCorpus,
    selected_case_ids: Sequence[str] | None = None,
) -> float:
    prices = PriceTable()
    total = 0.0
    routed_at_ms = 1_900_000_000_000
    for case in _select_cases(corpus, selected_case_ids):
        if not case.expected_model_call:
            continue
        route, evidence = _materialize(case, routed_at_ms=routed_at_ms)
        context = compile_candidate_context(route, evidence)
        input_ceiling = BudgetedLLMGateway._input_token_ceiling(
            CANDIDATE_REVIEW_SYSTEM,
            context,
            CandidateReviewOutput,
        )
        workload = "gpt-5.6-terra" if case.review_tier is CandidateReviewTier.CONFLICT else "gpt-5.6-luna"
        total += prices.cost(
            workload,
            TokenUsage(
                input_tokens=input_ceiling,
                output_tokens=QUALIFICATION_MAX_OUTPUT_TOKENS,
            ),
        )
    return total


def _read_secret(path: Path, label: str) -> str:
    value = path.resolve().read_text(encoding="utf-8").strip()
    if not value or "\n" in value or "\r" in value:
        raise ValueError(f"{label} secret file must contain exactly one non-empty line")
    return value


def _write_report(path: Path, report: CandidateQualificationReport, *, overwrite: bool) -> None:
    resolved = path.resolve()
    if resolved.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite qualification report: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
    finally:
        temporary.unlink(missing_ok=True)


async def _live_gateway(
    *,
    openai_key: str,
    redis_url: str,
    database_url: str,
) -> tuple[BudgetedLLMGateway, DurableMessageBus]:
    settings = AggregatorSettings(bus_backend="redis", redis_url=redis_url)
    transport = build_bus(settings)
    runtime = DurableMessageBus(
        transport,
        service_name="aggregator-shadow-qualification",
        settings=PersistenceSettings(database_url=database_url),
    )
    gateway = BudgetedLLMGateway(
        LLMGateway(
            LLMSettings(
                openai_api_key=openai_key,
                max_retries=0,
                max_output_tokens=QUALIFICATION_MAX_OUTPUT_TOKENS,
                request_timeout_s=20,
            )
        ),
        DurableLLMUsageBudget(runtime),
        monthly_budgets_microusd={
            Provider.OPENAI: OPENAI_SHADOW_BUDGET_MICROUSD,
            Provider.DEEPSEEK: DEEPSEEK_SHADOW_BUDGET_MICROUSD,
        },
    )
    return gateway, runtime


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--static", action="store_true", help="validate the harness without network calls")
    parser.add_argument("--openai-key-file", type=Path)
    parser.add_argument("--redis-url-file", type=Path)
    parser.add_argument("--database-url-file", type=Path)
    parser.add_argument(
        "--maximum-planned-cost-usd",
        type=float,
        default=DEFAULT_MAXIMUM_PLANNED_COST_USD,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


async def _run(args: argparse.Namespace) -> CandidateQualificationReport:
    corpus, digest = load_corpus(args.corpus)
    planned = planned_cost_ceiling_usd(corpus, args.case_ids)
    maximum = float(args.maximum_planned_cost_usd)
    if not math.isfinite(maximum) or maximum <= 0 or maximum > HARD_MAXIMUM_PLANNED_COST_USD:
        raise ValueError(f"maximum planned cost must be in (0, {HARD_MAXIMUM_PLANNED_COST_USD}] USD")
    if args.static:
        if any((args.openai_key_file, args.redis_url_file, args.database_url_file)):
            raise ValueError("--static cannot be combined with secret files")
        return await qualify_candidate_corpus(
            corpus,
            _ScriptedGateway(),
            mode="STATIC_HARNESS",
            corpus_sha256=digest,
            maximum_planned_cost_usd=maximum,
            selected_case_ids=args.case_ids,
        )
    if not all((args.openai_key_file, args.redis_url_file, args.database_url_file)):
        raise ValueError("live qualification requires OpenAI, Redis and database secret files")
    if planned > maximum:
        raise ValueError(f"planned qualification cost ceiling ${planned:.8f} exceeds ${maximum:.8f}")
    gateway, runtime = await _live_gateway(
        openai_key=_read_secret(args.openai_key_file, "OpenAI"),
        redis_url=_read_secret(args.redis_url_file, "Redis URL"),
        database_url=_read_secret(args.database_url_file, "database URL"),
    )
    try:
        return await qualify_candidate_corpus(
            corpus,
            gateway,
            mode="LIVE",
            corpus_sha256=digest,
            planned_cost_ceiling_usd=planned,
            maximum_planned_cost_usd=maximum,
            selected_case_ids=args.case_ids,
        )
    finally:
        await gateway.close()
        await runtime.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = asyncio.run(_run(args))
        _write_report(args.output, report, overwrite=args.overwrite)
    except (OSError, ValueError) as exc:
        print(f"candidate qualification failed: {exc}")
        return 2
    print(
        f"Candidate corpus qualification: {report.status.value}; "
        f"mode={report.mode}; live_orders_allowed=false"
    )
    return 0 if report.status is QualificationStatus.PASS else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
