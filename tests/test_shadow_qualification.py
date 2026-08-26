"""Frozen candidate corpus and fail-closed qualification tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from kairos_llm import LLMResult, TokenUsage

from kairos_aggregator.candidate_review import CandidateReviewOutput
from kairos_aggregator.shadow_qualification import (
    HARD_MAXIMUM_PLANNED_COST_USD,
    CandidateCorpus,
    QualificationStatus,
    _ScriptedGateway,
    load_corpus,
    main,
    planned_cost_ceiling_usd,
    qualify_candidate_corpus,
)

NOW_MS = 1_900_000_000_000


async def test_packaged_corpus_passes_network_free_safety_harness() -> None:
    corpus, digest = load_corpus()
    report = await qualify_candidate_corpus(
        corpus,
        _ScriptedGateway(),
        mode="STATIC_HARNESS",
        corpus_sha256=digest,
        maximum_planned_cost_usd=0.1,
        now_ms=NOW_MS,
    )

    assert report.status is QualificationStatus.PASS
    assert report.live_orders_allowed is False
    assert len(report.observations) == 7
    assert all(item.intent_preserved and item.deadline_met for item in report.observations)
    assert all(
        item.decision != "ALLOW" for item in report.observations if item.category == "deterministic_forbidden"
    )
    model_cases = [item for item in report.observations if item.model_called]
    assert len(model_cases) == 3
    assert all(item.model_schema_valid is True for item in model_cases)
    assert all(item.cost_usd == 0 for item in report.observations)


class _UnsafeGateway:
    async def complete(self, **_kwargs) -> LLMResult:
        parsed = CandidateReviewOutput.model_validate(
            {"decision": "ALLOW", "priority": 100, "reason_codes": ["ALLOW"]}
        )
        return LLMResult(
            content=parsed.model_dump_json(),
            parsed=parsed,
            model="gpt-5.6-luna",
            effort="medium",
            usage=TokenUsage(input_tokens=10, output_tokens=10),
            cost_usd=0.001,
            latency_s=0.1,
            provider="openai",
            request_id="unsafe",
            resolved_model="gpt-5.6-luna",
            budget_reservation_id="kairos-llm-v1:openai:unsafe",
        )


async def test_corpus_fails_when_conflict_is_allowed_but_deterministic_cases_do_not_call_model() -> None:
    corpus, digest = load_corpus()
    report = await qualify_candidate_corpus(
        corpus,
        _UnsafeGateway(),
        mode="LIVE",
        corpus_sha256=digest,
        planned_cost_ceiling_usd=1.0,
        maximum_planned_cost_usd=1.0,
        now_ms=NOW_MS,
    )

    assert report.status is QualificationStatus.FAIL
    conflict = next(item for item in report.observations if item.category == "conflict")
    assert conflict.decision == "ALLOW"
    assert "decision_outside_expected_set" in conflict.reasons
    deterministic = [item for item in report.observations if not item.model_called]
    assert len(deterministic) == 4


def test_corpus_validation_rejects_missing_required_category_and_duplicate_id() -> None:
    corpus, _digest = load_corpus()
    payload = corpus.model_dump(mode="json")
    payload["cases"] = [item for item in payload["cases"] if item["category"] != "stale"]
    with pytest.raises(ValueError, match="required safety category"):
        CandidateCorpus.model_validate(payload)

    payload = corpus.model_dump(mode="json")
    payload["cases"][1]["case_id"] = payload["cases"][0]["case_id"]
    with pytest.raises(ValueError, match="unique"):
        CandidateCorpus.model_validate(payload)


def test_planned_cost_is_positive_bounded_and_only_counts_model_cases() -> None:
    corpus, _digest = load_corpus()
    planned = planned_cost_ceiling_usd(corpus)
    assert 0 < planned < HARD_MAXIMUM_PLANNED_COST_USD


def test_atomic_report_writer_and_static_cli(tmp_path: Path) -> None:
    output = tmp_path / "candidate.json"
    assert main(["--static", "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["mode"] == "STATIC_HARNESS"
    assert payload["live_orders_allowed"] is False
    rendered = json.dumps(payload).casefold()
    assert "stop_price" not in rendered
    assert "official.example.invalid" not in rendered

    assert main(["--static", "--output", str(output)]) == 2


def test_static_mode_rejects_secret_files_before_reading_them(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    output = tmp_path / "report.json"
    assert (
        main(
            [
                "--static",
                "--openai-key-file",
                str(missing),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert not output.exists()
