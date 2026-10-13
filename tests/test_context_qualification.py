"""V2 synthetic-context harness must exercise current service boundary, offline."""

from __future__ import annotations

import json
from pathlib import Path

from kairos_aggregator.context_qualification import (
    ContextCorpus,
    _materialize,
    load_context_corpus,
    main,
    qualify_context_corpus,
)


async def test_v2_context_corpus_qualifies_current_service_with_synthetic_local_gateway() -> None:
    corpus, digest = load_context_corpus()

    report = await qualify_context_corpus(corpus, digest)

    assert report.status == "PASS"
    assert report.qualification_kind == "synthetic_local_engineering_only"
    assert report.paid_qualification is False
    assert report.source_authenticity_verified is False
    assert report.production_ab is False
    assert report.live_orders_allowed is False
    by_id = {item.case_id: item for item in report.observations}
    for case_id in (
        "normal_official_support",
        "conflict_official_invalidation",
        "conflict_allow_guard",
        "untrusted_prompt_injection",
    ):
        item = by_id[case_id]
        assert item.model_called is True
        assert item.model_call_count == 1
        assert item.context_complete_at_model_call is True
        assert item.provenance_complete is True
        assert item.cost_usd == 0
    assert by_id["normal_official_support"].decision == "ALLOW"
    assert by_id["conflict_official_invalidation"].decision in {"VETO", "DEFER"}
    assert by_id["conflict_allow_guard"].decision == "DEFER"
    assert "CONFLICT_ALLOW_GUARD" in by_id["conflict_allow_guard"].reason_codes
    for case_id in (
        "forbidden_symbol",
        "stale_text",
        "future_text",
        "missing_market",
        "missing_macro",
        "missing_bar",
        "missing_text",
    ):
        assert by_id[case_id].decision == "DEFER"
        assert by_id[case_id].model_called is False
        assert by_id[case_id].model_call_count == 0
    assert all(item.intent_preserved and item.deadline_met for item in report.observations)


def test_v2_fixtures_declare_exact_bar_tail_and_macro_requirement() -> None:
    corpus, _digest = load_context_corpus()
    case = corpus.cases[0]
    route, source = _materialize(case, 0)

    assert route.intent.provenance.input_bar_sha256s == (source["bar"].bar_sha256,)
    assert source["bar"].close_time_ms == route.intent.decision_ts_ms
    assert source["market"].symbol == route.intent.symbol
    assert source["macro"].produced_at.timestamp() * 1_000 <= route.routed_at_ms
    assert source["text"].produced_at.timestamp() * 1_000 <= route.routed_at_ms


def test_v2_corpus_validation_keeps_synthetic_qualification_scope_explicit() -> None:
    corpus, _digest = load_context_corpus()
    payload = corpus.model_dump(mode="json")
    payload["qualification_kind"] = "live"
    try:
        ContextCorpus.model_validate(payload)
    except ValueError:
        pass
    else:
        raise AssertionError("qualification kind must not be relabeled as live")


def test_v2_cli_writes_non_authorizing_offline_report(tmp_path: Path) -> None:
    output = tmp_path / "context-v2.json"

    assert main(["--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["paid_qualification"] is False
    assert payload["source_authenticity_verified"] is False
    assert payload["production_ab"] is False
    assert payload["live_orders_allowed"] is False
    assert main(["--output", str(output)]) == 2
