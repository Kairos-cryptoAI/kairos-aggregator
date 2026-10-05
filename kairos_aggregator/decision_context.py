"""Bounded immutable receive-time source cache for the candidate service."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime

from kairos_core.contracts import (
    CandidateRouteV1,
    ClosedBarEventV1,
    MarketSnapshot,
    SentimentSignal,
    StrategicAllocation,
)
from kairos_core.contracts.base import (
    KairosMessage,
    canonical_json_bytes,
    canonical_sha256,
    datetime_from_unix_ms,
)
from kairos_core.contracts.decision_context import (
    ContextSourceKind,
    DecisionContextReceiptV1,
    DecisionContextSourceV1,
    DecisionContextV1,
)
from kairos_core.topics import Topics

from .config import AggregatorSettings

_MODELS: dict[ContextSourceKind, type[KairosMessage]] = {
    "market": MarketSnapshot,
    "closed_bars": ClosedBarEventV1,
    "text": SentimentSignal,
    "macro": StrategicAllocation,
}
_TOPICS = {
    "market": Topics.MARKET_SNAPSHOT,
    "closed_bars": Topics.CLOSED_BAR,
    "text": Topics.SENTIMENT_SIGNAL,
    "macro": Topics.STRATEGIC_ALLOCATION,
}


@dataclass(frozen=True)
class _ObservedSource:
    kind: ContextSourceKind
    payload_json: str
    receipt: DecisionContextReceiptV1

    def payload(self) -> dict:
        import json

        return json.loads(self.payload_json)


def _timestamp(value: datetime) -> int:
    if value.utcoffset() is None:
        raise ValueError("source timestamps must be timezone-aware")
    return int(value.timestamp() * 1_000)


def _unavailable(kind: ContextSourceKind, reason: str) -> DecisionContextSourceV1:
    return DecisionContextSourceV1(kind=kind, availability="UNAVAILABLE", reason_code=reason)


def _available(kind: ContextSourceKind, values: tuple[_ObservedSource, ...]) -> DecisionContextSourceV1:
    return DecisionContextSourceV1(
        kind=kind,
        availability="AVAILABLE",
        receipts=tuple(value.receipt for value in values),
        payloads_json=canonical_json_bytes({"items": [value.payload() for value in values]}).decode("utf-8"),
    )


class CandidateContextStore:
    """First receipt wins; a conflicting identity quarantines that source kind.

    No DB restore is attempted: after restart historical messages get their new
    local receipt clocks. They cannot prove past availability from produced_at.
    """

    def __init__(self, settings: AggregatorSettings) -> None:
        self.settings = settings
        self._entries: OrderedDict[tuple[str, str], _ObservedSource] = OrderedDict()
        self._digests: OrderedDict[tuple[str, str], str] = OrderedDict()
        self._bar_slots: OrderedDict[tuple[str, int], str] = OrderedDict()
        self.quarantined_kinds: set[str] = set()

    def ingest(self, kind: ContextSourceKind, payload: dict, *, received_at_ms: int) -> None:
        message = _MODELS[kind].model_validate(payload)
        normalized = message.model_dump(mode="json")
        if canonical_json_bytes(normalized) != canonical_json_bytes(payload):
            raise ValueError("source contains ignored/unknown fields or noncanonical scope")
        digest = canonical_sha256(normalized)
        key = (kind, message.message_id)
        previous = self._digests.get(key)
        if previous is not None:
            if previous != digest:
                self.quarantined_kinds.add(kind)
                raise ValueError("context source message identity conflict")
            return  # Never replace the first receipt clock during replay.
        self._digests[key] = digest
        while len(self._digests) > self.settings.processed_cache_size:
            self._digests.popitem(last=False)
        produced = _timestamp(message.produced_at)
        event = message.close_time_ms if isinstance(message, ClosedBarEventV1) else produced
        if produced > received_at_ms or event > received_at_ms or event > produced:
            raise ValueError("context source timestamp is in the future at local receipt")
        if isinstance(message, (MarketSnapshot, ClosedBarEventV1)):
            if message.symbol not in self.settings.trading_symbols:
                raise ValueError("context source symbol is outside the configured universe")
        if isinstance(message, MarketSnapshot):
            if message.order_book.best_ask < message.order_book.best_bid:
                raise ValueError("context market book is crossed")
        if isinstance(message, StrategicAllocation):
            if any(weight < 0 for weight in message.strategy_weights.values()):
                raise ValueError("macro context contains a negative allocation weight")
        if isinstance(message, ClosedBarEventV1):
            slot = (message.symbol, message.open_time_ms)
            previous_bar = self._bar_slots.get(slot)
            if previous_bar is not None and previous_bar != message.bar_sha256:
                self.quarantined_kinds.add(kind)
                raise ValueError("context source closed-bar coordinate conflict")
            if message.bar_sha256 is None:  # impossible for a validated bar
                raise ValueError("closed-bar source has no canonical identity")
            self._bar_slots[slot] = message.bar_sha256
            while len(self._bar_slots) > self.settings.processed_cache_size:
                self._bar_slots.popitem(last=False)
        ttl = {
            "market": self.settings.snapshot_ttl_s,
            "text": self.settings.sentiment_ttl_s,
            "macro": self.settings.candidate_macro_ttl_s,
            # Old bars are intentionally historical, not latest-market snapshots.
            "closed_bars": self.settings.candidate_context_max_bars * 60 + self.settings.snapshot_ttl_s,
        }[kind]
        observed = _ObservedSource(
            kind=kind,
            payload_json=canonical_json_bytes(normalized).decode("utf-8"),
            receipt=DecisionContextReceiptV1(
                message_id=message.message_id,
                source=message.source,
                topic=_TOPICS[kind],
                content_sha256=digest,
                event_at_ms=event,
                produced_at_ms=produced,
                received_at_ms=received_at_ms,
                ttl_ms=int(ttl * 1_000),
            ),
        )
        self._entries[key] = observed
        while len(self._entries) > self.settings.processed_cache_size:
            self._entries.popitem(last=False)

    def _eligible(self, value: _ObservedSource, *, event_cut: int, cutoff: int, captured: int) -> bool:
        receipt = value.receipt
        return (
            receipt.event_at_ms <= event_cut
            and receipt.produced_at_ms <= cutoff
            and receipt.received_at_ms <= cutoff
            and captured - receipt.event_at_ms <= receipt.ttl_ms
        )

    def _snapshot(self, kind: ContextSourceKind, route: CandidateRouteV1, cutoff: int, captured: int):
        if kind in self.quarantined_kinds:
            return _unavailable(kind, "IDENTITY_CONFLICT")
        matching = tuple(
            value
            for (entry_kind, _), value in self._entries.items()
            if entry_kind == kind and (kind == "macro" or value.payload()["symbol"] == route.intent.symbol)
        )
        eligible = tuple(
            value
            for value in matching
            if self._eligible(value, event_cut=route.routed_at_ms, cutoff=cutoff, captured=captured)
        )
        if not eligible:
            return _unavailable(kind, "MISSING_OR_NONCAUSAL_OR_STALE")
        latest = max(eligible, key=lambda value: (value.receipt.event_at_ms, value.receipt.message_id))
        return _available(kind, (latest,))

    def build(self, route: CandidateRouteV1, *, cutoff_ms: int, captured_at_ms: int) -> DecisionContextV1:
        sources = [
            self._snapshot("market", route, cutoff_ms, captured_at_ms),
            self._snapshot("macro", route, cutoff_ms, captured_at_ms),
        ]
        expected = route.intent.provenance.input_bar_sha256s[-self.settings.candidate_context_max_bars :]
        bars = {
            value.payload()["bar_sha256"]: value
            for (kind, _), value in self._entries.items()
            if kind == "closed_bars"
            and self._eligible(
                value, event_cut=route.intent.decision_ts_ms, cutoff=cutoff_ms, captured=captured_at_ms
            )
        }
        if "closed_bars" in self.quarantined_kinds:
            bar_source = _unavailable("closed_bars", "IDENTITY_CONFLICT")
        elif not all(digest in bars for digest in expected):
            bar_source = _unavailable("closed_bars", "DECLARED_BAR_EVIDENCE_UNAVAILABLE")
        else:
            bar_source = _available("closed_bars", tuple(bars[digest] for digest in expected))
        sources.append(bar_source)
        selected = tuple(self._entries.get(("text", identity)) for identity in route.evidence_ids)
        present = tuple(value for value in selected if value is not None)
        if "text" in self.quarantined_kinds:
            text = _unavailable("text", "IDENTITY_CONFLICT")
        elif not route.evidence_ids:
            text = _unavailable("text", "NO_REFERENCED_TEXT")
        elif len(present) != len(selected):
            text = _unavailable("text", "EVIDENCE_UNAVAILABLE")
        elif not all(
            self._eligible(value, event_cut=route.routed_at_ms, cutoff=cutoff_ms, captured=captured_at_ms)
            for value in present
        ):
            text = _unavailable("text", "EVIDENCE_NONCAUSAL_OR_STALE")
        elif any(
            value.payload()["topic"].strip().upper().isalnum()
            and value.payload()["topic"].strip().upper().endswith("USDT")
            and value.payload()["topic"] != route.intent.symbol
            for value in present
        ):
            text = _unavailable("text", "EVIDENCE_SYMBOL_MISMATCH")
        else:
            text = _available("text", present)
        sources.append(text)
        required: list[ContextSourceKind] = ["market", "closed_bars"]
        if route.evidence_ids or self.settings.candidate_require_text_context:
            required.append("text")
        if self.settings.candidate_require_macro_context:
            required.append("macro")
        return DecisionContextV1(
            source=self.settings.service_name,
            produced_at=datetime_from_unix_ms(captured_at_ms),
            correlation_id=route.intent.correlation_id or route.intent.intent_id,
            causation_id=route.message_id,
            route_id=route.route_id,
            intent_id=route.intent.intent_id,
            route_identity_json=canonical_json_bytes(route.identity_payload()).decode("utf-8"),
            intent_identity_json=route.intent.canonical_intent_bytes().decode("utf-8"),
            symbol=route.intent.symbol,
            event_as_of_ms=route.routed_at_ms,
            knowledge_cutoff_ms=cutoff_ms,
            captured_at_ms=captured_at_ms,
            review_deadline_ms=route.review_deadline_ms,
            maximum_closed_bars=self.settings.candidate_context_max_bars,
            required_sources=tuple(required),
            sources=tuple(sorted(sources, key=lambda value: value.kind)),
        )

    def integrity_error(self, context: DecisionContextV1) -> bool:
        return any(source.kind in self.quarantined_kinds for source in context.sources if source.receipts)


__all__ = ["CandidateContextStore"]
