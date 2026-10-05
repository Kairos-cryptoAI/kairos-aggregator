# Candidate DecisionContext v1

The ordinary `CandidateReviewService` builds and publishes one frozen
`DecisionContextV1` before calling a model. It subscribes to the existing
`MARKET_SNAPSHOT`, `CLOSED_BAR`, `SENTIMENT_SIGNAL`, `STRATEGIC_ALLOCATION`,
`SYSTEM_CONTROL` and `STRATEGY_ROUTE` topics; it does not accept an externally
published context as authority. The output topic is
`kairos.aggregator.decision_context.v1`, followed by `kairos.aggregator.review.v1`.

The context binds the exact canonical intent and route identities, source
payload bytes/hashes, producer/event clocks and first local receipt clocks.
The event cut is `route.routed_at_ms`; the knowledge cutoff is the first local
route receipt, captured **before** waiting for the route lock. Capture may be
later, but queued delivery cannot expand the cutoff. Producer timestamps or
envelope metadata cannot prove earlier availability. Sources produced/received
in the future, observed after the cut, stale, differently scoped, or conflicting
under one message/bar identity cannot support review.

All four source slots are explicit. Market features and closed-bar evidence are
mandatory. `MarketSnapshot` is only compact numeric/book/derivatives evidence,
not an OHLC window. Closed bars contain the exact ordered tail of declared
`input_bar_sha256s` (default at most 60); the final bar must be the strategy's
decision bar. This does **not** verify or reconstruct the entire input window or
its feature/window hash. Macro is advisory and optional by default. Text uses
only exact routed message IDs and is mandatory when any are referenced; no
latest-news substitution is permitted. An unavailable optional source is
`UNAVAILABLE`, not fabricated neutral sentiment or allocation. Required missing
sources deterministically `DEFER` with no model call. Macro/text can additionally
be required with `candidate_require_macro_context` / `candidate_require_text_context`.

Market/text freshness uses existing snapshot/sentiment TTLs; macro defaults to
26 hours. Historical bar-tail TTL is the configured maximum tail duration plus
snapshot TTL. Freshness is checked at capture, before dispatch and after model
completion. Explicit single-pair `*USDT` text topics must exactly match the
candidate in both the Core validator and store. Legacy free-form topics such as
`SEC ETF` have no typed symbol tagging; the context does not claim otherwise.

A context publication failure blocks model dispatch. Retried publication uses
the same immutable context and cannot upgrade it with late sources. A review
publication retry reuses the original review/provenance without another model
call; it is not a fresh review. A newly known source identity conflict prevents
cached review republication instead of rewriting the original. Identity conflicts
quarantine the affected source kind for this service instance. In-process caches
are bounded by `processed_cache_size`; there is no durable exactly-once model
dispatch or restored receipt history here. A process restart or cache eviction
is not proof of historical availability. Bus output durability does not convert
these local receipt clocks into a durable point-in-time archive attestation.

The model schema remains only `ALLOW`/`VETO`/`DEFER`, priority and reason codes.
It cannot alter any trading parameters. Successful paid-call provenance hashes
the actual context prompt, and every context-backed review references the content-addressed
context. Risk, venue, execution, `REJECT_ALL`, frozen evaluator and research gates
are unchanged. This is engineering migration evidence, not alpha/profitability,
production qualification or permission to trade.

`CandidateReviewBrain.review_legacy_engineering` (and its compatibility alias
`review`) retain the old direct fixture adapter. The ordinary service always uses
`review_with_context`; there is no runtime fallback. The packaged historical
candidate corpus lacks real market/bar context, so the unchanged qualification
runner now returns `FAIL` / not qualified without model calls. Historical PASS
receipts must not be relabelled or reused for the new source identity; a new
versioned context-aware qualification artifact is future gated work.
