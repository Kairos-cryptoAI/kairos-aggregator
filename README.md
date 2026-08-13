# kairos-aggregator

Layer 3 of Kairos fuses a compact market snapshot and text sentiment into one
tactical command. Calm decisions select the explicit `AGGREGATOR_NORMAL` LLM
workload; signal conflicts select `AGGREGATOR_CONFLICT`. Provider and model
selection belong to `kairos-llm`, while the router's domain-level
`ReasoningEffort` is preserved in `TacticalCommand.effort_used`. Invalid model
output always becomes a deterministic `NO_TRADE` / `WAIT_CONFIRMATION` command.

## Evidence and abstention semantics

- A router decision is evaluated only against its exact `snapshot_id` and
  `sentiment_ids`; unrelated signals that happen to be in memory cannot change a
  replayed decision.
- Out-of-order decisions remain unacknowledged until their referenced evidence
  arrives. Seen-but-evicted evidence, stale snapshots/decisions, future timestamps,
  symbol mismatches, and quant-provenance mismatches produce a deterministic
  `NO_TRADE` without an LLM call.
- Sentiment messages are deduplicated by `message_id`, ordered newest-first with a
  stable ID tie-break, and confidence-calibrated as `sentiment * confidence`.
  Neutral-impact or impact/sign-contradictory evidence contributes zero directional
  score. The derived bias must match the Router decision; the shared deployment
  defaults are `KAIROS_SENTIMENT_MIN_CONFIDENCE=0.25` and
  `KAIROS_SENTIMENT_DEADBAND=0.25`.
- Model status/reason/side combinations are validated as one semantic unit. New
  entries and rebalances below `KAIROS_MIN_ENTRY_CONFIDENCE` abstain; protective
  `REDUCE_LEVERAGE` and `EXIT` outputs are not blocked by that entry threshold.

## Local development

The project is locked with `uv` 0.12.3, defaults to Python 3.11 and is also
tested on Linux Python 3.14 and Windows Python 3.11. `kairos-core` and
`kairos-llm` resolve from exact Git commits in `pyproject.toml` and `uv.lock`.

```sh
uv sync --locked
uv run --locked ruff check kairos_aggregator tests
uv run --locked ruff format --check kairos_aggregator tests
uv run --locked mypy kairos_aggregator
uv run --locked bandit -q -r kairos_aggregator -x tests
uv run --locked pytest -q --tb=short
uv build --no-sources
```

`make check` exposes the same complete verification sequence. Run the service
with `uv run --locked python -m kairos_aggregator`.

## Delivery and degraded-mode semantics

The service consumes market snapshots, sentiment signals, router decisions and
`kairos.system.control`. A message is acknowledged only after validation and
all related side effects, including tactical-command publication, succeed.
Failures remain unacknowledged for Redis Streams recovery.

- `TEXT_LOCAL_FILTER` clears pre-outage sentiment and marks subsequent local
  signals as degraded. Their sentiment is weighted by their confidence before
  it enters the compact LLM context.
- `CONFLICT_SAFE` suppresses the `AGGREGATOR_CONFLICT` workload and emits
  `WAIT_CONFIRMATION`; the `AGGREGATOR_NORMAL` workload remains available.
- `LOCAL_QUANT_MODE` suppresses every Aggregator LLM route.

The runtime uses `asyncio.TaskGroup` for all four consumers. Shutdown cancels
the group and closes both the message bus and LLM gateway.

Consumes `kairos.router.decision` plus snapshots, sentiment and system control;
emits `kairos.aggregator.command` and LLM health events.
