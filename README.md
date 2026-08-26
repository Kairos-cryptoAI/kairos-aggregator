# kairos-aggregator

Layer 3 has two deliberately separate entry points. The Strategy Parity path
reviews an immutable `CandidateRouteV1` and can return only `ALLOW`, `VETO`,
`DEFER` and a priority. The legacy DRY_RUN path still fuses a compact market
snapshot and text sentiment into a tactical command. Calm decisions select the explicit `AGGREGATOR_NORMAL` LLM
workload; signal conflicts select `AGGREGATOR_CONFLICT`. Provider and model
selection belong to `kairos-llm`, while the router's domain-level
`ReasoningEffort` is preserved in `TacticalCommand.effort_used`. Invalid model
output always becomes a deterministic `NO_TRADE` / `WAIT_CONFIRMATION` command.

## Immutable candidate review

Run `uv run --locked kairos-candidate-review` for the Strategy Parity/shadow
consumer:

```text
kairos.strategy.route.v1 -> Candidate Review -> kairos.strategy.review.v1
```

- The complete Strategy Intent is carried into the review unchanged; strict
  contracts reject any mutation of side, stop, target, timeout or provenance.
- The model schema contains only `decision`, `priority` and `reason_codes`.
  Normal routes use Luna/medium and conflict routes use Terra/high.
- Missing, stale or post-route evidence, disabled system modes, malformed output,
  provider errors, incomplete paid-call provenance and missed deadlines terminate
  the current intent as deterministic `DEFER` without an automatic second call.
- Every successful LLM review records provider request ID, resolved model,
  prompt/response hashes, durable budget reservation, latency and cost.
- A publish retry reuses the cached review and never calls the model again.

The candidate service enforces the qualification ceilings on the shared durable
ledger: OpenAI `$12` and DeepSeek `$1`. The remaining `$2` X allocation is owned by
Text Scouts. Technical EVEDEX canaries do not start this process and therefore make
no LLM call.

## Frozen-corpus shadow qualification

`kairos-candidate-qualify` replays the packaged V1 corpus through the real candidate
service boundary. The corpus covers normal support, a material conflict, untrusted
prompt injection, a forbidden symbol, stale evidence, missing evidence and evidence
that postdates the frozen route. It requires strict output, an unchanged intent,
deadline completion, complete paid-call provenance and zero `ALLOW` decisions on
deterministically forbidden cases.

Validate the harness without a network request or spend:

```sh
uv run --locked kairos-candidate-qualify --static \
  --output /tmp/kairos-candidate-harness.json
```

A real provider run additionally requires OpenAI, Redis and PostgreSQL one-value
secret files. It reserves every Luna/Terra call in the shared durable
`kairos-llm-v1/openai` ledger before network access and refuses a planned run above
the configured ceiling (default `$0.10`, hard maximum `$0.25`). Reports are atomic,
contain no prompts or credentials and always set `live_orders_allowed=false`:

```sh
uv run --locked kairos-candidate-qualify \
  --openai-key-file /run/secrets/openai_api_key \
  --redis-url-file /run/secrets/redis_url \
  --database-url-file /run/secrets/persistence_database_url \
  --maximum-planned-cost-usd 0.10 \
  --output /tmp/kairos-candidate-live.json
```

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

All Luna/Terra calls reserve capacity in the shared PostgreSQL
`kairos-llm-v1/openai` ledger before contacting OpenAI. The provider-wide
shadow ceiling is `$12`; in-memory runtimes deny paid calls. Automatic model
retry is disabled so an ambiguous response cannot silently spend twice.

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

## Runtime delivery durability

With Redis, consumed IDs, handler outputs and completion are committed through
`kairos-persistence`; Redis is ACKed only after PostgreSQL commits. Configure
`KAIROS_PERSISTENCE_DATABASE_URL` through the deployment secret provider. The
in-memory backend intentionally bypasses persistence for local tests.
