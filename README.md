# kairos-aggregator

**Layer 3 — The Aggregator.** Fuses the compact quant snapshot and text sentiment into a
single tactical command. The Router decides how hard it thinks:

| scenario | effort | behaviour | typical output |
| --- | --- | --- | --- |
| calm, signals agree | `medium` | maintain grid / follow trend | `STABLE_TREND_ENTRY`, `HOLD_GRID` |
| turbulence, signals conflict | `high` | weigh technicals vs. news, protect capital | `WAIT_CONFIRMATION`, `REDUCE_LEVERAGE` |

`compile_context()` enforces the core rule — **the LLM never sees raw numbers**, only a
small, rounded, decision-relevant JSON. Malformed model output always degrades to a safe
`NO_TRADE` / `WAIT_CONFIRMATION` command. All model calls go through
[`kairos-llm`](https://github.com/Kairos-cryptoAI/kairos-llm).

## Run
```bash
pip install -e ../kairos-core -e ../kairos-llm && pip install -e ".[dev]"
make test
python -m kairos_aggregator
```
Consumes `kairos.router.decision` (+ snapshots & sentiment); emits `kairos.aggregator.command`.

---
Part of the [Kairos](https://github.com/Kairos-cryptoAI/kairos) system. MIT licensed.
