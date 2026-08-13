"""Call the LLM gateway with an explicit role and parse a TacticalCommand.

The workload selects the architecture-owned provider/model route independently
of the domain ``ReasoningEffort`` retained on the resulting command:

* ``AGGREGATOR_NORMAL`` for calm-market decisions.
* ``AGGREGATOR_CONFLICT`` for conflicting quant and text signals.
"""

from __future__ import annotations

from kairos_core.contracts import GridAdjustment, TacticalCommand
from kairos_core.enums import ReasonCode, ReasoningEffort, Side, TacticalStatus
from kairos_llm import LLMWorkload
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .prompts import CONFLICT_SYSTEM, NORMAL_SYSTEM


class TacticalModelOutput(BaseModel):
    """Strict provider-independent schema for model-generated decisions."""

    model_config = ConfigDict(extra="forbid")

    status: TacticalStatus
    reason_code: ReasonCode
    target_side: Side = Side.FLAT
    requested_leverage: float = Field(1.0, gt=0, le=125)
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    grid: GridAdjustment | None = None
    rationale: str = Field(default="", max_length=280)

    @model_validator(mode="after")
    def validate_command_semantics(self) -> TacticalModelOutput:
        allowed_reasons = {
            TacticalStatus.STABLE_TREND_ENTRY: {
                ReasonCode.ENTER_LONG_TREND,
                ReasonCode.ENTER_SHORT_TREND,
            },
            TacticalStatus.HOLD_GRID: {ReasonCode.HOLD},
            TacticalStatus.SHIFT_GRID: {ReasonCode.REBALANCE},
            TacticalStatus.WAIT_CONFIRMATION: {ReasonCode.NO_TRADE},
            TacticalStatus.REDUCE_LEVERAGE: {ReasonCode.REDUCE_LEVERAGE},
            TacticalStatus.EXIT: {ReasonCode.CLOSE_POSITION},
        }
        if self.reason_code not in allowed_reasons[self.status]:
            raise ValueError(f"{self.reason_code.value} is incompatible with {self.status.value}")

        required_side = {
            ReasonCode.ENTER_LONG_TREND: Side.LONG,
            ReasonCode.ENTER_SHORT_TREND: Side.SHORT,
        }.get(self.reason_code)
        if required_side is not None and self.target_side is not required_side:
            raise ValueError(f"{self.reason_code.value} requires target_side={required_side.value}")
        if self.reason_code is ReasonCode.REBALANCE and self.target_side is Side.FLAT:
            raise ValueError("REBALANCE requires a directional target_side")
        if (
            self.reason_code
            not in {
                ReasonCode.ENTER_LONG_TREND,
                ReasonCode.ENTER_SHORT_TREND,
                ReasonCode.REBALANCE,
            }
            and self.target_side is not Side.FLAT
        ):
            raise ValueError(f"{self.reason_code.value} requires target_side=FLAT")
        if self.status is TacticalStatus.SHIFT_GRID and self.grid is None:
            raise ValueError("SHIFT_GRID requires a grid adjustment")
        return self


def safe_command(
    symbol: str,
    effort: ReasoningEffort,
    *,
    source: str = "aggregator",
    rationale: str = "fallback: invalid or failed model output",
) -> TacticalCommand:
    """Create the single deterministic no-trade response used by degraded paths."""
    return TacticalCommand(
        source=source,
        symbol=symbol,
        status=TacticalStatus.WAIT_CONFIRMATION,
        reason_code=ReasonCode.NO_TRADE,
        target_side=Side.FLAT,
        confidence=0.0,
        effort_used=effort,
        rationale=rationale,
    )


class AggregatorBrain:
    def __init__(
        self,
        gateway,
        *,
        source: str = "aggregator",
        min_entry_confidence: float = 0.60,
    ) -> None:
        self.gateway = gateway
        self.source = source
        self.min_entry_confidence = min_entry_confidence

    async def decide(self, symbol: str, context_json: str, effort: ReasoningEffort) -> TacticalCommand:
        conflict = effort is ReasoningEffort.HIGH
        system = CONFLICT_SYSTEM if conflict else NORMAL_SYSTEM
        workload = LLMWorkload.AGGREGATOR_CONFLICT if conflict else LLMWorkload.AGGREGATOR_NORMAL
        try:
            result = await self.gateway.complete(
                system=system,
                user=context_json,
                workload=workload,
                schema=TacticalModelOutput,
            )
            output = (
                result.parsed
                if isinstance(result.parsed, TacticalModelOutput)
                else TacticalModelOutput.model_validate(result.parsed)
            )
            if (
                output.reason_code
                in {
                    ReasonCode.ENTER_LONG_TREND,
                    ReasonCode.ENTER_SHORT_TREND,
                    ReasonCode.REBALANCE,
                }
                and output.confidence < self.min_entry_confidence
            ):
                return safe_command(
                    symbol,
                    effort,
                    source=self.source,
                    rationale=(
                        "abstention: entry confidence "
                        f"{output.confidence:.3f} below {self.min_entry_confidence:.3f}"
                    ),
                )
            return self._to_command(symbol, output, effort)
        except Exception:
            # Any gateway/parse failure becomes a safe, do-nothing command.
            return self._safe(symbol, effort)

    def _to_command(
        self,
        symbol: str,
        output: TacticalModelOutput,
        effort: ReasoningEffort,
    ) -> TacticalCommand:
        return TacticalCommand(
            source=self.source,
            symbol=symbol,
            effort_used=effort,
            **output.model_dump(),
        )

    def _safe(self, symbol: str, effort: ReasoningEffort) -> TacticalCommand:
        return safe_command(symbol, effort, source=self.source)
