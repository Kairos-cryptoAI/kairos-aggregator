"""Call the LLM gateway with the right effort and parse a TacticalCommand.

DeepSeek-first tactical tiers:
  * MEDIUM (calm market)   -> DeepSeek-V4-Pro, the routine STABLE_TREND_ENTRY flow.
  * HIGH   (signal conflict) -> GPT-5.6 Sol, weighing technicals vs. news flow.
"""

from __future__ import annotations

from kairos_core.contracts import GridAdjustment, TacticalCommand
from kairos_core.enums import ReasonCode, ReasoningEffort, Side, TacticalStatus
from pydantic import BaseModel, ConfigDict, Field

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
    def __init__(self, gateway, *, source: str = "aggregator") -> None:
        self.gateway = gateway
        self.source = source

    async def decide(self, symbol: str, context_json: str, effort: ReasoningEffort) -> TacticalCommand:
        system = CONFLICT_SYSTEM if effort is ReasoningEffort.HIGH else NORMAL_SYSTEM
        try:
            result = await self.gateway.complete(
                system=system,
                user=context_json,
                effort=effort,
                schema=TacticalModelOutput,
            )
            output = (
                result.parsed
                if isinstance(result.parsed, TacticalModelOutput)
                else TacticalModelOutput.model_validate(result.parsed)
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
