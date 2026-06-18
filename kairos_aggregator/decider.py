"""Call the LLM gateway with the right effort and parse a TacticalCommand."""
from __future__ import annotations

from kairos_core.contracts import TacticalCommand
from kairos_core.enums import ReasoningEffort, ReasonCode, Side, TacticalStatus

from .prompts import CONFLICT_SYSTEM, NORMAL_SYSTEM


class AggregatorBrain:
    def __init__(self, gateway, *, source: str = "aggregator") -> None:
        self.gateway = gateway
        self.source = source

    async def decide(self, symbol: str, context_json: str, effort: ReasoningEffort) -> TacticalCommand:
        system = CONFLICT_SYSTEM if effort is ReasoningEffort.HIGH else NORMAL_SYSTEM
        try:
            res = await self.gateway.complete(system=system, user=context_json, effort=effort)
            data = res.parsed if isinstance(res.parsed, dict) else {}
            return self._to_command(symbol, data, effort)
        except Exception:
            # Any gateway/parse failure becomes a safe, do-nothing command.
            return self._safe(symbol, effort)

    def _to_command(self, symbol: str, data: dict, effort: ReasoningEffort) -> TacticalCommand:
        try:
            return TacticalCommand(
                source=self.source, symbol=symbol,
                status=TacticalStatus(data["status"]),
                reason_code=ReasonCode(data["reason_code"]),
                target_side=Side(data.get("target_side", "FLAT")),
                requested_leverage=float(data.get("requested_leverage", 1.0)),
                confidence=float(data.get("confidence", 0.5)),
                effort_used=effort,
                rationale=str(data.get("rationale", ""))[:280],
            )
        except (KeyError, ValueError):
            return self._safe(symbol, effort)

    def _safe(self, symbol: str, effort: ReasoningEffort) -> TacticalCommand:
        return TacticalCommand(
            source=self.source, symbol=symbol, status=TacticalStatus.WAIT_CONFIRMATION,
            reason_code=ReasonCode.NO_TRADE, target_side=Side.FLAT, confidence=0.0,
            effort_used=effort, rationale="fallback: invalid or failed model output",
        )
