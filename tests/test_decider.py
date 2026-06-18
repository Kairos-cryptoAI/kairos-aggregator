import asyncio
from types import SimpleNamespace

from kairos_aggregator.decider import AggregatorBrain
from kairos_core.enums import ReasoningEffort, ReasonCode, TacticalStatus


class FakeGateway:
    def __init__(self, parsed): self._p = parsed
    async def complete(self, *, system, user, effort, schema=None):
        return SimpleNamespace(parsed=self._p)


def test_valid_command_parsed():
    gw = FakeGateway({"status": "STABLE_TREND_ENTRY", "reason_code": "ENTER_LONG_TREND",
                      "target_side": "LONG", "requested_leverage": 3, "confidence": 0.8, "rationale": "trend up"})
    cmd = asyncio.run(AggregatorBrain(gw).decide("BTCUSD", "{}", ReasoningEffort.MEDIUM))
    assert cmd.status is TacticalStatus.STABLE_TREND_ENTRY
    assert cmd.reason_code is ReasonCode.ENTER_LONG_TREND
    assert cmd.requested_leverage == 3


def test_bad_output_falls_back_to_safe():
    gw = FakeGateway({"status": "NONSENSE"})
    cmd = asyncio.run(AggregatorBrain(gw).decide("BTCUSD", "{}", ReasoningEffort.HIGH))
    assert cmd.reason_code is ReasonCode.NO_TRADE
    assert cmd.status is TacticalStatus.WAIT_CONFIRMATION
