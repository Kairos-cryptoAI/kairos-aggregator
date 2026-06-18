"""The production gateway must publish LLM health events back to the bus."""
from kairos_aggregator.config import AggregatorSettings
from kairos_aggregator.service import AggregatorService


def test_gateway_health_hook_is_wired():
    svc = AggregatorService(AggregatorSettings(bus_backend="memory"))
    assert svc.brain.gateway._on_health is not None
