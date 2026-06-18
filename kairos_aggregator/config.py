from __future__ import annotations

from kairos_core.config import CoreSettings


class AggregatorSettings(CoreSettings):
    service_name: str = "kairos-aggregator"
    max_sentiment_window: int = 5
