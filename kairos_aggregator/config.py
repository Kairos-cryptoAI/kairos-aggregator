from __future__ import annotations

from kairos_core.config import CoreSettings
from pydantic import Field


class AggregatorSettings(CoreSettings):
    service_name: str = "kairos-aggregator"
    max_sentiment_window: int = 5
    sentiment_ttl_s: float = Field(600.0, gt=0)
    processed_cache_size: int = Field(10_000, ge=1)
    snapshot_cache_size: int = Field(1_000, ge=1)
