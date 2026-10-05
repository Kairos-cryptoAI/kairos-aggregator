from __future__ import annotations

from kairos_core.config import CoreSettings
from pydantic import Field


class AggregatorSettings(CoreSettings):
    service_name: str = "kairos-aggregator"
    max_sentiment_window: int = Field(5, ge=1)
    sentiment_ttl_s: float = Field(600.0, gt=0)
    snapshot_ttl_s: float = Field(120.0, gt=0)
    router_decision_ttl_s: float = Field(120.0, gt=0)
    max_future_skew_s: float = Field(5.0, ge=0)
    sentiment_deadband: float = Field(0.25, ge=0.0, le=1.0)
    sentiment_min_confidence: float = Field(0.25, ge=0.0, le=1.0)
    min_entry_confidence: float = Field(0.60, ge=0.0, le=1.0)
    processed_cache_size: int = Field(10_000, ge=1)
    snapshot_cache_size: int = Field(1_000, ge=1)
    # Receive-time candidate context, not a full historic OHLC reconstruction.
    candidate_context_max_bars: int = Field(60, ge=1, le=1_440)
    candidate_macro_ttl_s: float = Field(26 * 60 * 60, gt=0)
    candidate_require_macro_context: bool = False
    candidate_require_text_context: bool = False
