"""Kairos Layer 3 — The Aggregator.

Fuses the compact quant snapshot and text sentiment into a single tactical
command. The Router tells it how hard to think: ``medium`` for a calm market
(maintain the grid / follow the trend) or ``high`` to resolve a conflict between
indicators and news.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .compiler import compile_context
from .decider import AggregatorBrain

__all__ = ["compile_context", "AggregatorBrain", "__version__"]
