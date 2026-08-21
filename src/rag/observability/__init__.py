"""Measurement of what the system spends and how it behaves at runtime.

Kept out of `rag.extraction` on purpose: cost accounting is not a property of
relationship extraction, it is a property of every paid call the system makes
-- embeddings, the extractor, the answering agents, the reranker. One tracker
type, one set of rates from config, one summary format, so the ingest report
and the query report are directly comparable.
"""
from __future__ import annotations

from rag.observability.cost import CallCost, CostTracker

__all__ = ["CallCost", "CostTracker"]
