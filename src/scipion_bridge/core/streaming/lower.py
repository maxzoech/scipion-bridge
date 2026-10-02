"""Polymorphic lowering engine for transforming surface Node DAGs to IR DAGs."""

from __future__ import annotations

from .node import lower, LoweringContext


__all__ = ["lower", "LoweringContext"]
