"""
snapvla.pipeline
================

LangGraph-style generic data bus for SnapVLA pipelines.

Public API
----------
BaseContext
    Pydantic base model for all pipeline data contexts.  Subclass this to
    define every field that flows between stages.  ``extra="forbid"`` means
    any undeclared field raises a ``ValidationError`` immediately.

TContext
    TypeVar bound to ``BaseContext``.  Used to parameterise ``Stage`` and
    ``VLAPipeline`` for IDE type inference.

Stage
    Generic base class for a single pipeline step.  Override ``process()``
    to implement computation.  Declare ``required_inputs`` and
    ``provided_outputs`` so that ``VLAPipeline.build()`` can validate wiring.

VLAPipeline
    Ordered container of stages.  Accepts the user's context schema *type*,
    validates wiring with ``build()``, and executes the full stage sequence
    with ``run_once()``.
"""

from snapvla.pipeline.base import BaseContext, Stage, TContext, VLAPipeline

__all__ = [
    "BaseContext",
    "TContext",
    "Stage",
    "VLAPipeline",
]
