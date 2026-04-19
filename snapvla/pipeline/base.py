"""Pipeline primitives with built-in observability.

Every Stage run emits a StageTrace capturing inputs, outputs, shapes, dtypes,
and wall-clock latency. That is the core "see the data flow" feature of
SnapVLA — the whole point of using the pipeline abstraction for a learning
project is making what happens at each step inspectable.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class BaseContext(BaseModel):
    """Framework base for user-defined pipeline data contexts.

    `arbitrary_types_allowed` lets stages drop numpy arrays / torch tensors
    directly onto the context. `extra="forbid"` turns field-name typos into
    immediate ValidationErrors instead of silent dangling attributes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


TContext = TypeVar("TContext", bound=BaseContext)


@dataclass
class StageTrace:
    """One stage's execution record — the observability event.

    Stages auto-populate this via VLAPipeline.run_once(). User code usually
    only reads it (to print, log, or ship over the wire).
    """

    stage_name: str
    started_at: float
    latency_ms: float
    inputs_snapshot: dict[str, str] = field(default_factory=dict)
    outputs_snapshot: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage_name,
            "t0": self.started_at,
            "ms": round(self.latency_ms, 3),
            "in": self.inputs_snapshot,
            "out": self.outputs_snapshot,
            "err": self.error,
        }


class Stage(Generic[TContext]):
    """Single-responsibility unit of computation.

    Subclass and override `process()`. Declare `required_inputs` and
    `provided_outputs` as context field names so `VLAPipeline.build()` can
    validate wiring before any execution happens.
    """

    name: str = "Unnamed Stage"
    required_inputs: list[str] = []
    provided_outputs: list[str] = []

    def process(self, ctx: TContext) -> TContext:
        raise NotImplementedError(f"Stage '{self.name}' must implement process().")


def _snapshot(ctx: BaseContext, fields: list[str]) -> dict[str, str]:
    """Summarise selected fields of the context for trace logging.

    For numpy/torch-like values we record shape + dtype rather than the full
    tensor, so traces stay small and readable.
    """
    snap: dict[str, str] = {}
    for name in fields:
        value = getattr(ctx, name, None)
        if value is None:
            snap[name] = "None"
            continue
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if shape is not None:
            snap[name] = f"{type(value).__name__}{tuple(shape)} {dtype}"
        elif isinstance(value, (bytes, bytearray)):
            snap[name] = f"bytes[{len(value)}]"
        elif isinstance(value, str):
            snap[name] = f"str[{len(value)}]={value[:40]!r}"
        else:
            snap[name] = f"{type(value).__name__}={value!r}"[:80]
    return snap


TraceSink = Callable[[StageTrace], None]


class VLAPipeline(Generic[TContext]):
    """Ordered sequence of stages sharing a typed context.

    build() runs a topology check (every required_inputs field must be
    produced by an earlier stage or declared on the context schema).
    run_once() instantiates the context and drives all stages, emitting a
    StageTrace per stage to the configured trace_sink.
    """

    def __init__(
        self,
        context_schema: type[TContext],
        *,
        trace_sink: TraceSink | None = None,
    ) -> None:
        self.context_schema = context_schema
        self.stages: list[Stage[TContext]] = []
        self._is_built = False
        self._trace_sink = trace_sink or self._default_sink

    @staticmethod
    def _default_sink(trace: StageTrace) -> None:
        logger.info("STAGE %s +%.2fms in=%s out=%s err=%s",
                    trace.stage_name, trace.latency_ms,
                    trace.inputs_snapshot, trace.outputs_snapshot, trace.error)

    def add_stage(self, stage: Stage[TContext]) -> VLAPipeline[TContext]:
        self.stages.append(stage)
        self._is_built = False
        return self

    def build(self) -> VLAPipeline[TContext]:
        """Validate stage-to-stage field wiring.

        Seeds the available-fields set from ALL declared fields on the context
        schema — including required ones (those without defaults). Required
        fields are assumed to be supplied by the caller via ``initial_data``
        passed to ``run_once()``; if they are missing at runtime, Pydantic will
        raise a ``ValidationError`` there. Topology validation only checks that
        each stage's ``required_inputs`` are either declared on the context or
        produced by an earlier stage.
        """
        available: set[str] = set(self.context_schema.model_fields)
        for stage in self.stages:
            missing = [f for f in stage.required_inputs if f not in available]
            if missing:
                raise ValueError(
                    f"Stage '{stage.name}' requires {missing}; not produced by any "
                    f"earlier stage and not declared on '{self.context_schema.__name__}'. "
                    f"Available: {sorted(available)}"
                )
            available.update(stage.provided_outputs)
        self._is_built = True
        return self

    def run_once(
        self,
        initial_data: dict[str, Any] | None = None,
    ) -> tuple[TContext, list[StageTrace]]:
        if not self._is_built:
            raise RuntimeError("Pipeline not built. Call build() first.")
        ctx = self.context_schema(**(initial_data or {}))
        traces: list[StageTrace] = []
        for stage in self.stages:
            t0 = time.perf_counter()
            in_snap = _snapshot(ctx, stage.required_inputs)
            error: str | None = None
            try:
                ctx = stage.process(ctx)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                dt = (time.perf_counter() - t0) * 1000.0
                out_snap = _snapshot(ctx, stage.provided_outputs)
                trace = StageTrace(
                    stage_name=stage.name,
                    started_at=t0,
                    latency_ms=dt,
                    inputs_snapshot=in_snap,
                    outputs_snapshot=out_snap,
                    error=error,
                )
                traces.append(trace)
                self._trace_sink(trace)
        return ctx, traces
