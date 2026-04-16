"""
Core pipeline primitives for SnapVLA.

Provides the three building blocks of every SnapVLA pipeline:

    BaseContext   – Pydantic model that the user subclasses to define the data
                    that flows between stages.  ``extra="forbid"`` means any
                    field not declared on the subclass raises a validation error
                    immediately, keeping the contract explicit.

    Stage         – Lightweight generic unit of computation.  Receives the
                    current context, mutates or replaces it, and returns it.
                    Class-level ``required_inputs`` / ``provided_outputs`` lists
                    let VLAPipeline.build() validate wiring at construction time
                    rather than at runtime.

    VLAPipeline   – Ordered container of stages.  Accepts the user's context
                    schema *type* and instantiates it from ``initial_data`` in
                    ``run_once()``.  ``build()`` performs a forward-pass topology
                    check so broken wiring is caught before any inference loop.
"""

from __future__ import annotations

from typing import Generic, List, TypeVar

from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# BaseContext
# ---------------------------------------------------------------------------


class BaseContext(BaseModel):
    """Framework-level base for all pipeline data contexts.

    Users subclass this to declare every field that will travel through the
    pipeline.  Pydantic enforces types on assignment, so stages always see
    well-typed data.

    Design notes:
        - ``arbitrary_types_allowed=True`` is required because VLA pipelines
          routinely store ``numpy.ndarray`` objects (images, point clouds, etc.)
          directly on the context.
        - ``extra="forbid"`` ensures that a typo in a stage (e.g. writing to
          ``ctx.rgb_iamge`` instead of ``ctx.rgb_image``) raises a
          ``ValidationError`` immediately rather than silently creating a
          dangling attribute.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


# ---------------------------------------------------------------------------
# TypeVar
# ---------------------------------------------------------------------------

TContext = TypeVar("TContext", bound=BaseContext)
"""Generic type variable bound to BaseContext.

Used to parameterise both Stage and VLAPipeline so that IDE type checkers
can infer the concrete context type flowing through a specific pipeline.
"""


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


class Stage(Generic[TContext]):
    """Generic, single-responsibility unit of computation in a VLA pipeline.

    Subclass this and override ``process()`` to implement any step: image
    decoding, depth-to-pointcloud projection, policy inference, action
    execution, logging, etc.

    Class attributes:
        name (str): Human-readable identifier used in error messages and logs.
        required_inputs (List[str]): Context field names that this stage reads.
            ``VLAPipeline.build()`` checks that each name is either a default-
            bearing field on the context schema *or* is produced by an earlier
            stage's ``provided_outputs``.
        provided_outputs (List[str]): Context field names that this stage
            writes.  Used by the topology validator to prove downstream
            requirements are met.

    Example::

        class DepthStage(Stage[PillBottleContext]):
            name = "DepthToPointCloud"
            required_inputs = ["depth_image"]
            provided_outputs = ["point_cloud"]

            def process(self, ctx: PillBottleContext) -> PillBottleContext:
                ctx.point_cloud = depth_to_xyz(ctx.depth_image)
                return ctx
    """

    name: str = "Unnamed Stage"
    required_inputs: List[str] = []
    provided_outputs: List[str] = []

    def process(self, ctx: TContext) -> TContext:
        """Execute this stage's logic on the shared context.

        Args:
            ctx: The current pipeline context.  Stages may mutate fields
                 in-place or return a new context instance.

        Returns:
            The (possibly modified) context, which is passed to the next stage.

        Raises:
            NotImplementedError: Always, until the subclass overrides this.
        """
        raise NotImplementedError(
            f"Stage '{self.name}' must implement process()."
        )


# ---------------------------------------------------------------------------
# VLAPipeline
# ---------------------------------------------------------------------------


class VLAPipeline(Generic[TContext]):
    """Ordered sequence of stages that transform a shared context.

    The pipeline is generic over ``TContext`` so that type checkers can
    verify stage compatibility at authoring time.

    Usage::

        pipeline = (
            VLAPipeline(PillBottleContext)
            .add_stage(RGBCaptureStage())
            .add_stage(DepthStage())
            .add_stage(PolicyStage())
            .add_stage(ActionStage())
        )
        pipeline.build()          # topology check — raises ValueError on bad wiring
        ctx = pipeline.run_once({"robot_id": "arm_0"})

    Args:
        context_schema: The *class* (not an instance) of the user's
            ``BaseContext`` subclass.  ``run_once()`` calls
            ``context_schema(**initial_data)`` to create the starting context.
    """

    def __init__(self, context_schema: type[TContext]) -> None:
        self.context_schema = context_schema
        self.stages: List[Stage[TContext]] = []
        self._is_built: bool = False

    # ------------------------------------------------------------------
    # Assembly API
    # ------------------------------------------------------------------

    def add_stage(self, stage: Stage[TContext]) -> "VLAPipeline[TContext]":
        """Append a stage to the end of the pipeline.

        Marks the pipeline as un-built so that ``build()`` must be called
        again before ``run_once()`` if stages are added after an initial build.

        Args:
            stage: A ``Stage`` instance whose ``TContext`` matches this
                   pipeline's context type.

        Returns:
            ``self``, enabling method chaining::

                pipeline.add_stage(A()).add_stage(B()).add_stage(C())
        """
        self.stages.append(stage)
        self._is_built = False
        return self

    # ------------------------------------------------------------------
    # Build / topology validation
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Validate stage wiring and freeze the pipeline for execution.

        Performs a single forward pass through the stage list, maintaining a
        running set of *available fields* that starts with every field
        that has a default value on the context schema (i.e. fields that will
        exist on a freshly constructed context without being supplied via
        ``initial_data``).

        For each stage in order:
            1. Every name in ``required_inputs`` must already be in the
               available-fields set.  If not, a ``ValueError`` is raised that
               names the offending stage and the missing field.
            2. Every name in ``provided_outputs`` is added to the
               available-fields set so that subsequent stages can depend on it.

        This catches the two most common wiring mistakes:
            - A stage that reads a field no earlier stage produces.
            - Two stages whose order was accidentally swapped.

        Raises:
            ValueError: If any stage's ``required_inputs`` cannot be satisfied
                by prior stages' outputs or by the context schema's defaults.
        """
        # Collect every field declared on the context schema that carries a
        # default (meaning a freshly constructed context will already have it).
        # Pydantic v2 exposes this through model_fields; a field has a default
        # when its FieldInfo.is_required() returns False.
        available: set[str] = {
            name
            for name, field_info in self.context_schema.model_fields.items()
            if not field_info.is_required()
        }

        for stage in self.stages:
            missing = [
                field
                for field in stage.required_inputs
                if field not in available
            ]
            if missing:
                raise ValueError(
                    f"Stage '{stage.name}' requires input(s) {missing} "
                    f"that are not provided by any preceding stage and have "
                    f"no default value on '{self.context_schema.__name__}'. "
                    f"Available fields at this point: {sorted(available)}"
                )
            available.update(stage.provided_outputs)

        self._is_built = True

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_once(self, initial_data: dict | None = None) -> TContext:
        """Instantiate the context and run all stages once in order.

        Args:
            initial_data: Keyword arguments forwarded to the context
                constructor.  Use this to supply required fields that no
                stage produces (e.g. the initial sensor frame or a robot ID).
                Defaults to an empty dict so pipelines with all-default
                contexts need no argument.

        Returns:
            The fully processed context after all stages have run.

        Raises:
            RuntimeError: If ``build()`` has not been called since the last
                ``add_stage()`` call.
        """
        if not self._is_built:
            raise RuntimeError(
                "Pipeline has not been built. Call build() before run_once()."
            )
        ctx: TContext = self.context_schema(**(initial_data or {}))
        for stage in self.stages:
            ctx = stage.process(ctx)
        return ctx
