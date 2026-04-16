# snapvla.pipeline

LangGraph-style generic data bus for SnapVLA pipelines.

---

## Overview

`snapvla.pipeline` is the execution backbone of SnapVLA. It provides a
**typed, ordered sequence of stages** that pass a shared context object from
one step to the next — similar in spirit to LangGraph's node/edge model, but
without the graph overhead for linear VLA inference loops.

Every frame or inference step is represented as a single context object.
Stages read from it and write to it. The pipeline's job is to wire those
stages together, validate the wiring at build time, and then drive execution.

---

## Three-layer design

```
Layer 1 — BaseContext (framework)
    └── Layer 2 — UserContext  (user-defined, e.g. PillBottleContext)
            └── Layer 3 — Stages + VLAPipeline  (user-defined logic)
```

### Layer 1: `BaseContext`

The framework provides a single Pydantic `BaseModel` subclass with two
configuration flags that matter for robotics:

- `arbitrary_types_allowed=True` — numpy arrays, torch tensors, and other
  non-Pydantic types can be stored directly on the context.
- `extra="forbid"` — writing to any field not declared on the subclass raises
  a `ValidationError` immediately. This turns typos like `ctx.rgb_iamge` into
  loud failures rather than silent dangling attributes.

The framework declares **no fields**. It makes no assumptions about what data
flows through your pipeline.

### Layer 2: User context

You subclass `BaseContext` and declare every field your pipeline will use.
Pydantic validates types on assignment. Fields with defaults are considered
"pre-available" by the topology validator.

### Layer 3: Stages and pipeline

You subclass `Stage[YourContext]`, override `process()`, and list
`required_inputs` / `provided_outputs` as field-name strings. `VLAPipeline`
holds an ordered list of stages, validates wiring with `build()`, and executes
them one-by-one in `run_once()`.

---

## Why this design

| Goal | How it is achieved |
|---|---|
| Framework does not guess user data | `BaseContext` is empty; all fields live on the user's subclass |
| Pydantic validation at every assignment | `BaseModel` with `extra="forbid"` |
| IDE autocomplete and type safety | `Stage[TContext]` and `VLAPipeline[TContext]` are generic over the user's context type |
| Wiring errors caught before the robot moves | `build()` topology check at construction time |
| Minimal boilerplate | `Stage` is not an ABC; `add_stage()` returns `self` for chaining |

---

## Full usage example

```python
from typing import Optional
import numpy as np
from pydantic import Field
from snapvla.pipeline import BaseContext, Stage, VLAPipeline


# ---------------------------------------------------------------------------
# 1. Define a custom context
# ---------------------------------------------------------------------------

class PillBottleContext(BaseContext):
    """All data that flows through one inference step."""

    # Supplied at construction time via initial_data
    robot_id: str

    # Populated by stages — Optional + default=None means they are
    # "pre-available" from the topology validator's perspective
    rgb_image:   Optional[np.ndarray] = None
    depth_image: Optional[np.ndarray] = None
    point_cloud: Optional[np.ndarray] = None
    action:      Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# 2. Define stages
# ---------------------------------------------------------------------------

class RGBCaptureStage(Stage[PillBottleContext]):
    name = "RGBCapture"
    required_inputs  = []               # needs nothing pre-populated
    provided_outputs = ["rgb_image", "depth_image"]

    def process(self, ctx: PillBottleContext) -> PillBottleContext:
        # Replace with your real sensor call
        ctx.rgb_image   = np.zeros((480, 640, 3), dtype=np.uint8)
        ctx.depth_image = np.zeros((480, 640),    dtype=np.float32)
        return ctx


class DepthStage(Stage[PillBottleContext]):
    name = "DepthToPointCloud"
    required_inputs  = ["depth_image"]
    provided_outputs = ["point_cloud"]

    def process(self, ctx: PillBottleContext) -> PillBottleContext:
        # Replace with real projection logic
        h, w = ctx.depth_image.shape
        ctx.point_cloud = np.zeros((h * w, 3), dtype=np.float32)
        return ctx


class PolicyStage(Stage[PillBottleContext]):
    name = "PolicyInference"
    required_inputs  = ["rgb_image", "point_cloud"]
    provided_outputs = ["action"]

    def process(self, ctx: PillBottleContext) -> PillBottleContext:
        # Replace with your VLA model call
        ctx.action = np.zeros(7, dtype=np.float32)
        return ctx


class ActionStage(Stage[PillBottleContext]):
    name = "ActionExecution"
    required_inputs  = ["action"]
    provided_outputs = []

    def process(self, ctx: PillBottleContext) -> PillBottleContext:
        print(f"[{ctx.robot_id}] Sending action: {ctx.action}")
        return ctx


# ---------------------------------------------------------------------------
# 3. Assemble and run the pipeline
# ---------------------------------------------------------------------------

pipeline = (
    VLAPipeline(PillBottleContext)
    .add_stage(RGBCaptureStage())
    .add_stage(DepthStage())
    .add_stage(PolicyStage())
    .add_stage(ActionStage())
)

pipeline.build()   # topology check — raises ValueError on bad wiring

ctx = pipeline.run_once({"robot_id": "arm_0"})
print(ctx.action)  # numpy array output from the policy
```

---

## How to create a new custom context

1. Import `BaseContext`.
2. Subclass it and declare all fields with standard Pydantic syntax.
3. Fields that must be supplied before the pipeline starts (e.g. `robot_id`)
   should have **no default**. Fields that stages will fill in should default
   to `None` (or an appropriate sentinel).

```python
from snapvla.pipeline import BaseContext

class MyContext(BaseContext):
    robot_id:    str                       # required at construction
    rgb_image:   Optional[np.ndarray] = None   # filled by a capture stage
    action:      Optional[np.ndarray] = None   # filled by a policy stage
```

---

## How to create a new stage

1. Import `Stage` and your context class.
2. Subclass `Stage[YourContext]`.
3. Set the `name` class attribute to a human-readable string.
4. List field names in `required_inputs` (what you read) and
   `provided_outputs` (what you write).
5. Override `process()` — mutate `ctx` and return it.

```python
from snapvla.pipeline import Stage

class NormalisationStage(Stage[MyContext]):
    name = "Normalise"
    required_inputs  = ["rgb_image"]
    provided_outputs = ["rgb_image"]   # overwrites the same field

    def process(self, ctx: MyContext) -> MyContext:
        ctx.rgb_image = ctx.rgb_image.astype(np.float32) / 255.0
        return ctx
```

---

## Topology validation — what `build()` checks

`VLAPipeline.build()` performs a single forward pass through the ordered
stage list before any execution occurs.

**Starting set of available fields**

The validator seeds the available-fields set with every field on the context
schema that has a default value (i.e. fields where
`field_info.is_required() == False`). These fields are guaranteed to exist on
a freshly constructed context even when `initial_data` is empty.

> Note: required fields (no default) are *not* in the initial set. If a stage
> lists them in `required_inputs` but no earlier stage produces them, `build()`
> raises a `ValueError`. Supply them through `initial_data` in `run_once()` —
> they will be present on the context from the moment it is constructed, but
> the topology validator intentionally does not treat them as pre-available,
> because their presence depends entirely on the caller remembering to pass
> them. List them in the first stage's `required_inputs` only when you want an
> explicit declaration that the stage depends on caller-supplied data.

**Per-stage check**

For each stage in order:

1. Every name in `required_inputs` must already be in the available-fields
   set. If any name is missing, `build()` raises:
   ```
   ValueError: Stage 'PolicyInference' requires input(s) ['point_cloud']
   that are not provided by any preceding stage and have no default value
   on 'PillBottleContext'. Available fields at this point: ['rgb_image']
   ```
2. Every name in `provided_outputs` is added to the available-fields set so
   that subsequent stages can satisfy their own `required_inputs`.

**What this catches**

- A stage that reads a field no earlier stage (and no default) provides.
- Two stages whose order was accidentally swapped.
- A stage that was removed but whose consumers were not updated.

`build()` does **not** check:
- Whether `provided_outputs` fields actually exist on the context schema
  (Pydantic's `extra="forbid"` handles that at runtime).
- Whether the data types are correct (Pydantic type annotations handle that).
