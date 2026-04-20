# Panel Matching Example

A reference implementation of a SnapVLA pipeline.  Use this as the template
when starting a new problem.

## The standard workflow

When you get a new problem that fits the "multi-stage typed data flow"
pattern, write the files in this order:

1. **`types.py`** — Define domain dataclasses / value objects.
   No framework dependencies.  Pure Python data structures that describe
   the problem domain (panels, calibrations, poses, etc.).

2. **`context.py`** — Define the Pydantic `BaseContext` subclass
   that carries all data through the pipeline.  Use the two-tier convention:
   - Fields with `= None` default are caller-supplied (pre-available to topology).
   - Fields without a default must be earned by a stage's `provided_outputs`.

3. **`stages/`** — One file per stage.  Each stage declares
   `required_inputs`, `provided_outputs`, and a `process()` method.
   Keep each file self-contained: import only what it needs.

4. **`mock_data.py`** — Fixtures for testing and demos without
   real hardware.  Returns a plain `dict` ready for `pipeline.run_once()`.

5. **`main.py`** — Imports everything and wires the pipeline
   with `add_stage` chains.  Nothing else belongs here.

Optional:

6. **`scenarios.py`** — Failure-mode demos (build-time errors you
   expect the pipeline to catch).  Useful for documentation and unit tests.

## File-by-file summary

| File | Role |
|---|---|
| `types.py` | Domain dataclasses: `PanelCAD`, `CameraCalibration`, `PanelPose` |
| `context.py` | `PanelMatchingContext(BaseContext)` — the typed data bus |
| `stages/undistort.py` | `UndistortStage` — removes lens distortion |
| `stages/sam3_segment.py` | `SAM3SegmentStage` — SAM3 instance segmentation |
| `stages/contour_extract.py` | `ContourExtractStage` — mask → polygon contours |
| `stages/umeyama_match.py` | `UmeyamaMatchStage` — Umeyama alignment + Hungarian assignment |
| `stages/__init__.py` | Re-exports all four stage classes |
| `mock_data.py` | `make_mock_initial_data()` — hardware-free fixtures |
| `scenarios.py` | Three deliberate failure demos (wrong order, missing producer, typo) |
| `main.py` | Clean happy-path entry point |
| `__init__.py` | Package exports: `build_pipeline`, context, and domain types |

## Running the demo

```bash
cd c:/Users/29838/Documents/Researches/SnapVLA/SnapVLA
PYTHONPATH=. ../.venv/Scripts/python.exe -m examples.panel_matching.main
```

The output ends with a per-stage profile table showing which stage dominates
runtime — no instrumentation needed in your own code.

To see failure-mode demos:

```bash
PYTHONPATH=. ../.venv/Scripts/python.exe -m examples.panel_matching.scenarios
```

## Mapping to Solomon's real xxl-xcs-parser

Each mock stage corresponds to a real service in `xxl-xcs-parser/backend/services/`.
Swap the mock body for the real service call to deploy in production:

| Stage | Real service |
|---|---|
| `UndistortStage` | `backend/services/undistortion.py` |
| `SAM3SegmentStage` | `backend/services/detection.py` (SAM3 GPU inference) |
| `ContourExtractStage` | `cv2.findContours` + `cv2.approxPolyDP` per mask |
| `UmeyamaMatchStage` | `backend/services/umeyama_matcher.py` + `ransac_match.py` + `icp.py` |
