"""SAM3SegmentStage — segments panels from the undistorted image via SAM3."""

from __future__ import annotations

import numpy as np

from snapvla.pipeline import Stage
from examples.panel_matching.context import PanelMatchingContext


class SAM3SegmentStage(Stage[PanelMatchingContext]):
    """Wraps backend/services/detection.py (SAM3 segmentation)."""

    name             = "SAM3Segment"
    required_inputs  = ["undistorted_image"]
    provided_outputs = ["sam_masks"]

    def process(self, ctx: PanelMatchingContext) -> PanelMatchingContext:
        # Real code: SAM3 model inference on CUDA GPU
        H, W, _ = ctx.undistorted_image.shape
        rng = np.random.default_rng(0)
        ctx.sam_masks = [rng.random((H, W)) > 0.5 for _ in range(3)]
        print(f"  [SAM3Segment]     produced {len(ctx.sam_masks)} masks ({H}x{W})")
        return ctx
