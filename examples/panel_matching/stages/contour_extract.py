"""ContourExtractStage — converts SAM3 masks into polygon contours."""

from __future__ import annotations

# Real dependency: import cv2

import numpy as np

from snapvla.pipeline import Stage
from examples.panel_matching.context import PanelMatchingContext


class ContourExtractStage(Stage[PanelMatchingContext]):
    """Wraps contour extraction from masks (uses cv2.findContours in production)."""

    name             = "ContourExtract"
    required_inputs  = ["sam_masks"]
    provided_outputs = ["extracted_contours"]

    def process(self, ctx: PanelMatchingContext) -> PanelMatchingContext:
        # Real code: cv2.findContours per mask, simplify with cv2.approxPolyDP
        ctx.extracted_contours = [
            np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32)
            for _ in ctx.sam_masks
        ]
        print(f"  [ContourExtract]  produced {len(ctx.extracted_contours)} contours")
        return ctx
