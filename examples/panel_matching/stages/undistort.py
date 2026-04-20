"""UndistortStage — removes lens distortion from the raw camera image."""

from __future__ import annotations

# Real dependency: import cv2

import numpy as np

from snapvla.pipeline import Stage
from examples.panel_matching.context import PanelMatchingContext


class UndistortStage(Stage[PanelMatchingContext]):
    """Wraps backend/services/undistortion.py."""

    name             = "Undistort"
    required_inputs  = ["raw_image", "calibration"]
    provided_outputs = ["undistorted_image"]

    def process(self, ctx: PanelMatchingContext) -> PanelMatchingContext:
        # Real code: cv2.undistort(ctx.raw_image, ctx.calibration.K, ctx.calibration.dist)
        print(f"  [Undistort]       input  shape={ctx.raw_image.shape}")
        ctx.undistorted_image = ctx.raw_image.copy()
        print(f"  [Undistort]       output shape={ctx.undistorted_image.shape}")
        return ctx
