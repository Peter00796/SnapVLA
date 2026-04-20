"""Fixtures for running the pipeline without real hardware."""

from __future__ import annotations

import numpy as np

from examples.panel_matching.types import CameraCalibration, PanelCAD


def make_mock_initial_data() -> dict:
    """Construct the kind of initial_data an xxl-xcs-parser request would pass.

    Caller-supplied inputs are populated with real values.  Stage-produced
    intermediates and outputs are explicitly set to None — each stage will
    replace them as the pipeline executes.
    """
    return {
        # Caller-supplied inputs
        "raw_image": np.zeros((480, 640, 3), dtype=np.uint8),
        "calibration": CameraCalibration(
            K=np.eye(3),
            dist=np.zeros(5),
            resolution=(640, 480),
        ),
        "nesting_plan": [
            PanelCAD(
                panel_id=f"P{i:03d}",
                polygon=np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32),
                expected_position=(0.0, 0.0),
            )
            for i in range(5)
        ],
        "job_id": "DEMO-001",
        # Stage-produced fields: must be supplied (as None) because they have
        # no default on the schema.  Stages will fill them in during execution.
        "undistorted_image": None,
        "sam_masks": None,
        "extracted_contours": None,
        "panel_poses": None,
    }
