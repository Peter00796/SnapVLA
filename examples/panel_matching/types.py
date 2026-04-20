"""Domain value objects for the panel-matching pipeline. No snapvla dependencies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PanelCAD:
    """One panel entry parsed from the CNC nesting file (XCS/XXL)."""

    panel_id: str
    polygon: np.ndarray             # (N, 2) CAD vertices in mm
    expected_position: tuple[float, float]


@dataclass
class CameraCalibration:
    """Intrinsics + distortion coefficients for the overhead camera."""

    K: np.ndarray                   # 3x3 intrinsic matrix
    dist: np.ndarray                # distortion coefficients
    resolution: tuple[int, int]     # (width, height)


@dataclass
class PanelPose:
    """Final per-panel output the robot consumes."""

    panel_id: str
    x: float
    y: float
    theta: float                    # radians
    confidence: float
