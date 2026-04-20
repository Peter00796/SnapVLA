"""Panel Matching example package — exports the public types and context.

Import ``build_pipeline`` directly from ``examples.panel_matching.main``
to avoid a circular import when running main as a module.
"""

from examples.panel_matching.types import CameraCalibration, PanelCAD, PanelPose
from examples.panel_matching.context import PanelMatchingContext

__all__ = [
    "PanelMatchingContext",
    "PanelCAD",
    "CameraCalibration",
    "PanelPose",
]
