"""Re-exports all pipeline stages for clean top-level imports."""

from examples.panel_matching.stages.undistort import UndistortStage
from examples.panel_matching.stages.sam3_segment import SAM3SegmentStage
from examples.panel_matching.stages.contour_extract import ContourExtractStage
from examples.panel_matching.stages.umeyama_match import UmeyamaMatchStage

__all__ = [
    "UndistortStage",
    "SAM3SegmentStage",
    "ContourExtractStage",
    "UmeyamaMatchStage",
]
