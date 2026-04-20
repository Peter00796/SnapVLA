"""PanelMatchingContext — the typed data bus that flows through the pipeline."""

from __future__ import annotations

from typing import Optional

import numpy as np

from snapvla.pipeline import BaseContext
from examples.panel_matching.types import CameraCalibration, PanelCAD, PanelPose


class PanelMatchingContext(BaseContext):
    """All data that flows through one panel-matching job.

    Convention used throughout this context:
      - Caller-supplied inputs and metadata have a default (``= None``).
        The topology validator treats them as pre-available.
      - Stage-produced intermediates and outputs have NO default.
        The topology validator requires that an earlier stage declares them
        in ``provided_outputs`` before a later stage can list them in
        ``required_inputs``.  Pass ``None`` for all of these in
        ``initial_data`` when calling ``run_once()``; each stage will replace
        the ``None`` with a real value as the pipeline executes.
    """

    # ----- Caller-supplied inputs (default = None; topology-pre-available) -----
    raw_image:    Optional[np.ndarray]        = None
    calibration:  Optional[CameraCalibration] = None
    nesting_plan: Optional[list[PanelCAD]]    = None

    # ----- Stage-produced intermediates (no default; must be earned) -----
    undistorted_image:  Optional[np.ndarray]
    sam_masks:          Optional[list[np.ndarray]]
    extracted_contours: Optional[list[np.ndarray]]

    # ----- Stage-produced final output (no default; must be earned) -----
    panel_poses: Optional[list[PanelPose]]

    # ----- Metadata (default = None; topology-pre-available) -----
    job_id: Optional[str] = None
