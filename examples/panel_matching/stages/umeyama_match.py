"""UmeyamaMatchStage — aligns extracted contours to CAD panels via Umeyama + Hungarian."""

from __future__ import annotations

from snapvla.pipeline import Stage
from examples.panel_matching.context import PanelMatchingContext
from examples.panel_matching.types import PanelPose


class UmeyamaMatchStage(Stage[PanelMatchingContext]):
    """Wraps backend/services/umeyama_matcher.py + ransac_match.py + icp.py."""

    name             = "UmeyamaMatch"
    required_inputs  = ["extracted_contours", "nesting_plan"]
    provided_outputs = ["panel_poses"]

    def process(self, ctx: PanelMatchingContext) -> PanelMatchingContext:
        # Real code: Umeyama alignment + Hungarian assignment against nesting plan
        n = min(len(ctx.extracted_contours), len(ctx.nesting_plan))
        ctx.panel_poses = [
            PanelPose(
                panel_id=ctx.nesting_plan[i].panel_id,
                x=100.0 * i,
                y=50.0,
                theta=0.0,
                confidence=0.95,
            )
            for i in range(n)
        ]
        print(f"  [UmeyamaMatch]    matched {n} panels")
        return ctx
