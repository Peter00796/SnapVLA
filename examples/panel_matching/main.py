"""Panel Matching — entry point.

This file demonstrates the one canonical way to construct and run the
Panel Matching pipeline.  Read the files in the order listed in README.md
to understand how a new SnapVLA pipeline is built from scratch.
"""

from snapvla.pipeline import VLAPipeline

from examples.panel_matching.context import PanelMatchingContext
from examples.panel_matching.mock_data import make_mock_initial_data
from examples.panel_matching.stages import (
    ContourExtractStage,
    SAM3SegmentStage,
    UmeyamaMatchStage,
    UndistortStage,
)


def build_pipeline() -> VLAPipeline[PanelMatchingContext]:
    """Wire every stage into an ordered, build-validated pipeline."""
    return (
        VLAPipeline(PanelMatchingContext)
        .add_stage(UndistortStage())
        .add_stage(SAM3SegmentStage())
        .add_stage(ContourExtractStage())
        .add_stage(UmeyamaMatchStage())
    )


def main() -> None:
    pipeline = build_pipeline()
    pipeline.build()
    print(pipeline.describe())

    result = pipeline.run_once(make_mock_initial_data())

    print(f"\nPipeline produced {len(result.panel_poses)} panel poses:")
    for p in result.panel_poses:
        print(f"  {p.panel_id}: x={p.x:.1f}  y={p.y:.1f}  theta={p.theta:.2f}  conf={p.confidence}")

    print()
    print(pipeline.last_profile.report())


if __name__ == "__main__":
    main()
