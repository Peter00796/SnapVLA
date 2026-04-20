"""Failure-mode demos that show what build() catches before any stage runs."""

from __future__ import annotations

from typing import Callable

from snapvla.pipeline import Stage, VLAPipeline

from examples.panel_matching.context import PanelMatchingContext
from examples.panel_matching.stages import (
    ContourExtractStage,
    SAM3SegmentStage,
    UndistortStage,
    UmeyamaMatchStage,
)


def demo_wrong_stage_order() -> None:
    """build() catches stages added in wrong order."""
    print("\n" + "=" * 72)
    print("Scenario 1: build() catches stages added in wrong order")
    print("=" * 72)

    bad = (
        VLAPipeline(PanelMatchingContext)
        .add_stage(UmeyamaMatchStage())    # needs contours — but none produced yet
        .add_stage(UndistortStage())
        .add_stage(SAM3SegmentStage())
        .add_stage(ContourExtractStage())
    )

    try:
        bad.build()
        print("UNEXPECTED: build() should have raised ValueError.")
    except ValueError as e:
        print(f"Caught expected error:\n{e}")


def demo_missing_producer() -> None:
    """build() catches a missing producer stage."""
    print("\n" + "=" * 72)
    print("Scenario 2: build() catches missing producer stage")
    print("=" * 72)

    # Forgot to add ContourExtractStage — UmeyamaMatch has nothing to match
    incomplete = (
        VLAPipeline(PanelMatchingContext)
        .add_stage(UndistortStage())
        .add_stage(SAM3SegmentStage())
        # .add_stage(ContourExtractStage())   <-- missing
        .add_stage(UmeyamaMatchStage())
    )

    try:
        incomplete.build()
        print("UNEXPECTED: build() should have raised ValueError.")
    except ValueError as e:
        print(f"Caught expected error:\n{e}")


def demo_typo_in_provided_outputs() -> None:
    """build() catches a typo in provided_outputs via schema field validation."""
    print("\n" + "=" * 72)
    print("Scenario 3: build() catches typo in provided_outputs")
    print("=" * 72)

    class TypoUmeyamaStage(Stage[PanelMatchingContext]):
        name             = "TypoUmeyama"
        required_inputs  = ["extracted_contours", "nesting_plan"]
        provided_outputs = ["panel_posess"]   # typo!

        def process(self, ctx: PanelMatchingContext) -> PanelMatchingContext:
            return ctx

    bad = (
        VLAPipeline(PanelMatchingContext)
        .add_stage(UndistortStage())
        .add_stage(SAM3SegmentStage())
        .add_stage(ContourExtractStage())
        .add_stage(TypoUmeyamaStage())
    )

    try:
        bad.build()
        print("UNEXPECTED: build() should have raised ValueError.")
    except ValueError as e:
        print(f"Caught expected error:\n{e}")


def main() -> None:
    demo_wrong_stage_order()
    demo_missing_producer()
    demo_typo_in_provided_outputs()
    print("\n" + "=" * 72)
    print("All failure-mode scenarios finished — every error was caught at build() time.")
    print("=" * 72)


if __name__ == "__main__":
    main()
