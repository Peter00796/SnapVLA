"""Smoke test for VLAPipeline topology fix and end-to-end field flow.

Verifies that build() no longer rejects required context fields (frame_id,
prompt, jpeg) that are supplied at runtime via run_once(initial_data={...}).
Uses a FakeVLMStage to avoid loading torch/transformers.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from snapvla.pipeline import Stage, VLAPipeline
from snapvla.server.contexts import InferenceContext
from snapvla.server.stages import DecodeJpegStage, LogStage


class FakeVLMStage(Stage[InferenceContext]):
    name = "FakeVLM"
    required_inputs = ["pil_image", "prompt"]
    provided_outputs = ["text"]

    def process(self, ctx: InferenceContext) -> InferenceContext:
        ctx.text = f"fake-text-for-{ctx.frame_id}"
        return ctx


def _make_jpeg(width: int = 64, height: int = 64) -> bytes:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :, 0] = 128  # non-trivial red channel so PIL round-trips cleanly
    pil = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG")
    return buf.getvalue()


def test_build_accepts_required_context_fields() -> None:
    jpeg_bytes = _make_jpeg()

    pipeline: VLAPipeline[InferenceContext] = (
        VLAPipeline(InferenceContext)
        .add_stage(DecodeJpegStage())
        .add_stage(FakeVLMStage())
        .add_stage(LogStage())
    )

    # Must not raise — this is the core regression being tested.
    pipeline.build()

    ctx, traces = pipeline.run_once({"frame_id": 7, "prompt": "test", "jpeg": jpeg_bytes})

    assert ctx.text == "fake-text-for-7"

    assert len(traces) == 3
    assert [t.stage_name for t in traces] == ["DecodeJPEG", "FakeVLM", "Log"]

    for trace in traces:
        assert trace.latency_ms >= 0

    # DecodeJpegStage outputs rgb; _snapshot formats ndarray shape as "(64, 64, 3)".
    assert "(64, 64, 3)" in traces[0].outputs_snapshot["rgb"]
