"""Pipeline context schemas used by the inference server."""

from __future__ import annotations

from typing import Any

from snapvla.pipeline import BaseContext


class InferenceContext(BaseContext):
    """Data that flows through one VLM inference call.

    Fields with defaults are filled in by stages. `jpeg` and `prompt` are the
    only two pieces of input we take from the wire; everything else is
    produced inside the pipeline.
    """

    frame_id: int
    prompt: str
    jpeg: bytes

    rgb: Any | None = None
    pil_image: Any | None = None
    text: str = ""
    action: list[float] | None = None
