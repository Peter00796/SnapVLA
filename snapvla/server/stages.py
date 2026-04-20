"""Server-side pipeline stages.

Each stage is intentionally small so the StageTrace output surfaces the data
shape after every step — the "see the data flow" goal of SnapVLA.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
from PIL import Image

from snapvla.pipeline import Stage
from snapvla.server.contexts import InferenceContext

logger = logging.getLogger(__name__)


class DecodeJpegStage(Stage[InferenceContext]):
    name = "DecodeJPEG"
    required_inputs = ["jpeg"]
    provided_outputs = ["rgb", "pil_image"]

    def process(self, ctx: InferenceContext) -> InferenceContext:
        pil = Image.open(io.BytesIO(ctx.jpeg)).convert("RGB")
        ctx.pil_image = pil
        ctx.rgb = np.asarray(pil, dtype=np.uint8)
        return ctx


class LogStage(Stage[InferenceContext]):
    name = "Log"
    required_inputs = ["text"]
    provided_outputs = []

    def process(self, ctx: InferenceContext) -> InferenceContext:
        logger.info("frame=%d prompt=%r text=%r", ctx.frame_id, ctx.prompt, ctx.text)
        return ctx


class MoondreamStage(Stage[InferenceContext]):
    """Wraps Moondream2 as a pipeline stage.

    Loads lazily on first process() so pipeline construction is cheap. Uses
    GPU + fp16 when available. Meant as the v1 VLM; swap in another model by
    replacing this stage only.
    """

    name = "Moondream"
    required_inputs = ["pil_image", "prompt"]
    provided_outputs = ["text"]

    def __init__(self, model_id: str = "vikhyatk/moondream2", revision: str = "2024-08-26") -> None:
        self.model_id = model_id
        self.revision = revision
        self._model: Any = None
        self._tokenizer: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device_map = "cuda" if torch.cuda.is_available() else "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=self.revision,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device_map,
        )
        logger.info("Moondream loaded (device=%s dtype=%s).", device_map, dtype)

    def process(self, ctx: InferenceContext) -> InferenceContext:
        self._ensure_loaded()
        enc = self._model.encode_image(ctx.pil_image)
        answer = self._model.answer_question(enc, ctx.prompt or "Describe this image.", self._tokenizer)
        ctx.text = answer
        return ctx


# Minimal registry so callers can resolve a stage by name without importing
# the heavy VLA deps. stages_registry extends this dict when available.
MODEL_STAGES: dict[str, type[Stage[InferenceContext]]] = {"moondream": MoondreamStage}
