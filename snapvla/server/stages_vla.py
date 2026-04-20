"""OpenVLA-OFT pipeline stage.

Only importable inside the server-vla venv (requires transformers==4.54.1,
bitsandbytes, timm 1.x). Deliberately kept in the shared snapvla package so
the stage registry can discover it without a separate package install, but the
import will fail gracefully in the server venv because bitsandbytes is absent.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import torch

from snapvla.pipeline import Stage
from snapvla.server.contexts import InferenceContext

logger = logging.getLogger(__name__)

# Default weights location matches the known-working SimpleVLA spike.
# Callers override via SNAPVLA_OPENVLA_PATH so the path is never hardcoded
# inside shared library code that runs on the Pi or in the Moondream venv.
_DEFAULT_MODEL_PATH = r"C:\Users\29838\Documents\Solomon\SimpleVLA\hf_cache\Openvla-oft-SFT-libero10-trajall"


class OpenVLAStage(Stage[InferenceContext]):
    """Wraps OpenVLA-OFT predict_action as a pipeline stage.

    Loads in 4-bit NF4 (the only quant validated on the 5070 Ti) via
    BitsAndBytesConfig. Patches norm_stats from dataset_statistics.json so
    the libero_10_no_noops unnorm_key is always available. Lazy-loads on the
    first process() call to keep pipeline construction cheap.
    """

    name = "OpenVLA-OFT"
    required_inputs = ["pil_image", "prompt"]
    provided_outputs = ["action"]

    def __init__(
        self,
        model_path: str | None = None,
        unnorm_key: str = "libero_10_no_noops",
    ) -> None:
        self.model_path: str = (
            model_path
            or os.environ.get("SNAPVLA_OPENVLA_PATH", _DEFAULT_MODEL_PATH)
        )
        self.unnorm_key = unnorm_key
        self._processor: Any = None
        self._model: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

        path = self.model_path
        logger.info("Loading OpenVLA-OFT from %s (4-bit NF4)...", path)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self._processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        self._model = AutoModelForVision2Seq.from_pretrained(
            path,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self._model.eval()
        torch.cuda.synchronize()

        t_load = time.time() - t0
        peak_gib = torch.cuda.max_memory_allocated() / 1024**3
        logger.info("OpenVLA-OFT loaded in %.1fs, peak VRAM %.2f GiB", t_load, peak_gib)

        # The checkpoint's trust_remote_code doesn't auto-populate norm_stats
        # from dataset_statistics.json; we do it here so unnorm_key works.
        self._patch_norm_stats(path)

    def _patch_norm_stats(self, model_dir: str) -> None:
        stats_path = Path(model_dir) / "dataset_statistics.json"
        if not stats_path.exists():
            logger.warning("dataset_statistics.json not found at %s; unnorm may fail", stats_path)
            return
        with open(stats_path, encoding="utf-8") as fh:
            extra = json.load(fh)
        if not hasattr(self._model, "norm_stats") or self._model.norm_stats is None:
            logger.warning("model.norm_stats is None; skipping patch")
            return
        new_keys = [k for k in extra if k not in self._model.norm_stats]
        self._model.norm_stats.update(extra)
        logger.info("Patched %d unnorm_key(s) into norm_stats: %s", len(new_keys), new_keys)

    def process(self, ctx: InferenceContext) -> InferenceContext:
        self._ensure_loaded()

        formatted = f"In: What action should the robot take to {ctx.prompt.lower()}?\nOut:"
        inputs = self._processor(formatted, ctx.pil_image).to(
            self._model.device, dtype=torch.float16
        )

        with torch.inference_mode():
            result = self._model.predict_action(
                **inputs, unnorm_key=self.unnorm_key, do_sample=False
            )

        # predict_action returns (actions_numpy_[1,7], hidden_states)
        raw = result[0] if isinstance(result, (tuple, list)) else result
        if isinstance(raw, torch.Tensor):
            flat: list[float] = raw.detach().float().cpu().flatten().tolist()
        else:
            # numpy array from the model's own unnorm step
            import numpy as np
            flat = np.asarray(raw, dtype=float).flatten().tolist()

        ctx.action = flat
        # VLA produces an action tensor, not a text caption
        ctx.text = ""
        return ctx
