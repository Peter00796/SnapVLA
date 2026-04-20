"""Extended stage registry that merges VLA stages when available.

The try/except exists because stages_vla imports bitsandbytes and
transformers 4.54, which are only present in the server-vla venv. Importing
this module in the server (Moondream) venv must not crash — the VLA entry
simply won't appear in MODEL_STAGES, which is the correct behaviour for that
runtime.
"""

from __future__ import annotations

import logging

from snapvla.server.stages import MODEL_STAGES

logger = logging.getLogger(__name__)

# Two-venv design: server (Moondream) venv does not have bitsandbytes or
# transformers 4.54, so importing stages_vla will fail there. That is
# expected — the registry is still usable with only the Moondream stage.
# The server-vla venv has both and will register OpenVLAStage successfully.
try:
    from snapvla.server.stages_vla import OpenVLAStage

    MODEL_STAGES["openvla"] = OpenVLAStage
except ImportError as exc:
    logger.debug(
        "VLA stage not registered (expected in Moondream-only venv): %s", exc
    )

__all__ = ["MODEL_STAGES"]
