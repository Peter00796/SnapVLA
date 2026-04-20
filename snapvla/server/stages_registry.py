"""Extended stage registry that merges VLA stages when available.

The try/except exists because stages_vla imports bitsandbytes and
transformers 4.54, which are only present in the server-vla venv. Importing
this module in the server (Moondream) venv must not crash — the VLA entry
simply won't appear in MODEL_STAGES, which is the correct behaviour for that
runtime.
"""

from __future__ import annotations

from snapvla.server.stages import MODEL_STAGES

try:
    from snapvla.server.stages_vla import OpenVLAStage

    MODEL_STAGES["openvla"] = OpenVLAStage
except ImportError:
    # bitsandbytes / transformers 4.54 not installed in this venv — VLA path
    # is unavailable but the registry is still usable for Moondream.
    pass

__all__ = ["MODEL_STAGES"]
