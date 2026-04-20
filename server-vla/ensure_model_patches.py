"""Idempotent patch-checker for the OpenVLA-OFT model's modeling_prismatic.py.

Checks whether all three patch sentinels are present in the model directory's
modeling_prismatic.py. If not, copies the vendored (already-patched) file from
server-vla/model_patches/, backing up the original first.

Run standalone before starting the server, or import ensure() from other scripts.

Usage:
    uv run --project server-vla python server-vla/ensure_model_patches.py
    uv run --project server-vla python server-vla/ensure_model_patches.py --model-path /path/to/model
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Sentinel strings from the known-patched file (same set as apply_vla_patches.py).
# All three must be present for the file to be considered fully patched.
_PATCHED_MARKERS: tuple[str, ...] = (
    'hasattr(self, "language_model")',
    "_V(timm.__version__)",
    "isinstance(result, (tuple, list))",
)

_DEFAULT_MODEL_PATH = r"C:\Users\29838\Documents\Solomon\SimpleVLA\hf_cache\Openvla-oft-SFT-libero10-trajall"

# The vendored file lives next to this script regardless of working directory.
_VENDORED = Path(__file__).resolve().parent / "model_patches" / "modeling_prismatic.py"


def ensure(model_path: str | None = None) -> int:
    """Check and apply patches if needed.

    Returns 0 on success (already patched or just patched), non-zero on error.
    """
    resolved = Path(
        model_path
        or os.environ.get("SNAPVLA_OPENVLA_PATH", _DEFAULT_MODEL_PATH)
    )
    target = resolved / "modeling_prismatic.py"

    if not target.exists():
        print(f"ERROR: modeling_prismatic.py not found at {target}", file=sys.stderr)
        return 1

    if not _VENDORED.exists():
        print(f"ERROR: vendored patch not found at {_VENDORED}", file=sys.stderr)
        print("Run: copy the patched file to server-vla/model_patches/modeling_prismatic.py", file=sys.stderr)
        return 1

    content = target.read_text(encoding="utf-8")
    hit = sum(1 for m in _PATCHED_MARKERS if m in content)

    if hit == len(_PATCHED_MARKERS):
        print(f"already patched (all {len(_PATCHED_MARKERS)} markers present): {target}")
        return 0

    if 0 < hit < len(_PATCHED_MARKERS):
        print(
            f"ERROR: partial patch — {hit}/{len(_PATCHED_MARKERS)} markers present. "
            "Re-download the model or copy the vendored file manually.",
            file=sys.stderr,
        )
        return 2

    # No markers present — overwrite with vendored copy after backup.
    backup = target.with_suffix(".py.orig")
    if not backup.exists():
        shutil.copyfile(target, backup)
        print(f"backed up original -> {backup}")

    shutil.copyfile(_VENDORED, target)
    print(f"copied vendored patched file -> {target}")

    # Verify after copy.
    new_content = target.read_text(encoding="utf-8")
    after = sum(1 for m in _PATCHED_MARKERS if m in new_content)
    if after != len(_PATCHED_MARKERS):
        print(
            f"ERROR: copy succeeded but only {after}/{len(_PATCHED_MARKERS)} markers "
            "found. The vendored file may be stale.",
            file=sys.stderr,
        )
        return 3

    print(f"patched OK ({len(_PATCHED_MARKERS)} markers verified)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Ensure OpenVLA-OFT modeling_prismatic.py is patched.")
    ap.add_argument("--model-path", default=None, help="Path to model directory (overrides env var).")
    args = ap.parse_args()
    return ensure(args.model_path)


if __name__ == "__main__":
    raise SystemExit(main())
