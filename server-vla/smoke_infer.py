"""VLA-side smoke test: build the OpenVLA pipeline and run a single inference.

Acceptance: ctx.action is a list of 7 floats AND OpenVLA-OFT trace has latency_ms>0.
Exit 0 on success, 1 on pipeline error, 2 on acceptance failure.

Run:
    uv run --project server-vla python server-vla/smoke_infer.py
    uv run --project server-vla python server-vla/smoke_infer.py --image captures/smoke_001.jpg
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path

from PIL import Image

from snapvla.pipeline import VLAPipeline
from snapvla.server.contexts import InferenceContext
from snapvla.server.stages import DecodeJpegStage, LogStage
from snapvla.server.stages_vla import OpenVLAStage


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="SnapVLA VLA smoke test.")
    parser.add_argument("--image", type=Path, default=Path("captures/smoke_001.jpg"))
    parser.add_argument("--prompt", type=str, default="pick up the object on the table")
    args = parser.parse_args()

    image_explicit = "--image" in sys.argv

    if args.image.exists():
        jpeg_bytes = args.image.read_bytes()
        src = str(args.image)
    elif image_explicit:
        print(f"[smoke] ERROR: --image path does not exist: {args.image}", file=sys.stderr)
        return 1
    else:
        img = Image.new("RGB", (224, 224), color=(64, 128, 192))
        for x in range(224):
            for y in range(224):
                if (x // 16 + y // 16) % 2 == 0:
                    img.putpixel((x, y), (240, 220, 80))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        jpeg_bytes = buf.getvalue()
        src = "<synthesized 224x224 checkerboard>"

    print(f"[smoke] image source: {src} ({len(jpeg_bytes)} bytes)")
    print(f"[smoke] prompt: {args.prompt!r}")

    pipe: VLAPipeline[InferenceContext] = VLAPipeline(InferenceContext)
    pipe.add_stage(DecodeJpegStage()).add_stage(OpenVLAStage()).add_stage(LogStage()).build()

    try:
        ctx, traces = pipe.run_once({"frame_id": 1, "prompt": args.prompt, "jpeg": jpeg_bytes})
    except Exception as exc:
        print(f"[smoke] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("\n[smoke] Stage traces:")
    for t in traces:
        print(f"  {t.stage_name:20s}  {t.latency_ms:10.2f} ms")

    if ctx.action is not None:
        formatted = "[" + ", ".join(f"{v:.3f}" for v in ctx.action) + "]"
        print(f"\n[smoke] ctx.action ({len(ctx.action)} values):\n  {formatted}\n")
    else:
        print("\n[smoke] ctx.action: None\n")

    vla_trace = next((t for t in traces if t.stage_name == "OpenVLA-OFT"), None)
    if (
        ctx.action is None
        or not isinstance(ctx.action, list)
        or len(ctx.action) != 7
        or not all(isinstance(v, float) for v in ctx.action)
        or vla_trace is None
        or vla_trace.latency_ms <= 0
    ):
        print("[smoke] ACCEPTANCE FAILED", file=sys.stderr)
        return 2

    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
