"""Server-side smoke test: build the pipeline and run a single inference.

Acceptance: ctx.text is non-empty AND MoondreamStage trace has latency_ms>0.
Run: uv run --project server python server/smoke_infer.py
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
from snapvla.server.stages import DecodeJpegStage, LogStage, MoondreamStage


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="SnapVLA server smoke test.")
    parser.add_argument("--image", type=Path, default=Path("captures/smoke_001.jpg"))
    parser.add_argument("--prompt", type=str, default="Describe this image in one short sentence.")
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
    pipe.add_stage(DecodeJpegStage()).add_stage(MoondreamStage()).add_stage(LogStage()).build()

    try:
        ctx, traces = pipe.run_once({"frame_id": 1, "prompt": args.prompt, "jpeg": jpeg_bytes})
    except Exception as exc:
        print(f"[smoke] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("\n[smoke] Stage traces:")
    for t in traces:
        print(f"  {t.stage_name:12s}  {t.latency_ms:10.2f} ms")

    print(f"\n[smoke] ctx.text ({len(ctx.text)} chars):\n{ctx.text}\n")

    md = next((t for t in traces if t.stage_name == "Moondream"), None)
    if not ctx.text or md is None or md.latency_ms <= 0:
        print("[smoke] ACCEPTANCE FAILED", file=sys.stderr)
        return 2
    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
