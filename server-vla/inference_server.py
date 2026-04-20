"""WebSocket inference server — OpenVLA-OFT path.

Accepts InferenceRequest messages, runs DecodeJPEG -> OpenVLAStage -> Log,
returns an InferenceResponse with the 7-DoF action tensor.

Usage (on the Windows PC):
    uv run --project server-vla python server-vla/inference_server.py
    uv run --project server-vla python server-vla/inference_server.py --host 0.0.0.0 --port 8766
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
from pathlib import Path

import websockets

from snapvla.common.wire import (
    InferenceResponse,
    decode_request,
    encode_response,
)
from snapvla.pipeline import VLAPipeline
from snapvla.server.contexts import InferenceContext
from snapvla.server.stages import DecodeJpegStage, LogStage
from snapvla.server.stages_vla import OpenVLAStage

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = r"C:\Users\29838\Documents\Solomon\SimpleVLA\hf_cache\Openvla-oft-SFT-libero10-trajall"


def _resolve_model_path() -> str:
    return os.environ.get("SNAPVLA_OPENVLA_PATH", _DEFAULT_MODEL_PATH)


def build_pipeline(model: str = "openvla") -> VLAPipeline[InferenceContext]:
    # Only "openvla" is valid in this venv; the arg exists for CLI symmetry
    # with a future multi-model server.
    if model != "openvla":
        raise ValueError(f"This server only supports model='openvla'; got {model!r}")
    pipeline = (
        VLAPipeline(InferenceContext)
        .add_stage(DecodeJpegStage())
        .add_stage(OpenVLAStage(model_path=_resolve_model_path()))
        .add_stage(LogStage())
    )
    return pipeline.build()


def _placeholder_jpeg() -> bytes:
    import io
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(buf, format="JPEG", quality=80)
    return buf.getvalue()


async def serve(host: str, port: int, model: str) -> None:
    # Verify patches before the expensive model load so a bad model dir fails fast.
    from ensure_model_patches import ensure
    rc = ensure(_resolve_model_path())
    if rc != 0:
        raise RuntimeError(f"ensure_model_patches returned {rc}; refusing to start")

    pipeline = build_pipeline(model)
    logger.info("warming up OpenVLA-OFT (this takes ~30s)...")
    warm_jpeg = _placeholder_jpeg()
    pipeline.run_once(
        {"frame_id": -1, "prompt": "pick up the object", "jpeg": warm_jpeg}
    )
    logger.info("model warm. accepting connections on %s:%d", host, port)

    async def handler(ws):
        peer = ws.remote_address
        logger.info("client connected: %s", peer)
        try:
            async for msg in ws:
                req = decode_request(msg)
                t0 = time.perf_counter()
                action = None
                text = ""
                try:
                    _ctx, traces = pipeline.run_once(
                        {
                            "frame_id": req.frame_id,
                            "prompt": req.prompt,
                            "jpeg": req.jpeg,
                        }
                    )
                    action = _ctx.action
                    text = _ctx.text
                    error = None
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    traces = []
                    logger.exception("pipeline failed for frame %d", req.frame_id)
                latency = (time.perf_counter() - t0) * 1000.0
                resp = InferenceResponse(
                    frame_id=req.frame_id,
                    ts_server=time.time(),
                    latency_ms=latency,
                    text=text,
                    action=action,
                    traces=[t.to_dict() for t in traces],
                    error=error,
                )
                await ws.send(encode_response(resp))
        except websockets.ConnectionClosed:
            logger.info("client disconnected: %s", peer)

    async with websockets.serve(handler, host, port, max_size=16 * 1024 * 1024):
        await asyncio.Future()  # run forever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--model", default="openvla", choices=["openvla"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        asyncio.run(serve(args.host, args.port, args.model))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
