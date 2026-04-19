"""WebSocket inference server.

Accepts InferenceRequest messages, runs the VLM pipeline, returns an
InferenceResponse including StageTrace entries so the client can see the
full per-stage data flow.

Usage (on the Windows PC):
    uv run python -m server.inference_server --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time

import websockets

from snapvla.common.wire import (
    InferenceResponse,
    decode_request,
    encode_response,
)
from snapvla.pipeline import VLAPipeline
from snapvla.server.contexts import InferenceContext
from snapvla.server.stages import DecodeJpegStage, LogStage, MoondreamStage

logger = logging.getLogger(__name__)


def build_pipeline() -> VLAPipeline[InferenceContext]:
    pipeline = (
        VLAPipeline(InferenceContext)
        .add_stage(DecodeJpegStage())
        .add_stage(MoondreamStage())
        .add_stage(LogStage())
    )
    return pipeline.build()


async def serve(host: str, port: int) -> None:
    pipeline = build_pipeline()
    # Warm up the model once so the first client request isn't a 10s stall.
    logger.info("warming up model...")
    warm_jpeg = _placeholder_jpeg()
    pipeline.run_once({"frame_id": -1, "prompt": "warmup", "jpeg": warm_jpeg})
    logger.info("model warm. accepting connections on %s:%d", host, port)

    async def handler(ws):
        peer = ws.remote_address
        logger.info("client connected: %s", peer)
        try:
            async for msg in ws:
                req = decode_request(msg)
                t0 = time.perf_counter()
                try:
                    _ctx, traces = pipeline.run_once(
                        {
                            "frame_id": req.frame_id,
                            "prompt": req.prompt,
                            "jpeg": req.jpeg,
                        }
                    )
                    text = _ctx.text
                    error = None
                except Exception as exc:
                    text = ""
                    error = f"{type(exc).__name__}: {exc}"
                    traces = []
                    logger.exception("pipeline failed for frame %d", req.frame_id)
                latency = (time.perf_counter() - t0) * 1000.0
                resp = InferenceResponse(
                    frame_id=req.frame_id,
                    ts_server=time.time(),
                    latency_ms=latency,
                    text=text,
                    traces=[t.to_dict() for t in traces],
                    error=error,
                )
                await ws.send(encode_response(resp))
        except websockets.ConnectionClosed:
            logger.info("client disconnected: %s", peer)

    async with websockets.serve(handler, host, port, max_size=16 * 1024 * 1024):
        await asyncio.Future()  # run forever


def _placeholder_jpeg() -> bytes:
    import io
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        asyncio.run(serve(args.host, args.port))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
