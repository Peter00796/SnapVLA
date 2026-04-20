"""Pi-side streaming client.

Grabs frames from a USB camera and ships each one to the inference server
over WebSocket, printing the text response so you can eyeball correctness.

Usage (from the Pi):
    # VLM (Moondream)
    uv run python -m edge.stream_client \
        --server ws://192.168.88.12:8765/ \
        --prompt "Describe what you see in one short sentence." \
        --fps 1 --frames 10

    # VLA (OpenVLA-OFT)
    uv run python -m edge.stream_client \
        --server ws://192.168.88.12:8765/ \
        --prompt "pick up the object" \
        --fps 1 --frames 3
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time

import websockets

from snapvla.common.wire import (
    InferenceRequest,
    decode_response,
    encode_request,
)
from snapvla.edge.usb_camera import USBCameraSource

logger = logging.getLogger(__name__)


async def run(server_url: str, prompt: str, fps: float, device: int | str, max_frames: int = 0) -> None:
    interval = 1.0 / fps if fps > 0 else 0.0
    frame_id = 0

    with USBCameraSource(device=device) as cam:
        async with websockets.connect(server_url, max_size=16 * 1024 * 1024) as ws:
            logger.info("connected to %s", server_url)
            while True:
                loop_t0 = time.perf_counter()
                frame = cam.capture_frame()
                jpeg = frame["jpeg"]
                if jpeg is None:
                    logger.warning("camera returned no jpeg; skipping")
                    continue
                req = InferenceRequest(
                    frame_id=frame_id,
                    ts_edge=time.time(),
                    jpeg=jpeg,
                    prompt=prompt,
                )
                await ws.send(encode_request(req))
                reply = await ws.recv()
                resp = decode_response(reply)
                rtt_ms = (time.perf_counter() - loop_t0) * 1000.0
                print(
                    f"[#{resp.frame_id:04d}] infer={resp.latency_ms:.1f}ms "
                    f"rtt={rtt_ms:.1f}ms  text={resp.text!r}"
                )
                if resp.action is not None:
                    if len(resp.action) == 7:
                        formatted = "[" + ", ".join(f"{v:.3f}" for v in resp.action) + "]"
                    else:
                        formatted = str(resp.action)
                    print(f"     action={formatted}")
                if resp.traces:
                    for t in resp.traces:
                        print(f"    trace {t['stage']:>20s}  +{t['ms']:.1f}ms")
                frame_id += 1
                if max_frames > 0 and frame_id >= max_frames:
                    break
                sleep_s = interval - (time.perf_counter() - loop_t0)
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="ws://192.168.88.12:8765/infer")
    parser.add_argument(
        "--prompt",
        required=True,
        help="Natural-language prompt. For Moondream use a question; for OpenVLA use a task instruction like 'pick up the object'.",
    )
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--device", default=0)
    parser.add_argument("--frames", type=int, default=0, help="Stop after N frames (0 = unlimited).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        asyncio.run(run(args.server, args.prompt, args.fps, args.device, args.frames))
    except KeyboardInterrupt:
        print("bye")


if __name__ == "__main__":
    main()
