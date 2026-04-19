"""One-shot USB camera smoke test for the SnapVLA edge runtime.

Opens USBCameraSource, captures a single frame, validates the payload,
writes the JPEG to disk, and prints a one-line OK summary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from snapvla.edge.usb_camera import USBCameraSource


def main() -> None:
    parser = argparse.ArgumentParser(description="SnapVLA USB camera smoke test.")
    parser.add_argument("--device", type=int, default=0, help="V4L2 device index (default: 0).")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("captures/smoke_001.jpg"),
        help="Where to save the captured JPEG.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with USBCameraSource(device=args.device) as cam:
        frame = cam.capture_frame()

    assert set(frame) >= {"rgb", "jpeg", "timestamp"}, f"missing keys: {set(frame)}"
    rgb = frame["rgb"]
    jpeg = frame["jpeg"]
    assert isinstance(rgb, np.ndarray), f"rgb is not ndarray: {type(rgb)}"
    assert rgb.shape == (480, 640, 3), f"rgb shape: {rgb.shape}"
    assert rgb.dtype == np.uint8, f"bad rgb dtype: {rgb.dtype}"
    assert jpeg is not None, "MJPG encode returned None — cv2.imencode failed"
    assert isinstance(jpeg, (bytes, bytearray)), f"jpeg not bytes: {type(jpeg)}"
    assert len(jpeg) > 1000, f"jpeg too small: {len(jpeg)} bytes"

    args.output.write_bytes(jpeg)
    print(f"OK: rgb={tuple(rgb.shape)} jpeg={len(jpeg)} saved={args.output}")


if __name__ == "__main__":
    main()
