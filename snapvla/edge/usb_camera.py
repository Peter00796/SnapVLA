"""USB camera adapter (V4L2 / OpenCV) for the Rasprover pan camera.

Prefers the MJPG pixel format when available so the edge can forward JPEG
bytes straight to the server without a re-encode hop.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import cv2
import numpy as np

from snapvla.edge.base import BaseSensorSource

logger = logging.getLogger(__name__)


class USBCameraSource(BaseSensorSource):
    def __init__(
        self,
        device: int | str = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        prefer_mjpg: bool = True,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or f"usb-cam-{device}")
        self._device = device
        self._width = width
        self._height = height
        self._fps = fps
        self._prefer_mjpg = prefer_mjpg
        self._cap: cv2.VideoCapture | None = None

    def connect(self) -> bool:
        cap = cv2.VideoCapture(self._device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise ConnectionError(
                f"[SnapVLA] USB camera {self._device!r} failed to open. "
                "Check that nothing else is holding /dev/video0."
            )
        if self._prefer_mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS, self._fps)
        ok, _ = cap.read()
        if not ok:
            cap.release()
            raise ConnectionError(
                f"[SnapVLA] USB camera {self._device!r} opened but returned no frame."
            )
        self._cap = cap
        logger.info(
            "[SnapVLA] %s connected (req %dx%d @ %d fps).",
            self.name, self._width, self._height, self._fps,
        )
        return True

    def capture_frame(self) -> dict[str, Any]:
        if self._cap is None:
            raise RuntimeError(f"[SnapVLA] {self.name} not connected.")
        ok, frame_bgr = self._cap.read()
        ts = time.perf_counter()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"[SnapVLA] {self.name} failed to read a frame.")
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        ok_jpg, jpg_buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        jpeg_bytes: bytes | None = jpg_buf.tobytes() if ok_jpg else None
        return {"rgb": rgb, "jpeg": jpeg_bytes, "timestamp": ts}

    def disconnect(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None
                logger.info("[SnapVLA] %s disconnected.", self.name)
