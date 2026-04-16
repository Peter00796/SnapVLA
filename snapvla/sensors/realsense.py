"""
RealSense camera adapter for SnapVLA.

Wraps LeRobot's ``RealSenseCamera`` / ``RealSenseCameraConfig`` pair behind the
``BaseSensorSource`` contract so the rest of the pipeline only ever sees the
standard ``{'rgb', 'depth', 'timestamp'}`` dict.

LeRobot references
------------------
- ``lerobot.cameras.realsense.RealSenseCamera``
- ``lerobot.cameras.realsense.RealSenseCameraConfig``
- ``lerobot.cameras.configs.ColorMode``

Dependencies (must be installed in the active environment)::

    pip install lerobot[intelrealsense]
    # which pulls in: pyrealsense2, opencv-python, numpy
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional

import numpy as np

from snapvla.sensors.base import BaseSensorSource

logger = logging.getLogger(__name__)


class RealSenseSource(BaseSensorSource):
    """SnapVLA adapter for Intel RealSense RGBD cameras.

    Uses LeRobot's ``RealSenseCamera`` under the hood.  The class hides
    LeRobot's verbose internal errors and surfaces short, actionable messages
    instead — which is especially useful when a camera's USB connection is
    loose or a serial number is mis-typed.

    Args:
        serial_number: The 12-digit serial number printed on the camera body
            (e.g. ``"838212070000"``).  Also accepts a unique human-readable
            device name (e.g. ``"Intel RealSense D435"``), but a serial number
            is preferred for stability when multiple cameras are attached.
        fps: Desired capture frame-rate.  Must be set together with *width*
            and *height*, or left as ``None`` to let the camera choose its
            default profile.
        width: Desired frame width in pixels.
        height: Desired frame height in pixels.
        warmup_s: Seconds to wait for the sensor to stabilise after pipeline
            start.  LeRobot enforces a minimum of 1 s regardless.
        name: Optional human-readable label used in log messages.  Defaults
            to ``"realsense-<serial_number>"``.

    Example::

        from snapvla.sensors.realsense import RealSenseSource

        with RealSenseSource(serial_number="838212070000", fps=30) as camera:
            frame = camera.capture_frame()
            print(f"RGB shape  : {frame['rgb'].shape}")    # (H, W, 3) uint8
            print(f"Depth shape: {frame['depth'].shape}")  # (H, W)    uint16
            print(f"Timestamp  : {frame['timestamp']}")    # float
    """

    def __init__(
        self,
        serial_number: str,
        fps: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        warmup_s: int = 1,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or f"realsense-{serial_number}")
        self.serial_number = serial_number
        self._fps = fps
        self._width = width
        self._height = height
        self._warmup_s = warmup_s

        # Populated in connect(); kept as None until then so that is_connected
        # correctly returns False.
        self._camera: object = None  # RealSenseCamera instance

    # ------------------------------------------------------------------
    # is_connected — delegate to LeRobot's own property once connected
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:  # type: ignore[override]
        """True after ``connect()`` succeeds and before ``disconnect()``."""
        if self._camera is None:
            return False
        # RealSenseCamera.is_connected checks whether the rs2 pipeline is live.
        return bool(self._camera.is_connected)

    @is_connected.setter
    def is_connected(self, _value: bool) -> None:
        # BaseSensorSource.__init__ assigns self.is_connected = False directly;
        # we accept that write silently and rely on the property getter above.
        pass

    # ------------------------------------------------------------------
    # BaseSensorSource contract
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Open the RealSense pipeline and warm up the sensor.

        Wraps LeRobot's initialisation sequence in a ``try/except`` so that
        the long internal tracebacks are replaced with a concise message that
        names the camera and suggests a remediation step.

        Returns:
            True on success.

        Raises:
            ConnectionError: If the camera cannot be opened (wrong serial
                number, USB bandwidth issue, driver not installed, etc.).
            RuntimeError: If the LeRobot or pyrealsense2 packages are not
                installed in the current environment.
        """
        # Lazy import — keeps the module importable even when lerobot or
        # pyrealsense2 are absent, so unit tests can mock them freely.
        try:
            from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
            from lerobot.cameras.configs import ColorMode
        except ImportError as exc:
            raise RuntimeError(
                "LeRobot is not installed. "
                "Run: pip install lerobot[intelrealsense]"
            ) from exc

        # Build the config.  RealSenseCameraConfig requires that fps/width/height
        # are either ALL set or ALL None.
        if all(v is not None for v in (self._fps, self._width, self._height)):
            config = RealSenseCameraConfig(
                serial_number_or_name=self.serial_number,
                fps=self._fps,
                width=self._width,
                height=self._height,
                color_mode=ColorMode.RGB,
                use_depth=True,
                warmup_s=self._warmup_s,
            )
        else:
            config = RealSenseCameraConfig(
                serial_number_or_name=self.serial_number,
                color_mode=ColorMode.RGB,
                use_depth=True,
                warmup_s=self._warmup_s,
            )

        camera = RealSenseCamera(config)

        try:
            camera.connect()
        except ConnectionError as exc:
            # LeRobot already raises ConnectionError — re-raise with a shorter
            # message that includes the serial number we tried to open.
            raise ConnectionError(
                f"[SnapVLA] Could not connect to RealSense camera "
                f"(serial={self.serial_number!r}). "
                f"Check that the camera is plugged in and the serial number is correct. "
                f"Original error: {exc}"
            ) from exc
        except Exception as exc:
            # Catch-all for unexpected LeRobot / pyrealsense2 internals.
            raise ConnectionError(
                f"[SnapVLA] Unexpected error while connecting to RealSense camera "
                f"(serial={self.serial_number!r}): {type(exc).__name__}: {exc}"
            ) from exc

        self._camera = camera
        logger.info(
            "[SnapVLA] %s connected (fps=%s, w=%s, h=%s).",
            self.name,
            camera.fps,
            camera.width,
            camera.height,
        )
        return True

    def capture_frame(self) -> Dict[str, np.ndarray]:
        """Capture one aligned RGBD frame.

        Calls LeRobot's ``read()`` for the colour image and ``read_depth()``
        for the depth map, then packages both into the standard SnapVLA
        contract dict.

        The ``timestamp`` value is recorded with ``time.perf_counter()``
        immediately after the colour read returns, giving a consistent,
        high-resolution clock across all sensor types.

        Returns:
            A dict with keys:

            * ``'rgb'``   — ``np.ndarray`` shape ``(H, W, 3)`` dtype ``uint8``,
              channels in RGB order.
            * ``'depth'`` — ``np.ndarray`` shape ``(H, W)`` dtype ``uint16``,
              raw depth values in millimetres as delivered by the RealSense SDK.
            * ``'timestamp'`` — ``float`` from ``time.perf_counter()``.

        Raises:
            RuntimeError: If the camera is not connected or a frame cannot be
                read within the SDK timeout.
        """
        if not self.is_connected or self._camera is None:
            raise RuntimeError(
                f"[SnapVLA] {self.name} is not connected. Call connect() first."
            )

        # read() returns the latest unconsumed colour frame (blocks until one
        # arrives, up to 10 s internally inside LeRobot's async_read).
        rgb: np.ndarray = self._camera.read()

        # Record the timestamp immediately after the colour frame is returned
        # so that rgb and depth share the same logical capture time.
        timestamp: float = time.perf_counter()

        # read_depth() does the same but for the depth stream.
        depth: np.ndarray = self._camera.read_depth()

        return {
            "rgb": rgb,
            "depth": depth,
            "timestamp": timestamp,
        }

    def disconnect(self) -> None:
        """Stop the RealSense pipeline and release all resources.

        Safe to call even if the camera was never connected or was already
        disconnected.
        """
        if self._camera is None:
            return

        try:
            if self._camera.is_connected:
                self._camera.disconnect()
        except Exception as exc:
            # Log but do not re-raise — disconnect must be idempotent.
            logger.warning(
                "[SnapVLA] %s: error during disconnect (ignored): %s",
                self.name,
                exc,
            )
        finally:
            self._camera = None
            logger.info("[SnapVLA] %s disconnected.", self.name)
