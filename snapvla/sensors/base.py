"""
Base class for all SnapVLA sensor sources.

Defines the strict input/output contract that every camera or depth sensor
adapter must satisfy. Any new sensor type (Azure Kinect, ZED, Orbbec, etc.)
must inherit from BaseSensorSource and implement all three abstract methods.
"""

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class BaseSensorSource(ABC):
    """
    SnapVLA sensor base class.

    Enforces a strict input/output contract for all 3D camera sources.
    All implementations must return frames as a dict with the keys
    'rgb', 'depth', and 'timestamp' from ``capture_frame()``.

    The class also provides context-manager support so sensors can be used
    inside ``with`` blocks, guaranteeing that ``disconnect()`` is always called
    even if an exception occurs mid-session.

    Args:
        name: A human-readable identifier for this sensor instance (used in
              log messages and error strings).

    Attributes:
        name (str): Human-readable sensor identifier.
        is_connected (bool): Whether the sensor is currently live.
            Subclasses that manage their own connection state (like
            ``RealSenseSource``) should override this as a property.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.is_connected: bool = False

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the sensor and complete any required warm-up.

        Implementations should block until the sensor is ready to deliver
        frames and then return ``True``.  On unrecoverable failure the method
        must raise ``ConnectionError`` with a clear, human-readable message —
        do *not* silently return ``False``.

        Returns:
            True if the connection succeeded.

        Raises:
            ConnectionError: If the sensor cannot be opened or warmed up.
        """

    @abstractmethod
    def capture_frame(self) -> Dict[str, np.ndarray]:
        """Capture one aligned frame from the sensor.

        This is the core contract method.  Every implementation must return a
        dict that satisfies the following schema exactly:

        .. code-block:: python

            {
                "rgb":       np.ndarray,   # shape (H, W, 3), dtype uint8, channels RGB
                "depth":     np.ndarray,   # shape (H, W),    dtype uint16 (mm) or float32 (m)
                "timestamp": float,        # time.perf_counter() value at capture time
            }

        The ``rgb`` and ``depth`` arrays must be spatially aligned to the same
        pixel grid (i.e. pixel (r, c) in ``rgb`` corresponds to the same 3-D
        point as pixel (r, c) in ``depth``).

        Returns:
            A dict with mandatory keys ``'rgb'``, ``'depth'``, and
            ``'timestamp'``.

        Raises:
            RuntimeError: If a frame cannot be delivered (camera not
                connected, hardware timeout, etc.).
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Safely release all camera resources.

        Must be idempotent — calling ``disconnect()`` on an already-
        disconnected sensor must not raise.
        """

    # ------------------------------------------------------------------
    # Context-manager support (inherited by all subclasses for free)
    # ------------------------------------------------------------------

    def __enter__(self) -> "BaseSensorSource":
        """Connect on entry and return self for use inside ``with`` blocks."""
        self.connect()
        return self

    def __exit__(self, _exc_type: object, _exc_val: object, _exc_tb: object) -> None:
        """Always disconnect on exit, even if an exception was raised."""
        self.disconnect()

    def __repr__(self) -> str:
        status = "connected" if self.is_connected else "disconnected"
        return f"{self.__class__.__name__}(name={self.name!r}, status={status})"
