"""Sensor contract for the SnapVLA edge.

All sensor adapters return a uniform frame dict so the rest of the pipeline
never has to care which camera is attached.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseSensorSource(ABC):
    """Context-manager-friendly camera abstraction.

    capture_frame() must return:
        {
            "rgb":       np.ndarray (H, W, 3) uint8,
            "jpeg":      bytes | None,          # optional MJPG payload
            "timestamp": float,                 # time.perf_counter() at capture
        }
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def connect(self) -> bool: ...

    @abstractmethod
    def capture_frame(self) -> dict[str, Any]: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    def __enter__(self) -> BaseSensorSource:
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.disconnect()
