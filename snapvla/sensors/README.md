# snapvla.sensors

Contract-driven sensor adapters for the SnapVLA pipeline.

Every sensor in this module wraps a hardware-specific SDK behind a single,
uniform interface so that the rest of the pipeline never needs to know which
physical camera is attached.

---

## Frame contract

Every sensor adapter must return a plain Python `dict` from `capture_frame()`
with exactly these keys:

| Key | Type | Shape | Dtype | Description |
|---|---|---|---|---|
| `rgb` | `np.ndarray` | `(H, W, 3)` | `uint8` | Colour image, channels in **RGB** order |
| `depth` | `np.ndarray` | `(H, W)` | `uint16` | Depth map in **millimetres** (raw SDK values) |
| `timestamp` | `float` | scalar | — | `time.perf_counter()` value recorded at capture |

The `rgb` and `depth` arrays are spatially aligned: pixel `(r, c)` in `rgb`
corresponds to the same 3-D point as pixel `(r, c)` in `depth`.

---

## Usage — RealSenseSource

### Minimal (camera picks its default profile)

```python
from snapvla.sensors import RealSenseSource

with RealSenseSource(serial_number="838212070000") as camera:
    frame = camera.capture_frame()
    print(frame["rgb"].shape)    # e.g. (720, 1280, 3)
    print(frame["depth"].shape)  # e.g. (720, 1280)
    print(frame["timestamp"])    # e.g. 12345.678
```

### Explicit resolution and FPS

```python
from snapvla.sensors import RealSenseSource

with RealSenseSource(
    serial_number="838212070000",
    fps=30,
    width=1280,
    height=720,
) as camera:
    frame = camera.capture_frame()
```

### Without context manager

```python
from snapvla.sensors import RealSenseSource

camera = RealSenseSource(serial_number="838212070000", fps=30, width=640, height=480)
camera.connect()

try:
    for _ in range(100):
        frame = camera.capture_frame()
        process(frame)
finally:
    camera.disconnect()
```

### Find connected cameras

Use LeRobot's bundled CLI to list available cameras and their serial numbers:

```bash
lerobot-find-cameras realsense
```

---

## Adding a new sensor type

1. Create a new file in this directory, e.g. `snapvla/sensors/zed.py`.
2. Subclass `BaseSensorSource` and implement the three abstract methods:

```python
from typing import Dict
import numpy as np
from snapvla.sensors.base import BaseSensorSource

class ZedSource(BaseSensorSource):
    def __init__(self, camera_id: int = 0) -> None:
        super().__init__(name=f"zed-{camera_id}")
        self._camera_id = camera_id
        self._sdk_handle = None

    def connect(self) -> bool:
        # Initialise the ZED SDK here.
        # Raise ConnectionError with a clear message on failure.
        ...
        return True

    def capture_frame(self) -> Dict[str, np.ndarray]:
        # Must return {"rgb": ..., "depth": ..., "timestamp": ...}
        ...

    def disconnect(self) -> None:
        # Release SDK resources. Must be idempotent.
        ...
```

3. Export the new class from `snapvla/sensors/__init__.py`:

```python
from snapvla.sensors.zed import ZedSource

__all__ = [..., "ZedSource"]
```

Rules to follow:
- The frame contract (`rgb`, `depth`, `timestamp`) is mandatory and non-negotiable.
- `connect()` must raise `ConnectionError` on failure — never return `False` silently.
- `disconnect()` must be idempotent (safe to call multiple times).
- Wrap SDK-specific errors with a short, actionable message before re-raising.

---

## Dependencies

| Package | Purpose | How to install |
|---|---|---|
| `lerobot[intelrealsense]` | LeRobot core + RealSense extras | `pip install lerobot[intelrealsense]` |
| `pyrealsense2` | Intel RealSense SDK Python bindings (pulled in by the above) | included in `lerobot[intelrealsense]` |
| `opencv-python` | Image colour conversion and rotation | included in `lerobot` |
| `numpy` | Array handling | included in `lerobot` |

### Environment setup (recommended)

```bash
conda create -n snapvla python=3.11
conda activate snapvla
pip install lerobot[intelrealsense]
```

Or with `uv`:

```bash
uv venv .venv --python 3.11
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install lerobot[intelrealsense]
```
