"""
snapvla.sensors
===============

Contract-driven sensor adapters for SnapVLA.

Public API
----------
BaseSensorSource
    Abstract base class that every sensor adapter must implement.
    Enforces the ``{'rgb', 'depth', 'timestamp'}`` frame contract.

RealSenseSource
    Concrete adapter for Intel RealSense RGBD cameras.
    Wraps LeRobot's ``RealSenseCamera`` behind the standard contract.
"""

from snapvla.sensors.base import BaseSensorSource
from snapvla.sensors.realsense import RealSenseSource

__all__ = [
    "BaseSensorSource",
    "RealSenseSource",
]
