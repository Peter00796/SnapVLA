"""Wire protocol for Pi <-> PC inference traffic.

Messages are msgpack-encoded dicts sent as WebSocket binary frames.

Client -> Server (InferenceRequest)
    frame_id   : int     monotonic counter assigned by the edge
    ts_edge    : float   time.time() at the edge when capture finished
    jpeg       : bytes   MJPG-compressed RGB frame
    prompt     : str     optional text prompt for the VLM/VLA

Server -> Client (InferenceResponse)
    frame_id   : int     echoes the request
    ts_server  : float   time.time() on the server after inference
    latency_ms : float   server-side inference wall time
    text       : str     model output (caption / answer / action tokens)
    action     : list[float]|None  7-DoF action from VLA path; None for VLM-only
    traces     : list    list of StageTrace.to_dict() entries
    error      : str|None
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import msgpack


@dataclass
class InferenceRequest:
    frame_id: int
    ts_edge: float
    jpeg: bytes
    prompt: str = ""

    def to_msgpack(self) -> bytes:
        return msgpack.packb(
            {
                "frame_id": self.frame_id,
                "ts_edge": self.ts_edge,
                "jpeg": self.jpeg,
                "prompt": self.prompt,
            },
            use_bin_type=True,
        )


@dataclass
class InferenceResponse:
    frame_id: int
    ts_server: float
    latency_ms: float
    text: str
    action: list[float] | None = None
    traces: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def to_msgpack(self) -> bytes:
        return msgpack.packb(
            {
                "frame_id": self.frame_id,
                "ts_server": self.ts_server,
                "latency_ms": self.latency_ms,
                "text": self.text,
                "action": self.action,
                "traces": self.traces,
                "error": self.error,
            },
            use_bin_type=True,
        )


def encode_request(req: InferenceRequest) -> bytes:
    return req.to_msgpack()


def encode_response(resp: InferenceResponse) -> bytes:
    return resp.to_msgpack()


def decode_request(buf: bytes) -> InferenceRequest:
    d = msgpack.unpackb(buf, raw=False)
    return InferenceRequest(
        frame_id=d["frame_id"],
        ts_edge=d["ts_edge"],
        jpeg=d["jpeg"],
        prompt=d.get("prompt", ""),
    )


def decode_response(buf: bytes) -> InferenceResponse:
    d = msgpack.unpackb(buf, raw=False)
    return InferenceResponse(
        frame_id=d["frame_id"],
        ts_server=d["ts_server"],
        latency_ms=d["latency_ms"],
        text=d["text"],
        action=d.get("action"),
        traces=d.get("traces", []),
        error=d.get("error"),
    )
