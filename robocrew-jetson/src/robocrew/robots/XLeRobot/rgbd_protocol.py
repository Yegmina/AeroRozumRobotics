"""Wire protocol shared by the Orbbec RGB-D service and RoboCrew client."""

from __future__ import annotations

import json
import socket
import struct
from typing import BinaryIO


HEADER = struct.Struct("!I")


def send_packet(stream: BinaryIO | socket.socket, metadata: dict, payload: bytes = b"") -> None:
    header = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    stream.sendall(HEADER.pack(len(header)) + header + HEADER.pack(len(payload)) + payload)


def _read_exact(stream: BinaryIO | socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.recv(remaining)
        if not chunk:
            raise ConnectionError("RGB-D service closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def receive_packet(stream: BinaryIO | socket.socket) -> tuple[dict, bytes]:
    header_size = HEADER.unpack(_read_exact(stream, HEADER.size))[0]
    metadata = json.loads(_read_exact(stream, header_size).decode("utf-8"))
    payload_size = HEADER.unpack(_read_exact(stream, HEADER.size))[0]
    return metadata, _read_exact(stream, payload_size)
