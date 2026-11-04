"""The messages between the phone (brain) and the medula (ESP32 on the Waveshare servo board), ADR-005.

Two packets, both at most 20 bytes, so they fit a BLE write or notification with the default MTU (23) and need no
MTU negotiation. The same bytes go over ESP-NOW or UDP later. Every multi-byte field is little-endian (the ESP32
and the phone's ARM both are); the last two bytes are a CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, as
`binascii.crc_hqx(data, 0xFFFF)`) of everything before them.

    COMMAND, phone -> medula, 14 bytes, 10 times a second and at once on a stop or a new posture:
      0     version << 4 | type (0x11)
      1     seq          u8, +1 per packet (wraps)
      2-3   vx           i16 mm/s     forward
      4-5   vy           i16 mm/s     left
      6-7   wz           i16 mrad/s   turn left
      8     posture      u8: 0 stand, 1 sit, 2 lie
      9     posture_id   u8, +1 each time the phone asks for a posture: the medula acts once per new id, so a lost
                         or repeated packet neither drops nor repeats the request
      10    flags        bit 0 STOP: zero the velocity now, without the ramp
      11    reserved     0 (later: body height / lean, ADR-004's expressive posture)
      12-13 crc

    TELEMETRY, medula -> phone, 20 bytes, 10 times a second:
      0     version << 4 | type (0x12)
      1     seq          u8
      2     ack          u8, the last COMMAND seq the medula took
      3     state        u8: STATES below
      4     posture_id   u8, the last posture request it acted on
      5     flags        bit 0 WATCHDOG (no command for WATCHDOG s: holding still), bit 1 LOW_BATTERY,
                         bit 2 CUTOFF (battery too low: sat down, ignores velocity), bit 3 SERVO_FAULT
      6-7   battery      u16 mV
      8-9   roll         i16 centidegrees (the medula's own IMU)
      10-11 pitch        i16 centidegrees
      12-13 vx           i16 mm/s, what the walker is doing now (after the ramp and the watchdog)
      14-15 wz           i16 mrad/s
      16    falls        u8, since power-on (wraps)
      17    reserved     0
      18-19 crc

    uv run python sim/brain/protocol.py        # the checks and the reference packets (hex) for the firmware and the app
"""

from __future__ import annotations

import binascii
import struct
import sys
from dataclasses import dataclass

VERSION = 1
COMMAND, TELEMETRY = 0x1, 0x2

STAND, SIT, LIE = 0, 1, 2
POSTURES = {"stand": STAND, "sit": SIT, "lie": LIE}
STOP = 0x01                                        # COMMAND flags
WATCHDOG, LOW_BATTERY, CUTOFF, SERVO_FAULT = 0x01, 0x02, 0x04, 0x08   # TELEMETRY flags
STATES = ("walking", "sitting down", "sitting", "lying down", "lying", "standing up", "getting up")

_CMD = struct.Struct("<BBhhhBBBB")                 # + crc u16 = 14 bytes
_TEL = struct.Struct("<BBBBBBHhhhhBB")             # + crc u16 = 20 bytes
CMD_SIZE, TEL_SIZE = _CMD.size + 2, _TEL.size + 2


def crc16(data: bytes) -> int:
    return binascii.crc_hqx(data, 0xFFFF)


def _i16(x: float) -> int:
    return max(-32768, min(32767, int(round(x))))


def newer(seq: int, last: int) -> bool:
    """`seq` comes after `last` (u8 counters that wrap): up to 127 ahead counts as newer."""
    return 0 < (seq - last) & 0xFF < 128


def _seal(body: bytes) -> bytes:
    return body + struct.pack("<H", crc16(body))


def _open(data: bytes, size: int, kind: int) -> bytes:
    if len(data) != size:
        raise ValueError(f"length {len(data)}, expected {size}")
    body, (crc,) = data[:-2], struct.unpack("<H", data[-2:])
    if crc16(body) != crc:
        raise ValueError("bad crc")
    if body[0] >> 4 != VERSION or body[0] & 0x0F != kind:
        raise ValueError(f"header 0x{body[0]:02x}, expected 0x{VERSION << 4 | kind:02x}")
    return body


@dataclass
class Command:
    seq: int = 0
    vx: float = 0.0               # m/s
    vy: float = 0.0               # m/s
    wz: float = 0.0               # rad/s
    posture: int = STAND
    posture_id: int = 0
    stop: bool = False

    def encode(self) -> bytes:
        return _seal(_CMD.pack(VERSION << 4 | COMMAND, self.seq & 0xFF, _i16(self.vx * 1000), _i16(self.vy * 1000),
                               _i16(self.wz * 1000), self.posture, self.posture_id & 0xFF, STOP if self.stop else 0,
                               0))

    @classmethod
    def decode(cls, data: bytes) -> "Command":
        _, seq, vx, vy, wz, posture, pid, flags, _ = _CMD.unpack(_open(data, CMD_SIZE, COMMAND))
        if posture > LIE:
            raise ValueError(f"posture {posture}")
        return cls(seq, vx / 1000, vy / 1000, wz / 1000, posture, pid, bool(flags & STOP))


@dataclass
class Telemetry:
    seq: int = 0
    ack: int = 0
    state: int = 0                # index in STATES
    posture_id: int = 0
    flags: int = 0
    battery: float = 8.4          # V
    roll: float = 0.0             # degrees
    pitch: float = 0.0
    vx: float = 0.0               # m/s, executing
    wz: float = 0.0               # rad/s
    falls: int = 0

    @property
    def state_name(self) -> str:
        return STATES[self.state]

    def encode(self) -> bytes:
        return _seal(_TEL.pack(VERSION << 4 | TELEMETRY, self.seq & 0xFF, self.ack & 0xFF, self.state,
                               self.posture_id & 0xFF, self.flags, max(0, min(65535, round(self.battery * 1000))),
                               _i16(self.roll * 100), _i16(self.pitch * 100), _i16(self.vx * 1000),
                               _i16(self.wz * 1000), self.falls & 0xFF, 0))

    @classmethod
    def decode(cls, data: bytes) -> "Telemetry":
        _, seq, ack, state, pid, flags, mv, roll, pitch, vx, wz, falls, _ = _TEL.unpack(
            _open(data, TEL_SIZE, TELEMETRY))
        if state >= len(STATES):
            raise ValueError(f"state {state}")
        return cls(seq, ack, state, pid, flags, mv / 1000, roll / 100, pitch / 100, vx / 1000, wz / 1000, falls)


# the reference packets: the firmware's and the app's tests must produce and accept exactly these bytes
EXAMPLES = [
    ("walk 0.2 m/s, turn left 0.5 rad/s", Command(seq=7, vx=0.2, wz=0.5)),
    ("stop", Command(seq=8, stop=True)),
    ("sit (request 3)", Command(seq=200, posture=SIT, posture_id=3)),
    ("walking, all fine", Telemetry(seq=42, ack=7, battery=7.85, roll=1.5, pitch=-3.25, vx=0.2, wz=0.5)),
    ("lying, watchdog, low battery, 2 falls", Telemetry(seq=255, ack=200, state=4, posture_id=3,
                                                       flags=WATCHDOG | LOW_BATTERY, battery=6.95, falls=2)),
]


def selftest() -> int:
    assert crc16(b"123456789") == 0x29B1                      # the CRC-16/CCITT-FALSE check value
    assert CMD_SIZE == 14 and TEL_SIZE == 20
    for label, msg in EXAMPLES:
        data = msg.encode()
        back = type(msg).decode(data)
        assert back.encode() == data, label
        print(f"  {label:40s} {data.hex(' ')}")
        for i in range(len(data)):                            # any one corrupted byte is refused
            bad = bytearray(data)
            bad[i] ^= 0x5A
            try:
                type(msg).decode(bytes(bad))
            except ValueError:
                continue
            raise AssertionError(f"{label}: byte {i} corrupted and accepted")
    c = Command.decode(Command(seq=1, vx=-0.1234, vy=0.05, wz=-1.0).encode())
    assert (c.vx, c.vy, c.wz) == (-0.123, 0.05, -1.0)
    for bad in (b"", Command().encode()[:-1], Telemetry().encode()):   # wrong length / wrong type
        try:
            Command.decode(bad)
            raise AssertionError("accepted")
        except ValueError:
            pass
    assert newer(0, 255) and newer(5, 250) and not newer(3, 3) and not newer(2, 3) and not newer(130, 2)
    print("protocol: ok")
    return 0


if __name__ == "__main__":
    sys.exit(selftest())
