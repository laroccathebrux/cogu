// The packets between the phone and the medula, version 1 (ADR-005). The reference is sim/brain/protocol.py; the
// example packets it prints are in test_native.cpp.
#pragma once

#include <stdint.h>

namespace cogu {
namespace proto {

constexpr uint8_t VERSION = 1, COMMAND = 0x1, TELEMETRY = 0x2;
constexpr int CMD_SIZE = 14, TEL_SIZE = 20;

enum Posture : uint8_t { STAND = 0, SIT = 1, LIE = 2 };
constexpr uint8_t STOP = 0x01;                                                    // COMMAND flags
constexpr uint8_t WATCHDOG = 0x01, LOW_BATTERY = 0x02, CUTOFF = 0x04, SERVO_FAULT = 0x08;   // TELEMETRY flags
enum State : uint8_t { WALKING, SITTING_DOWN, SITTING, LYING_DOWN, LYING, STANDING_UP, GETTING_UP };

struct Command {
  uint8_t seq = 0;
  float vx = 0, vy = 0, wz = 0;     // m/s, m/s, rad/s
  uint8_t posture = STAND, posture_id = 0;
  bool stop = false;
};

struct Telemetry {
  uint8_t seq = 0, ack = 0, state = WALKING, posture_id = 0, flags = 0;
  float battery = 8.4f;             // V
  float roll = 0, pitch = 0;        // degrees
  float vx = 0, wz = 0;             // m/s, rad/s, executing
  uint8_t falls = 0;
};

uint16_t crc16(const uint8_t* data, int n);           // CRC-16/CCITT-FALSE
bool newer(uint8_t seq, uint8_t last);                // up to 127 ahead (u8 counters wrap)
int encode(const Command& c, uint8_t out[CMD_SIZE]);
int encode(const Telemetry& t, uint8_t out[TEL_SIZE]);
bool decode(const uint8_t* data, int n, Command* c);  // false: wrong length, crc, header or posture
bool decode(const uint8_t* data, int n, Telemetry* t);

}  // namespace proto
}  // namespace cogu
