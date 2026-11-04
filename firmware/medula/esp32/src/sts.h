// The Feetech STS servo bus protocol (STS3215): ping, read registers, and one SYNC WRITE for all the goals.
// Packet: FF FF id len instr params... checksum, checksum = ~(id + len + instr + params) & 0xFF.
#pragma once

#include <Arduino.h>

namespace sts {

// STS3215 registers
constexpr uint8_t TORQUE_ENABLE = 40, ACC = 41, GOAL_POSITION = 42, PRESENT_POSITION = 56, PRESENT_SPEED = 58,
                  PRESENT_VOLTAGE = 62, PRESENT_TEMPERATURE = 63;
constexpr uint8_t PING = 0x01, READ = 0x02, WRITE = 0x03, SYNC_WRITE = 0x83;

class Bus {
 public:
  void begin(HardwareSerial& port, uint32_t baud, int rx, int tx);
  bool ping(uint8_t id);
  // reads `n` bytes from `addr`; false on timeout or a bad checksum. `error` gets the servo's status byte.
  bool read(uint8_t id, uint8_t addr, uint8_t* out, uint8_t n, uint8_t* error = nullptr);
  bool write(uint8_t id, uint8_t addr, const uint8_t* data, uint8_t n);
  // position (0..4095) and max speed / acceleration for each id at once (no reply)
  void sync_goal(const uint8_t* ids, const int16_t* pos, int count, uint16_t speed = 0, uint8_t acc = 0);
  void torque(const uint8_t* ids, int count, bool on);

 private:
  void send(uint8_t id, uint8_t instr, const uint8_t* params, uint8_t n);
  bool reply(uint8_t id, uint8_t* params, uint8_t n, uint8_t* error);
  HardwareSerial* port_ = nullptr;
};

// the STS sign convention: bit 15 is the sign
inline int16_t from_sm(uint16_t v) { return (v & 0x8000) ? -static_cast<int16_t>(v & 0x7FFF) : static_cast<int16_t>(v); }
inline uint16_t to_sm(int16_t v) { return v < 0 ? static_cast<uint16_t>(0x8000 | (-v & 0x7FFF)) : static_cast<uint16_t>(v); }

}  // namespace sts
