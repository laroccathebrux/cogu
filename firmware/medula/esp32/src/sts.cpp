#include "sts.h"

namespace sts {

namespace {
constexpr uint32_t TIMEOUT_US = 2000;
}

void Bus::begin(HardwareSerial& port, uint32_t baud, int rx, int tx) {
  port_ = &port;
  port_->begin(baud, SERIAL_8N1, rx, tx);
  port_->setTimeout(2);
}

void Bus::send(uint8_t id, uint8_t instr, const uint8_t* params, uint8_t n) {
  uint8_t buf[64];
  const uint8_t len = n + 2;
  buf[0] = 0xFF;
  buf[1] = 0xFF;
  buf[2] = id;
  buf[3] = len;
  buf[4] = instr;
  uint8_t sum = id + len + instr;
  for (uint8_t i = 0; i < n; ++i) {
    buf[5 + i] = params[i];
    sum += params[i];
  }
  buf[5 + n] = ~sum;
  while (port_->available()) port_->read();   // drop anything left over (the echo of a broadcast, noise)
  port_->write(buf, 6 + n);
  port_->flush();
}

bool Bus::reply(uint8_t id, uint8_t* params, uint8_t n, uint8_t* error) {
  // FF FF id len error params... checksum
  uint8_t head[5], tail[1];
  const uint32_t t0 = micros();
  int got = 0;
  while (got < 5 && micros() - t0 < TIMEOUT_US) {
    if (port_->available()) {
      head[got] = port_->read();
      if (got < 2 && head[got] != 0xFF) continue;    // resync on the FF FF header
      ++got;
    }
  }
  if (got < 5 || head[2] != id || head[3] != n + 2) return false;
  uint8_t sum = head[2] + head[3] + head[4];
  for (uint8_t i = 0; i < n; ++i) {
    while (!port_->available()) if (micros() - t0 > TIMEOUT_US) return false;
    params[i] = port_->read();
    sum += params[i];
  }
  while (!port_->available()) if (micros() - t0 > TIMEOUT_US) return false;
  tail[0] = port_->read();
  if (error) *error = head[4];
  return static_cast<uint8_t>(~sum) == tail[0];
}

bool Bus::ping(uint8_t id) {
  send(id, PING, nullptr, 0);
  return reply(id, nullptr, 0, nullptr);
}

bool Bus::read(uint8_t id, uint8_t addr, uint8_t* out, uint8_t n, uint8_t* error) {
  const uint8_t p[2] = {addr, n};
  send(id, READ, p, 2);
  return reply(id, out, n, error);
}

bool Bus::write(uint8_t id, uint8_t addr, const uint8_t* data, uint8_t n) {
  uint8_t p[32];
  p[0] = addr;
  for (uint8_t i = 0; i < n; ++i) p[1 + i] = data[i];
  send(id, WRITE, p, n + 1);
  return reply(id, nullptr, 0, nullptr);
}

void Bus::sync_goal(const uint8_t* ids, const int16_t* pos, int count, uint16_t speed, uint8_t acc) {
  // ACC (41), GOAL_POSITION (42-43), GOAL_TIME (44-45, unused), GOAL_SPEED (46-47): 7 bytes per servo
  uint8_t p[2 + 8 * 12];
  p[0] = ACC;
  p[1] = 7;
  int k = 2;
  for (int i = 0; i < count; ++i) {
    const uint16_t v = to_sm(pos[i]);
    p[k++] = ids[i];
    p[k++] = acc;
    p[k++] = v & 0xFF;
    p[k++] = v >> 8;
    p[k++] = 0;
    p[k++] = 0;
    p[k++] = speed & 0xFF;
    p[k++] = speed >> 8;
  }
  send(0xFE, SYNC_WRITE, p, static_cast<uint8_t>(k));   // broadcast: no reply
}

void Bus::torque(const uint8_t* ids, int count, bool on) {
  const uint8_t v = on ? 1 : 0;
  for (int i = 0; i < count; ++i) write(ids[i], TORQUE_ENABLE, &v, 1);
}

}  // namespace sts
