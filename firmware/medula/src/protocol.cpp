#include "protocol.h"

#include <math.h>

namespace cogu {
namespace proto {

uint16_t crc16(const uint8_t* data, int n) {
  uint16_t crc = 0xFFFF;
  for (int i = 0; i < n; ++i) {
    crc ^= static_cast<uint16_t>(data[i]) << 8;
    for (int k = 0; k < 8; ++k) crc = (crc & 0x8000) ? static_cast<uint16_t>((crc << 1) ^ 0x1021) : crc << 1;
  }
  return crc;
}

bool newer(uint8_t seq, uint8_t last) {
  const uint8_t d = static_cast<uint8_t>(seq - last);
  return d > 0 && d < 128;
}

namespace {

int16_t i16(double x) {             // round half to even, as Python's round()
  const double r = nearbyint(x);
  return static_cast<int16_t>(r < -32768 ? -32768 : r > 32767 ? 32767 : r);
}

void put16(uint8_t* p, uint16_t v) {
  p[0] = static_cast<uint8_t>(v & 0xFF);
  p[1] = static_cast<uint8_t>(v >> 8);
}

uint16_t get16(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }

int seal(uint8_t* out, int body) {
  put16(out + body, crc16(out, body));
  return body + 2;
}

bool open(const uint8_t* data, int n, int size, uint8_t kind) {
  if (n != size) return false;
  if (crc16(data, size - 2) != get16(data + size - 2)) return false;
  return data[0] >> 4 == VERSION && (data[0] & 0x0F) == kind;
}

}  // namespace

int encode(const Command& c, uint8_t out[CMD_SIZE]) {
  out[0] = VERSION << 4 | COMMAND;
  out[1] = c.seq;
  put16(out + 2, static_cast<uint16_t>(i16(c.vx * 1000.0)));
  put16(out + 4, static_cast<uint16_t>(i16(c.vy * 1000.0)));
  put16(out + 6, static_cast<uint16_t>(i16(c.wz * 1000.0)));
  out[8] = c.posture;
  out[9] = c.posture_id;
  out[10] = c.stop ? STOP : 0;
  out[11] = 0;
  return seal(out, 12);
}

bool decode(const uint8_t* d, int n, Command* c) {
  if (!open(d, n, CMD_SIZE, COMMAND) || d[8] > LIE) return false;
  c->seq = d[1];
  c->vx = static_cast<int16_t>(get16(d + 2)) / 1000.0f;
  c->vy = static_cast<int16_t>(get16(d + 4)) / 1000.0f;
  c->wz = static_cast<int16_t>(get16(d + 6)) / 1000.0f;
  c->posture = d[8];
  c->posture_id = d[9];
  c->stop = d[10] & STOP;
  return true;
}

int encode(const Telemetry& t, uint8_t out[TEL_SIZE]) {
  out[0] = VERSION << 4 | TELEMETRY;
  out[1] = t.seq;
  out[2] = t.ack;
  out[3] = t.state;
  out[4] = t.posture_id;
  out[5] = t.flags;
  const double mv = nearbyint(t.battery * 1000.0);
  put16(out + 6, static_cast<uint16_t>(mv < 0 ? 0 : mv > 65535 ? 65535 : mv));
  put16(out + 8, static_cast<uint16_t>(i16(t.roll * 100.0)));
  put16(out + 10, static_cast<uint16_t>(i16(t.pitch * 100.0)));
  put16(out + 12, static_cast<uint16_t>(i16(t.vx * 1000.0)));
  put16(out + 14, static_cast<uint16_t>(i16(t.wz * 1000.0)));
  out[16] = t.falls;
  out[17] = 0;
  return seal(out, 18);
}

bool decode(const uint8_t* d, int n, Telemetry* t) {
  if (!open(d, n, TEL_SIZE, TELEMETRY) || d[3] > GETTING_UP) return false;
  t->seq = d[1];
  t->ack = d[2];
  t->state = d[3];
  t->posture_id = d[4];
  t->flags = d[5];
  t->battery = get16(d + 6) / 1000.0f;
  t->roll = static_cast<int16_t>(get16(d + 8)) / 100.0f;
  t->pitch = static_cast<int16_t>(get16(d + 10)) / 100.0f;
  t->vx = static_cast<int16_t>(get16(d + 12)) / 1000.0f;
  t->wz = static_cast<int16_t>(get16(d + 14)) / 1000.0f;
  t->falls = d[16];
  return true;
}

}  // namespace proto
}  // namespace cogu
