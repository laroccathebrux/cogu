// The firmware's protocol and policy checks, on the computer (no ESP32): the reference packets of
// sim/brain/protocol.py, byte for byte, every one-byte corruption refused, and the inference time of the walker.
//     make -C firmware/medula test
#include <stdio.h>
#include <string.h>

#include <chrono>

#include "controllers.h"
#include "mlp.h"
#include "protocol.h"

using namespace cogu;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { printf("  FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++failures; } } while (0)

static bool same(const uint8_t* a, const char* hex, int n) {
  for (int i = 0; i < n; ++i) {
    unsigned v;
    sscanf(hex + 3 * i, "%2x", &v);
    if (a[i] != v) return false;
  }
  return true;
}

template <typename Msg, int N>
static void reference(const char* label, const Msg& msg, const char* hex) {
  uint8_t data[N];
  CHECK(proto::encode(msg, data) == N);
  const bool ok = same(data, hex, N);
  CHECK(ok);
  Msg back;
  CHECK(proto::decode(data, N, &back));
  uint8_t again[N];
  proto::encode(back, again);
  CHECK(memcmp(data, again, N) == 0);
  int accepted = 0;
  for (int i = 0; i < N; ++i) {
    uint8_t bad[N];
    memcpy(bad, data, N);
    bad[i] ^= 0x5A;
    accepted += proto::decode(bad, N, &back);
  }
  CHECK(accepted == 0);
  printf("  %-40s %s\n", label, ok ? "same bytes as protocol.py" : "DIFFERENT");
}

int main() {
  CHECK(proto::crc16(reinterpret_cast<const uint8_t*>("123456789"), 9) == 0x29B1);
  proto::Command walk;
  walk.seq = 7; walk.vx = 0.2f; walk.wz = 0.5f;
  reference<proto::Command, proto::CMD_SIZE>("walk 0.2 m/s, turn left 0.5 rad/s", walk,
                                              "11 07 c8 00 00 00 f4 01 00 00 00 00 48 78");
  proto::Command stop;
  stop.seq = 8; stop.stop = true;
  reference<proto::Command, proto::CMD_SIZE>("stop", stop, "11 08 00 00 00 00 00 00 00 00 01 00 63 21");
  proto::Command sit;
  sit.seq = 200; sit.posture = proto::SIT; sit.posture_id = 3;
  reference<proto::Command, proto::CMD_SIZE>("sit (request 3)", sit, "11 c8 00 00 00 00 00 00 01 03 00 00 c9 f0");
  proto::Telemetry fine;
  fine.seq = 42; fine.ack = 7; fine.battery = 7.85f; fine.roll = 1.5f; fine.pitch = -3.25f; fine.vx = 0.2f;
  fine.wz = 0.5f;
  reference<proto::Telemetry, proto::TEL_SIZE>("walking, all fine", fine,
                                               "12 2a 07 00 00 00 aa 1e 96 00 bb fe c8 00 f4 01 00 00 5b c8");
  proto::Telemetry lying;
  lying.seq = 255; lying.ack = 200; lying.state = proto::LYING; lying.posture_id = 3;
  lying.flags = proto::WATCHDOG | proto::LOW_BATTERY; lying.battery = 6.95f; lying.falls = 2;
  reference<proto::Telemetry, proto::TEL_SIZE>("lying, watchdog, low battery, 2 falls", lying,
                                               "12 ff c8 04 03 03 26 1b 00 00 00 00 00 00 00 00 02 00 7b d8");
  proto::Command c, in;
  in.seq = 1; in.vx = -0.1234f; in.vy = 0.05f; in.wz = -1.0f;
  uint8_t buf[proto::TEL_SIZE];
  proto::encode(in, buf);
  CHECK(proto::decode(buf, proto::CMD_SIZE, &c));
  CHECK(c.vx == -0.123f && c.vy == 0.05f && c.wz == -1.0f);
  CHECK(!proto::decode(buf, 0, &c) && !proto::decode(buf, proto::CMD_SIZE - 1, &c));
  proto::encode(proto::Telemetry(), buf);
  CHECK(!proto::decode(buf, proto::TEL_SIZE, &c));                          // a telemetry is not a command
  CHECK(proto::newer(0, 255) && proto::newer(5, 250) && !proto::newer(3, 3) && !proto::newer(2, 3) &&
        !proto::newer(130, 2));

  // the walker's inference time on this computer (the ESP32 is measured on the bench)
  Mlp mlp(walk_policy());
  float obs[64] = {0}, act[16];
  const int n = 2000;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < n; ++i) {
    obs[i % mlp.inputs()] += 1e-3f;
    mlp.eval(obs, act);
  }
  const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / n;
  printf("  walker inference on this computer: %.1f us\n", us);
  printf(failures ? "native: %d FAILED\n" : "native: ok\n", failures);
  return failures ? 1 : 0;
}
