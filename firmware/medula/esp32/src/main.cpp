// The medula on the ESP32: every 20 ms read the servos and the IMU, take the phone's packets, run the medula (the
// same C++ as in the simulator), write the servo goals; telemetry over BLE at 10 Hz. A serial console for the bench.
//
// Safety on the bench: the torque stays off at power-on. It goes on with the first valid COMMAND from the phone or
// the console's 't', and the medula starts from wherever the legs are (no jump to `rest`).
#include <Arduino.h>
#include <NimBLEDevice.h>
#include <Wire.h>
#include <math.h>

#include "board.h"
#include "imu.h"
#include "imu6.h"
#include "medula.h"
#include "sts.h"

using cogu::Medula;
using cogu::Sensors;

namespace {

constexpr float STEP_RAD = 2.0f * 3.14159265f / 4096.0f;   // one STS3215 step
constexpr uint32_t PERIOD_US = 20000;                       // 50 Hz
const char* SERVICE_UUID = "c0901000-6f9b-4b3a-9c3e-436f67750001";   // "Cogu"
const char* COMMAND_UUID = "c0901001-6f9b-4b3a-9c3e-436f67750001";
const char* TELEMETRY_UUID = "c0901002-6f9b-4b3a-9c3e-436f67750001";

sts::Bus bus;
imu6::Mpu6050 imu;
cogu::GravityFilter gravity_filter;
Medula medula;
NimBLECharacteristic* telemetry_chr = nullptr;
bool torque_on = false, imu_ok = false;
uint32_t bus_errors = 0, overruns = 0, loops = 0;
uint32_t t_sense = 0, t_act = 0, t_write = 0, t_max = 0;   // µs, for the console's 's'
float volts = 8.0f;

// a few packets from the BLE task to the control loop
struct Packet {
  uint8_t data[20];
  uint8_t n;
};
constexpr int QUEUE = 8;
Packet queue[QUEUE];
volatile int q_head = 0, q_tail = 0;
portMUX_TYPE q_mux = portMUX_INITIALIZER_UNLOCKED;
volatile bool command_seen = false;

class CommandWrites : public NimBLECharacteristicCallbacks {
  void onWrite(NimBLECharacteristic* c) override {
    const std::string v = c->getValue();
    if (v.size() > sizeof(Packet::data)) return;
    portENTER_CRITICAL(&q_mux);
    const int next = (q_head + 1) % QUEUE;
    if (next != q_tail) {                         // full: drop (the next one carries the whole state again)
      memcpy(queue[q_head].data, v.data(), v.size());
      queue[q_head].n = static_cast<uint8_t>(v.size());
      q_head = next;
    }
    portEXIT_CRITICAL(&q_mux);
  }
};

bool pop(Packet* p) {
  bool got = false;
  portENTER_CRITICAL(&q_mux);
  if (q_tail != q_head) {
    *p = queue[q_tail];
    q_tail = (q_tail + 1) % QUEUE;
    got = true;
  }
  portEXIT_CRITICAL(&q_mux);
  return got;
}

void start_ble() {
  NimBLEDevice::init("Cogu");
  NimBLEDevice::setMTU(23);
  NimBLEServer* server = NimBLEDevice::createServer();
  NimBLEService* svc = server->createService(SERVICE_UUID);
  NimBLECharacteristic* cmd = svc->createCharacteristic(COMMAND_UUID, NIMBLE_PROPERTY::WRITE | NIMBLE_PROPERTY::WRITE_NR);
  cmd->setCallbacks(new CommandWrites());
  telemetry_chr = svc->createCharacteristic(TELEMETRY_UUID, NIMBLE_PROPERTY::READ | NIMBLE_PROPERTY::NOTIFY);
  svc->start();
  NimBLEAdvertising* adv = NimBLEDevice::getAdvertising();
  adv->addServiceUUID(SERVICE_UUID);
  adv->start();
}

// the servos and the IMU -> the medula's Sensors (pelvis frame, MJCF joint order and units)
bool read_body(Sensors* s, int round_robin) {
  bool ok = true;
  for (int j = 0; j < cogu::NU; ++j) {
    uint8_t raw[4], err = 0;
    if (!bus.read(board::SERVO_ID[j], sts::PRESENT_POSITION, raw, 4, &err)) {
      ok = false;
      continue;
    }
    const int16_t pos = sts::from_sm(static_cast<uint16_t>(raw[0] | raw[1] << 8));
    const int16_t vel = sts::from_sm(static_cast<uint16_t>(raw[2] | raw[3] << 8));
    s->q[j] = board::SERVO_DIR[j] * (pos - board::SERVO_ZERO[j]) * STEP_RAD;
    s->qd[j] = board::SERVO_DIR[j] * vel * STEP_RAD;
    if (err & 0x2F) ok = false;                   // voltage, angle, overheat, overload... (status bits)
  }
  uint8_t v;                                      // the battery, from one servo per step
  if (bus.read(board::SERVO_ID[round_robin % cogu::NU], sts::PRESENT_VOLTAGE, &v, 1)) volts = v * 0.1f;
  s->volts = volts;
  float g[3] = {0, 0, 0}, a[3] = {0, 0, 9.81f};
  if (imu_ok) imu_ok = imu.read(g, a);
  for (int i = 0; i < 3; ++i) {                   // to the pelvis frame
    s->gyro[i] = 0;
    float ai = 0;
    for (int k = 0; k < 3; ++k) {
      s->gyro[i] += board::IMU_TO_PELVIS[i][k] * g[k];
      ai += board::IMU_TO_PELVIS[i][k] * a[k];
    }
    s->lin_vel[i] = ai;                           // (scratch: the accelerometer in the pelvis frame)
  }
  gravity_filter.update(s->gyro, s->lin_vel, cogu::DT);
  gravity_filter.gravity(s->gravity);
  s->truth = false;
  s->height = 0;
  for (float& x : s->lin_vel) x = 0;
  return ok && imu_ok;
}

void write_goals(const float* target) {
  int16_t pos[cogu::NU];
  for (int j = 0; j < cogu::NU; ++j) {
    const float steps = board::SERVO_ZERO[j] + board::SERVO_DIR[j] * target[j] / STEP_RAD;
    pos[j] = static_cast<int16_t>(constrain(lroundf(steps), 0, 4095));
  }
  bus.sync_goal(board::SERVO_ID, pos, cogu::NU);
}

void console(const Sensors& s) {
  if (!Serial.available()) return;
  const char c = Serial.read();
  if (c == 't') {
    torque_on = !torque_on;
    bus.torque(board::SERVO_ID, cogu::NU, torque_on);
    if (torque_on) medula.begin(s, false);
    Serial.printf("torque %s\n", torque_on ? "on" : "off");
  } else if (c == 'p') {                          // raw positions, for the calibration in board.h
    for (int j = 0; j < cogu::NU; ++j) {
      uint8_t raw[2];
      const bool ok = bus.read(board::SERVO_ID[j], sts::PRESENT_POSITION, raw, 2);
      Serial.printf("id %2u: %s%d\n", board::SERVO_ID[j], ok ? "" : "no answer ", ok ? raw[0] | raw[1] << 8 : 0);
    }
  } else if (c == 's') {
    const auto& st = medula.stats();
    Serial.printf("loops %u overruns %u | sense %u us, medula %u us, write %u us, worst loop %u us | bus errors %u | "
                  "%.2f V | state %u | cmds %u rejected %u stale %u watchdog %u\n",
                  loops, overruns, t_sense, t_act, t_write, t_max, bus_errors, volts, medula.state(), st.commands,
                  st.rejected, st.stale, st.watchdog);
    t_max = 0;
  }
}

}  // namespace

void setup() {
  Serial.begin(115200);
  bus.begin(Serial1, board::SERVO_BAUD, board::SERVO_RX, board::SERVO_TX);
  Wire.begin(board::I2C_SDA, board::I2C_SCL, 400000);
  imu_ok = imu.begin();
  int found = 0;
  for (int j = 0; j < cogu::NU; ++j) found += bus.ping(board::SERVO_ID[j]);
  Serial.printf("Cogu medula: %d of %d servos, IMU %s. Torque off: 't' (or the phone) turns it on; 'p' positions, "
                "'s' timings.\n", found, cogu::NU, imu_ok ? "ok" : "MISSING");
  start_ble();
}

void loop() {
  static uint32_t next = micros();
  static Sensors s{};
  static float target[cogu::NU];
  static int tick = 0;
  const uint32_t t0 = micros();
  const bool ok = read_body(&s, tick);
  bus_errors += !ok;
  const uint32_t t1 = micros();
  if (torque_on) medula.sense(s);
  medula.set_servo_fault(!ok);
  Packet p;
  while (pop(&p)) {
    if (!torque_on) {                              // the first valid command turns the torque on
      cogu::proto::Command c;
      if (!cogu::proto::decode(p.data, p.n, &c)) continue;
      torque_on = true;
      bus.torque(board::SERVO_ID, cogu::NU, true);
      medula.begin(s, false);
    }
    medula.receive(p.data, p.n, medula.time());
  }
  const uint32_t t2 = micros();
  if (torque_on) medula.act(target);
  const uint32_t t3 = micros();
  if (torque_on) write_goals(target);
  const uint32_t t4 = micros();
  if (tick % 5 == 0 && telemetry_chr != nullptr) {  // 10 Hz
    uint8_t out[cogu::proto::TEL_SIZE];
    const int n = medula.telemetry(out);
    telemetry_chr->setValue(out, n);
    telemetry_chr->notify();
  }
  console(s);
  t_sense = t1 - t0;
  t_act = t3 - t2;
  t_write = t4 - t3;
  const uint32_t took = micros() - t0;
  if (took > t_max) t_max = took;
  ++loops;
  ++tick;
  next += PERIOD_US;
  const int32_t wait = static_cast<int32_t>(next - micros());
  if (wait > 0) delayMicroseconds(wait);
  else {
    ++overruns;
    next = micros();
  }
}
