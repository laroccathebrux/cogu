#include "imu6.h"

#include <Wire.h>

namespace imu6 {

namespace {
constexpr uint8_t PWR_MGMT_1 = 0x6B, CONFIG = 0x1A, GYRO_CONFIG = 0x1B, ACCEL_CONFIG = 0x1C, ACCEL_XOUT_H = 0x3B,
                  WHO_AM_I = 0x75;
constexpr float GYRO_SCALE = (1000.0f / 32768.0f) * (3.14159265f / 180.0f);   // ±1000 °/s
constexpr float ACCEL_SCALE = (4.0f * 9.81f) / 32768.0f;                       // ±4 g

bool put(uint8_t addr, uint8_t reg, uint8_t v) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(v);
  return Wire.endTransmission() == 0;
}
}  // namespace

bool Mpu6050::begin(uint8_t addr) {
  addr_ = addr;
  if (!put(addr_, PWR_MGMT_1, 0x01)) return false;   // wake up, clock from the gyro's PLL
  delay(50);
  put(addr_, CONFIG, 0x03);                          // digital low-pass ~44 Hz (the loop is 50 Hz)
  put(addr_, GYRO_CONFIG, 0x10);                     // ±1000 °/s
  put(addr_, ACCEL_CONFIG, 0x08);                    // ±4 g
  Wire.beginTransmission(addr_);
  Wire.write(WHO_AM_I);
  if (Wire.endTransmission(false) != 0 || Wire.requestFrom(addr_, static_cast<uint8_t>(1)) != 1) return false;
  Wire.read();
  return true;
}

bool Mpu6050::read(float gyro[3], float accel[3]) {
  Wire.beginTransmission(addr_);
  Wire.write(ACCEL_XOUT_H);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(addr_, static_cast<uint8_t>(14)) != 14) return false;
  int16_t raw[7];
  for (int i = 0; i < 7; ++i) raw[i] = static_cast<int16_t>((Wire.read() << 8) | Wire.read());
  for (int i = 0; i < 3; ++i) {
    accel[i] = raw[i] * ACCEL_SCALE;                  // raw[3] is the temperature
    gyro[i] = raw[4 + i] * GYRO_SCALE - bias_[i];
  }
  return true;
}

}  // namespace imu6
