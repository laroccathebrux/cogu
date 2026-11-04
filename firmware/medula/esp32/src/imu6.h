// The 6-axis IMU on I2C. Which one goes in the robot is not decided (ADR-002: "uma IMU de 6 eixos"); this is the
// MPU-6050 (also the MPU-6500 / 9250 register map), the most common. Another chip only needs its own read().
#pragma once

#include <Arduino.h>

namespace imu6 {

class Mpu6050 {
 public:
  bool begin(uint8_t addr = 0x68);
  // gyro rad/s and accelerometer m/s² in the chip's axes; false if the chip did not answer
  bool read(float gyro[3], float accel[3]);
  void set_gyro_bias(const float b[3]) { for (int i = 0; i < 3; ++i) bias_[i] = b[i]; }

 private:
  uint8_t addr_ = 0x68;
  float bias_[3] = {0, 0, 0};
};

}  // namespace imu6
