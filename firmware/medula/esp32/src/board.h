// The Waveshare Servo Driver with ESP32 and the Cogu's wiring. Everything here is to be confirmed on the bench.
#pragma once

#include <stdint.h>

namespace board {

// the servo bus (STS3215, half duplex through the board's buffer), as Waveshare's examples
constexpr int SERVO_RX = 18, SERVO_TX = 19;
constexpr uint32_t SERVO_BAUD = 1000000;

// I2C: the board's OLED is on these too
constexpr int I2C_SDA = 21, I2C_SCL = 22;

// The servo IDs in the MJCF's joint order (L yaw roll pitch knee ankle, then R), one chain per leg.
constexpr uint8_t SERVO_ID[10] = {1, 2, 3, 4, 5, 11, 12, 13, 14, 15};

// Calibration, per joint: the servo position (0..4095) at the joint's zero in the MJCF, and the direction (+1 / -1).
// The defaults are placeholders: measure each joint on the bench (console command 'p' prints the raw positions).
constexpr int16_t SERVO_ZERO[10] = {2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048};
constexpr int8_t SERVO_DIR[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

// The IMU's axes in the pelvis frame of the MJCF (x forward, y left, z up): pelvis = R * imu. Identity until the
// IMU is mounted.
constexpr float IMU_TO_PELVIS[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

}  // namespace board
