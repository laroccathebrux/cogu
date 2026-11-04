// Gravity in the IMU frame from a 6-axis IMU (gyro + accelerometer): the policies need it, the IMU does not give it.
// A complementary filter on the "up" direction: turn it with the gyro, pull it a little toward the accelerometer
// (which reads +g up at rest), less when the accelerometer is far from 1 g (a step, a bump). Yaw is not needed.
#pragma once

namespace cogu {

class GravityFilter {
 public:
  explicit GravityFilter(float gain = 2.0f) : gain_(gain) {}   // 1/s: how fast it trusts the accelerometer
  void reset(const float accel[3]);
  void update(const float gyro[3], const float accel[3], float dt);
  void gravity(float out[3]) const { for (int i = 0; i < 3; ++i) out[i] = -up_[i]; }

 private:
  float gain_;
  float up_[3] = {0, 0, 1};
  bool started_ = false;
};

}  // namespace cogu
