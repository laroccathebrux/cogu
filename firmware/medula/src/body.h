// What the medula reads from the body each control step, and the few thresholds it needs only on the robot.
#pragma once

#include "cogu_params.h"

namespace cogu {

struct Sensors {
  float gyro[3];        // rad/s, IMU frame
  float gravity[3];     // unit vector, IMU frame (points down: (0, 0, -1) level); from the IMU filter on the robot
  float q[NU];          // rad, servo positions, joint order of the MJCF
  float qd[NU];         // rad/s
  float volts;          // battery, V
  // The simulator also knows the pelvis height and speed; the robot does not. With `truth` set the medula uses them
  // exactly as the Python one (sim2sim equivalence); without it, only what the robot senses (see the robot-only
  // thresholds below).
  bool truth;
  float height;         // m
  float lin_vel[3];     // m/s, world
};

// The robot cannot measure the pelvis height or speed: "still" uses the IMU's rotation instead of the speed, and
// "standing / fallen" uses the tilt alone. Bench to confirm.
constexpr float STILL_GYRO = 0.3f;    // rad/s: |gyro| below this counts as still (instead of |pelvis speed| < 0.1)

}  // namespace cogu
