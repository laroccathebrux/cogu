#include "imu.h"

#include <math.h>

namespace cogu {

namespace {
constexpr float G = 9.81f;

void normalise(float* v) {
  const float n = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  if (n > 1e-6f) for (int i = 0; i < 3; ++i) v[i] /= n;
}
}  // namespace

void GravityFilter::reset(const float accel[3]) {
  for (int i = 0; i < 3; ++i) up_[i] = accel[i];
  normalise(up_);
  started_ = true;
}

void GravityFilter::update(const float w[3], const float accel[3], float dt) {
  if (!started_) {
    reset(accel);
    return;
  }
  // a vector fixed in the world, seen from a body turning at w: du/dt = u x w (rotated exactly by |w| dt)
  const float wn = sqrtf(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
  if (wn > 1e-9f) {
    const float k[3] = {-w[0] / wn, -w[1] / wn, -w[2] / wn};   // turn u by -w dt (Rodrigues)
    const float a = wn * dt, c = cosf(a), s = sinf(a);
    const float kxu[3] = {k[1] * up_[2] - k[2] * up_[1], k[2] * up_[0] - k[0] * up_[2], k[0] * up_[1] - k[1] * up_[0]};
    const float kdu = k[0] * up_[0] + k[1] * up_[1] + k[2] * up_[2];
    for (int i = 0; i < 3; ++i) up_[i] = up_[i] * c + kxu[i] * s + k[i] * kdu * (1 - c);
  }
  // pull toward the accelerometer, trusting it less the further it is from 1 g
  const float an = sqrtf(accel[0] * accel[0] + accel[1] * accel[1] + accel[2] * accel[2]);
  const float off = fabsf(an - G) / G;
  const float trust = off < 0.1f ? 1.0f : off > 0.5f ? 0.0f : (0.5f - off) / 0.4f;
  if (an > 1e-3f && trust > 0) {
    const float k = gain_ * dt * trust;
    for (int i = 0; i < 3; ++i) up_[i] += k * (accel[i] / an - up_[i]);
  }
  normalise(up_);
}

}  // namespace cogu
