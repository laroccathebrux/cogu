#include "medula.h"

#include <math.h>

namespace cogu {

namespace {
constexpr double NEVER = -1e300;
float clampf(float x, float lo, float hi) { return x < lo ? lo : x > hi ? hi : x; }
}  // namespace

void Medula::begin(const Sensors& s, bool from_rest) {
  s_ = s;
  if (from_rest) walker_.reset(s);
  else walker_.handover(s);
  mode_ = WALKING;
  acted_ = false;
}

// -- the phone's packets --------------------------------------------------------------------------------------------

void Medula::receive(const uint8_t* data, int n, double t) {
  proto::Command c;
  if (!proto::decode(data, n, &c)) {
    ++stats_.rejected;
    return;
  }
  if (have_seq_ && !watchdog_ && !proto::newer(c.seq, seq_)) {
    ++stats_.stale;                   // late or repeated (after a watchdog trip any seq starts afresh)
    return;
  }
  ++stats_.commands;
  if (last_cmd_ > NEVER) stats_.longest_gap = fmaxf(stats_.longest_gap, static_cast<float>(t - last_cmd_));
  seq_ = c.seq;
  have_seq_ = true;
  last_cmd_ = t;
  const float v[3] = {c.vx, c.vy, c.wz};
  for (int i = 0; i < 3; ++i) target_[i] = clampf(v[i], LIMIT_LO[i], LIMIT_HI[i]);
  if (c.stop) {
    for (int i = 0; i < 3; ++i) target_[i] = cmd_[i] = 0.0f;
  }
  if (!have_posture_ || c.posture_id != posture_id_) {
    posture_id_ = c.posture_id;
    have_posture_ = true;
    posture(c.posture);
  }
}

void Medula::posture(uint8_t kind) {
  // from walking, sit down (lying down starts from the seat); from sitting, stand up or lie down; from lying, get up
  if (cutoff_ && kind == proto::STAND) return;
  lie_when_seated_ = kind == proto::LIE;
  if (mode_ == WALKING && (kind == proto::SIT || kind == proto::LIE)) {
    for (float& x : cmd_) x = 0.0f;
    pose_.start(s_, proto::SIT, 0.0);
    mode_ = POSE;
  } else if (mode_ == POSE) {
    if (kind == proto::STAND && pose_.lying) {
      getup_.start(s_);
      mode_ = GETTING_UP;
    } else if (!pose_.lying) {
      pose_.mode = kind;
    }
  }
}

// -- the 50 Hz loop -------------------------------------------------------------------------------------------------

void Medula::act(float target[NU]) {
  const bool wd = t_ - last_cmd_ > WATCHDOG;   // (also before the first command: hold still)
  if (wd && !watchdog_ && last_cmd_ > NEVER) ++stats_.watchdog;
  watchdog_ = wd;
  if (s_.volts < CUTOFF_VOLTS && !cutoff_) {
    cutoff_ = true;
    if (mode_ == WALKING) posture(proto::SIT);
  }
  for (int i = 0; i < 3; ++i) {
    const float want = (wd || cutoff_) ? 0.0f : target_[i];
    const float lim = RAMP[i] * DT;
    cmd_[i] += clampf(want - cmd_[i], -lim, lim);
    if (fabsf(cmd_[i]) < 1e-3f) cmd_[i] = 0.0f;
  }
  if (mode_ == WALKING) {
    for (int i = 0; i < 3; ++i) walker_.command[i] = cmd_[i];
    walker_.act(target);
  } else if (mode_ == POSE) {
    if (lie_when_seated_ && pose_.state() == proto::SITTING) {
      pose_.mode = proto::LIE;
      lie_when_seated_ = false;
    }
    pose_.act(s_, target);
  } else {
    getup_.act(s_, target);
  }
  acted_ = true;
}

void Medula::sense(const Sensors& s) {
  s_ = s;
  if (!acted_) return;
  acted_ = false;
  if (mode_ == WALKING) {
    walker_.sense(s);
    if (walker_.fallen(s)) fell();
  } else if (mode_ == POSE) {
    pose_.sense(s);
    if (pose_.ready()) walk();                // back in `rest`: the walker takes over
    else if (pose_.fallen(s)) fell();
  } else {
    getup_.sense(s);
    if (getup_.ready()) walk();
  }
  t_ += DT_S;
}

void Medula::fell() {
  ++falls_;
  getup_.start(s_);
  mode_ = GETTING_UP;
}

void Medula::walk() {
  walker_.handover(s_);
  for (float& x : cmd_) x = 0.0f;
  mode_ = WALKING;
}

// -- what it tells the phone ----------------------------------------------------------------------------------------

uint8_t Medula::state() const {
  if (mode_ == GETTING_UP) return proto::GETTING_UP;
  if (mode_ == POSE) return pose_.state();
  return proto::WALKING;
}

int Medula::telemetry(uint8_t out[proto::TEL_SIZE]) {
  const float* g = s_.gravity;               // roll and pitch of the pelvis, from the IMU's gravity
  proto::Telemetry tel;
  tel.seq = tel_seq_++;
  tel.ack = have_seq_ ? seq_ : 0;
  tel.state = state();
  tel.posture_id = have_posture_ ? posture_id_ : 0;
  tel.flags = (watchdog_ ? proto::WATCHDOG : 0) | (s_.volts < LOW_VOLTS ? proto::LOW_BATTERY : 0) |
              (cutoff_ ? proto::CUTOFF : 0) | (servo_fault_ ? proto::SERVO_FAULT : 0);
  tel.battery = s_.volts;
  tel.roll = atan2f(-g[1], -g[2]) * (180.0f / 3.14159265f);
  tel.pitch = asinf(clampf(g[0], -1.0f, 1.0f)) * (180.0f / 3.14159265f);
  tel.vx = cmd_[0];
  tel.wz = cmd_[2];
  tel.falls = falls_;
  return proto::encode(tel, out);
}

}  // namespace cogu
