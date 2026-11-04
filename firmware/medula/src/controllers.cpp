#include "controllers.h"

#include <math.h>
#include <string.h>

#include "cogu_policies.h"
#include "protocol.h"

namespace cogu {

namespace {

constexpr float PI = 3.14159265358979f;

float clampf(float x, float lo, float hi) { return x < lo ? lo : x > hi ? hi : x; }
float deg(float r) { return r * (180.0f / PI); }
float rad(float d) { return d * (PI / 180.0f); }

float norm(const float* x, int n) {
  float s = 0;
  for (int i = 0; i < n; ++i) s += x[i] * x[i];
  return sqrtf(s);
}

// np.interp: piecewise linear, held flat past the ends. `fp` has `stride` columns, column `col`.
float interp(float x, const float* xp, int n, const float* fp, int stride, int col) {
  if (x <= xp[0]) return fp[col];
  if (x >= xp[n - 1]) return fp[(n - 1) * stride + col];
  int i = 1;
  while (xp[i] < x) ++i;
  const float t = (x - xp[i - 1]) / (xp[i] - xp[i - 1]);
  return fp[(i - 1) * stride + col] + t * (fp[i * stride + col] - fp[(i - 1) * stride + col]);
}

void clip_joints(float* t) {
  for (int j = 0; j < NU; ++j) t[j] = clampf(t[j], JOINT_LO[j], JOINT_HI[j]);
}

// both legs at hip pitch / knee / ankle, yaw and roll at zero
void legs(float hip, float knee, float ankle, float* out) {
  for (int side = 0; side < 2; ++side) {
    float* l = out + 5 * side;
    l[0] = 0; l[1] = 0; l[2] = hip; l[3] = knee; l[4] = ankle;
  }
}

float up_z(const Sensors& s) { return -s.gravity[2]; }   // the pelvis z axis, vertical component

bool still(const Sensors& s) {                            // play.py "slow"
  const float qd = norm(s.qd, NU);
  if (s.truth) return norm(s.lin_vel, 3) < 0.1f && qd < 1.5f;
  return norm(s.gyro, 3) < STILL_GYRO && qd < 1.5f;
}

}  // namespace

const MlpSpec& walk_policy() { return WALK_POLICY; }
const MlpSpec& getup_policy() { return GETUP_POLICY; }

float tilt_deg(const Sensors& s) {
  const float c = clampf(-s.gravity[2], -1.0f, 1.0f);
  return deg(acosf(c)) * (s.gravity[0] <= 0.0f ? 1.0f : -1.0f);
}

// -- walk v10 (play.Sim2Sim) ----------------------------------------------------------------------------------------

Walker::Walker() : mlp_(WALK_POLICY) {}

void Walker::restart(const Sensors& s) {
  memset(last_act_, 0, sizeof last_act_);
  memset(action_, 0, sizeof action_);
  phase_[0] = 0.0f;
  phase_[1] = PI;
  recover_time_ = recover_cooldown_ = 0.0f;
  build_obs(s);
}

void Walker::reset(const Sensors& s) {
  memset(filtered_, 0, sizeof filtered_);
  restart(s);
}

void Walker::handover(const Sensors& s) {
  for (int j = 0; j < NU; ++j) filtered_[j] = (s.q[j] - DEFAULT_POSE[j]) / WALK_SCALE;
  restart(s);
}

void Walker::build_obs(const Sensors& s) {   // CoguJoystick._get_obs()['state']
  float* o = obs_;
  for (int i = 0; i < 3; ++i) *o++ = s.gyro[i];
  for (int i = 0; i < 3; ++i) *o++ = s.gravity[i];
  for (int i = 0; i < 3; ++i) *o++ = command[i];
  for (int j = 0; j < NU; ++j) *o++ = s.q[j] - DEFAULT_POSE[j];
  for (int j = 0; j < NU; ++j) *o++ = s.qd[j];
  for (int j = 0; j < NU; ++j) *o++ = last_act_[j];
  *o++ = cosf(phase_[0]); *o++ = cosf(phase_[1]);
  *o++ = sinf(phase_[0]); *o++ = sinf(phase_[1]);
}

void Walker::act(float target[NU]) {
  mlp_.eval(obs_, action_);
  for (int j = 0; j < NU; ++j) {
    filtered_[j] = WALK_ALPHA * action_[j] + (1.0f - WALK_ALPHA) * filtered_[j];
    target[j] = DEFAULT_POSE[j] + filtered_[j] * WALK_SCALE;
  }
  clip_joints(target);
}

void Walker::sense(const Sensors& s) {
  // the observation after the step carries the *previous* action and phase (as in training), then both advance
  build_obs(s);
  memcpy(last_act_, action_, sizeof last_act_);
  for (int i = 0; i < 2; ++i) phase_[i] = fmodf(phase_[i] + WALK_PHASE_DT + PI, 2 * PI) - PI;
  // a push the IMU feels runs the gait clock for a while (v7), one window per push (v10)
  const bool disturbed = WALK_RECOVER && (norm(s.gyro, 2) > WALK_RECOVER_GYRO || up_z(s) < WALK_RECOVER_COS_TILT);
  const double rt = recover_time_, cd = recover_cooldown_;
  bool start = disturbed;
  if (WALK_LATCH) {
    start = disturbed && rt == 0.0 && cd == 0.0;
    recover_cooldown_ = (rt > 0.0 && rt <= DT_S) ? WALK_RECOVER_COOLDOWN : fmax(cd - DT_S, 0.0);
  }
  recover_time_ = start ? WALK_RECOVER_HOLD : fmax(rt - DT_S, 0.0);
  const bool stepping = norm(command, 3) > 0.05f || recover_time_ > 0.0;
  if (WALK_STAND_PHASE_RESET && !stepping) {
    phase_[0] = 0.0f;
    phase_[1] = PI;
  }
}

bool Walker::fallen(const Sensors& s) const {
  return up_z(s) < 0.5f || (s.truth && s.height < 0.10f);
}

// -- get-up v8 (play.GetupPolicy) -----------------------------------------------------------------------------------

Getup::Getup() : mlp_(GETUP_POLICY) {}

void Getup::start(const Sensors& s) {
  memcpy(ref_, s.q, sizeof ref_);
  memset(filtered_, 0, sizeof filtered_);
  memset(last_act_, 0, sizeof last_act_);
  throw_t_ = -1.0;
  side_ = 0;
  lying_t_ = calm_ = 0.0;
  rolled_ = false;
}

void Getup::act(const Sensors& s, float target[NU]) {
  float obs[N_OBS], action[NU];
  float* o = obs;
  for (int i = 0; i < 3; ++i) *o++ = s.gyro[i];
  for (int i = 0; i < 3; ++i) *o++ = s.gravity[i];
  for (int j = 0; j < NU; ++j) *o++ = s.q[j] - DEFAULT_POSE[j];
  for (int j = 0; j < NU; ++j) *o++ = s.qd[j];
  for (int j = 0; j < NU; ++j) *o++ = last_act_[j];
  for (int j = 0; j < NU; ++j) *o++ = ref_[j] - DEFAULT_POSE[j];
  const bool throwing = throw_t_ >= 0.0;
  const double total = fmax(GETUP_ROLL_ENDS[side_ * GETUP_ROLL_K + GETUP_ROLL_K - 1] + GETUP_RELAX, 1e-3);
  *o++ = throwing ? 1.0f : 0.0f;
  *o++ = throwing ? static_cast<float>(throw_t_ / total) : 0.0f;
  mlp_.eval(obs, action);
  for (int j = 0; j < NU; ++j) {
    action[j] = clampf(action[j], -1.0f, 1.0f);
    filtered_[j] = GETUP_ALPHA * action[j] + (1.0f - GETUP_ALPHA) * filtered_[j];
  }
  reference_step(s);
  if (throw_t_ >= 0.0) memset(filtered_, 0, sizeof filtered_);   // v7 on: no correction while relaxing / rolling
  for (int j = 0; j < NU; ++j) target[j] = ref_[j] + filtered_[j] * GETUP_RESIDUAL;
  clip_joints(target);
  memcpy(last_act_, action, sizeof last_act_);
}

void Getup::reference_step(const Sensors& s) {
  // the scripted catch: the legs along the tilt table, a little ahead of the tilt
  const float tilt = tilt_deg(s);
  const float want = (tilt > 0 ? 1.0f : tilt < 0 ? -1.0f : 0.0f) * fmaxf(fabsf(tilt) - GETUP_LEAD, 0.0f);
  float leg[5], catch_[NU];
  for (int j = 0; j < 5; ++j) leg[j] = interp(want, GETUP_TILTS, GETUP_N_TILTS, GETUP_LEGS, 5, j);
  const float step = GETUP_RATE * DT;
  for (int j = 0; j < NU; ++j) catch_[j] = ref_[j] + clampf(leg[j % 5] - ref_[j], -step, step);
  // relax, roll onto the back (front / side / other side), then the catch (cogu_getup._roll_step)
  const float xz = -s.gravity[0], yz = -s.gravity[1];
  const bool on_back = xz > fmaxf(fabsf(yz), 0.5f);
  const int lay = on_back ? 0 : -xz > fmaxf(fabsf(yz), 0.5f) ? 1 : yz > 0.5f ? 2 : yz < -0.5f ? 3 : 0;
  const bool lying = fabsf(tilt) > GETUP_LIE_TILT && throw_t_ < 0.0 && (!on_back || !rolled_);
  lying_t_ = lying ? lying_t_ + DT_S : 0.0;
  if (lying_t_ >= GETUP_LIE_TIME) {
    throw_t_ = 0.0;
    lying_t_ = 0.0;
  }
  const double tt = throw_t_;
  const bool relaxing = tt >= 0.0 && tt < GETUP_RELAX;
  if (relaxing) side_ = lay;
  const double* ends = GETUP_ROLL_ENDS + side_ * GETUP_ROLL_K;
  const double tr = tt - GETUP_RELAX;
  const bool rolling = tr >= 0.0 && tr < ends[GETUP_ROLL_K - 1];
  if (relaxing) {
    for (int j = 0; j < NU; ++j) ref_[j] += clampf(DEFAULT_POSE[j] - ref_[j], -step, step);
  } else if (rolling) {
    int k = 0;
    while (k < GETUP_ROLL_K && ends[k] <= tr) ++k;          // searchsorted(side="right")
    if (k > GETUP_ROLL_K - 1) k = GETUP_ROLL_K - 1;
    memcpy(ref_, GETUP_ROLL_KEYS + (side_ * GETUP_ROLL_K + k) * NU, sizeof ref_);
  } else {
    memcpy(ref_, catch_, sizeof ref_);
  }
  if (tt >= 0.0 && !(relaxing || rolling)) rolled_ = true;
  throw_t_ = (relaxing || rolling) ? tt + DT_S : -1.0;
}

void Getup::sense(const Sensors& s) {
  // ready to walk: upright, near `rest`, still (and, in the simulator, at height) — for 0.5 s in a row
  float rest = 0;
  for (int j = 0; j < NU; ++j) rest += (s.q[j] - DEFAULT_POSE[j]) * (s.q[j] - DEFAULT_POSE[j]);
  const bool high = !s.truth || s.height > 0.18f;
  calm_ = (up_z(s) > 0.95f && high && rest < 0.05f && still(s)) ? calm_ + DT_S : 0.0;
}

// -- sit / lie down (play.PosePolicy, scripted) ---------------------------------------------------------------------

namespace {

float ease(float s) { return 0.5f - 0.5f * cosf(PI * s); }

void sit_stages(double s, float* a, float* b) {
  *a = ease(clampf(static_cast<float>(s / SIT_SPLIT), 0.0f, 1.0f));
  *b = ease(clampf(static_cast<float>((s - SIT_SPLIT) / (1.0 - SIT_SPLIT)), 0.0f, 1.0f));
}

}  // namespace

void Pose::start(const Sensors& s, int m, double sit) {
  mode = m;
  sit_ = sit;
  lying = stretched_ = false;
  memcpy(ref_, s.q, sizeof ref_);
  calm_ = 0.0;
}

void Pose::reference_step(const Sensors& s) {
  // seated (past the crouch), lying down first stands back up to the crouch: from the seat it falls forward
  lying = lying || (mode == proto::LIE && sit_ <= SIT_SPLIT);
  if (lying) {
    const float tilt = tilt_deg(s);
    stretched_ = stretched_ || tilt >= LIE_END;
    float goal[NU], step;
    if (stretched_) {
      legs(LIE_STRETCH[0], LIE_STRETCH[1], LIE_STRETCH[2], goal);
      step = LIE_STRETCH_RATE * DT;
    } else {
      const float want = fminf(tilt + LIE_LEAD, LIE_END);
      legs(interp(want, LIE_TILTS, LIE_N_TILTS, LIE_LEGS, 3, 0), interp(want, LIE_TILTS, LIE_N_TILTS, LIE_LEGS, 3, 1),
           interp(want, LIE_TILTS, LIE_N_TILTS, LIE_LEGS, 3, 2), goal);
      step = LIE_RATE * DT;
    }
    for (int j = 0; j < NU; ++j) ref_[j] += clampf(goal[j] - ref_[j], -step, step);
  } else {
    sit_ = fmin(fmax(sit_ + (mode == proto::SIT ? 1.0 : -1.0) * DT_S / SIT_TIME, 0.0), 1.0);
    float a, b, v[3];
    sit_stages(sit_, &a, &b);
    for (int i = 0; i < 3; ++i) v[i] = rad(REST_LEGS[i] + (SIT_CROUCH[i] - REST_LEGS[i]) * a + (SIT_SEAT[i] - SIT_CROUCH[i]) * b);
    legs(v[1], v[2], -(v[1] + v[2]) - v[0], ref_);              // lean, hip, knee -> hip, knee, ankle (soles flat)
  }
}

void Pose::act(const Sensors& s, float target[NU]) {
  reference_step(s);
  memcpy(target, ref_, sizeof ref_);                            // no policy: the correction stays at zero
  clip_joints(target);
}

void Pose::sense(const Sensors& s) {
  calm_ = (!lying && sit_ == 0.0 && still(s)) ? calm_ + DT_S : 0.0;
}

bool Pose::fallen(const Sensors& s) const {
  float a, b;
  sit_stages(sit_, &a, &b);
  const float expected = SIT_CROUCH_TILT * a + (SIT_PELVIS_TILT - SIT_CROUCH_TILT) * b;
  return !lying && fabsf(tilt_deg(s) - expected) > POSE_FALL_TILT;
}

uint8_t Pose::state() const {
  if (lying) return stretched_ ? proto::LYING : proto::LYING_DOWN;
  if (mode == proto::SIT) return sit_ >= 1.0 ? proto::SITTING : proto::SITTING_DOWN;
  return proto::STANDING_UP;
}

}  // namespace cogu
