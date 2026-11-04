// The three ways the medula drives the legs: the walker (walk v10), the get-up (get-up v8) and the scripted sit / lie
// down. Each is a port of its sim2sim class in sim/playground/play.py (Sim2Sim, GetupPolicy, PosePolicy), with the
// same order of operations: `act` gives the servo targets for this step from the last sensor reading, `sense` takes
// the reading after the step.
#pragma once

#include <stdint.h>

#include "body.h"
#include "mlp.h"

namespace cogu {

class Walker {
 public:
  Walker();
  float command[3] = {0, 0, 0};          // vx, vy, wz: what act() walks at (set by the medula)
  void reset(const Sensors& s);          // the rest pose, filter at zero (power-on)
  void handover(const Sensors& s);       // take over a robot already standing: the first target is where it is
  void act(float target[NU]);
  void sense(const Sensors& s);
  bool fallen(const Sensors& s) const;
  static constexpr int N_OBS = 43;
  const float* obs() const { return obs_; }

 private:
  void build_obs(const Sensors& s);
  void restart(const Sensors& s);
  Mlp mlp_;
  float obs_[N_OBS];
  float action_[NU], last_act_[NU], filtered_[NU];
  float phase_[2];
  double recover_time_ = 0, recover_cooldown_ = 0;   // timers in double, as the Python
};

class Getup {
 public:
  Getup();
  void start(const Sensors& s);
  void act(const Sensors& s, float target[NU]);
  void sense(const Sensors& s);
  bool ready() const { return calm_ >= 0.5; }
  static constexpr int N_OBS = 48;

 private:
  void reference_step(const Sensors& s);
  Mlp mlp_;
  float ref_[NU], filtered_[NU], last_act_[NU];
  double throw_t_ = -1, lying_t_ = 0, calm_ = 0;
  int side_ = 0;
  bool rolled_ = false;
};

class Pose {
 public:
  void start(const Sensors& s, int mode, double sit);
  void act(const Sensors& s, float target[NU]);
  void sense(const Sensors& s);
  bool fallen(const Sensors& s) const;
  bool ready() const { return mode == 0 && calm_ >= 0.5; }
  uint8_t state() const;                 // proto::State: sitting down / sitting / lying down / lying / standing up
  int mode = 0;                          // proto::Posture asked for
  bool lying = false;

 private:
  void reference_step(const Sensors& s);
  float ref_[NU];
  double sit_ = 0, calm_ = 0;
  bool stretched_ = false;
};

// the two policies (the weights live in one translation unit only: they are 320 kB of flash)
const MlpSpec& walk_policy();
const MlpSpec& getup_policy();

float tilt_deg(const Sensors& s);        // pelvis lean in the sagittal plane, + backwards (as play.py)

}  // namespace cogu
