// The medula (ADR-005): owns the body, takes COMMAND packets from the phone, sends TELEMETRY back. A port of
// sim/brain/medula.py; its rules, whatever the phone says: ramp and limits, watchdog, one posture per request, get up
// after a fall, sit down when the battery is flat.
//
// One control step (50 Hz), on the robot and in the simulator alike:
//     sense(reading after the last step)  ->  receive(each packet that arrived)  ->  act(servo targets)
#pragma once

#include <stdint.h>

#include "body.h"
#include "controllers.h"
#include "protocol.h"

namespace cogu {

struct MedulaStats {
  uint32_t commands = 0, rejected = 0, stale = 0, watchdog = 0;
  float longest_gap = 0;   // s between two valid commands
};

class Medula {
 public:
  enum Mode : uint8_t { WALKING, POSE, GETTING_UP };

  // power-on. `from_rest`: the simulator starts standing in `rest` (the walker's filter at zero, as play.py); the
  // robot starts wherever the legs are, so the first target is where the joints already are (no jump)
  void begin(const Sensors& s, bool from_rest = true);
  void set_servo_fault(bool fault) { servo_fault_ = fault; }   // the main loop's bus check, reported in TELEMETRY
  void sense(const Sensors& s);
  void receive(const uint8_t* data, int n, double t);
  void act(float target[NU]);
  int telemetry(uint8_t out[proto::TEL_SIZE]);

  uint8_t state() const;                        // proto::State
  Mode mode() const { return mode_; }
  double time() const { return t_; }
  const float* cmd() const { return cmd_; }
  uint8_t falls() const { return falls_; }
  bool watchdog() const { return watchdog_; }
  bool cutoff() const { return cutoff_; }
  const MedulaStats& stats() const { return stats_; }
  const Sensors& last() const { return s_; }

 private:
  void posture(uint8_t kind);
  void fell();
  void walk();

  Walker walker_;
  Getup getup_;
  Pose pose_;
  Sensors s_{};
  Mode mode_ = WALKING;
  bool lie_when_seated_ = false;
  float target_[3] = {0, 0, 0}, cmd_[3] = {0, 0, 0};
  bool have_seq_ = false, have_posture_ = false;
  uint8_t seq_ = 0, posture_id_ = 0;
  double last_cmd_ = -1e300, t_ = 0;
  uint8_t falls_ = 0, tel_seq_ = 0;
  bool cutoff_ = false, watchdog_ = false, acted_ = false, servo_fault_ = false;
  MedulaStats stats_;
};

}  // namespace cogu
