// A C interface to the medula for the simulator (software in the loop): sim/brain/medula_c.py loads this as a shared
// library and runs the firmware's own code in place of medula.py, MuJoCo being the body. Not built for the ESP32.
#include <stdint.h>

#include "imu.h"
#include "medula.h"

using cogu::Medula;
using cogu::Sensors;

extern "C" {

int cogu_nu() { return cogu::NU; }
int cogu_sensors_size() { return static_cast<int>(sizeof(Sensors)); }

void* cogu_medula_new() { return new Medula(); }
void cogu_medula_free(void* m) { delete static_cast<Medula*>(m); }
void cogu_medula_begin(void* m, const Sensors* s) { static_cast<Medula*>(m)->begin(*s); }
void cogu_medula_sense(void* m, const Sensors* s) { static_cast<Medula*>(m)->sense(*s); }
void cogu_medula_receive(void* m, const uint8_t* data, int n, double t) {
  static_cast<Medula*>(m)->receive(data, n, t);
}
void cogu_medula_act(void* m, float* target) { static_cast<Medula*>(m)->act(target); }
int cogu_medula_telemetry(void* m, uint8_t* out) { return static_cast<Medula*>(m)->telemetry(out); }
int cogu_medula_state(void* m) { return static_cast<Medula*>(m)->state(); }
int cogu_medula_mode(void* m) { return static_cast<Medula*>(m)->mode(); }

// cmd vx vy wz, time, falls, watchdog, cutoff
void cogu_medula_info(void* m, double* out) {
  const Medula* md = static_cast<Medula*>(m);
  for (int i = 0; i < 3; ++i) out[i] = md->cmd()[i];
  out[3] = md->time();
  out[4] = md->falls();
  out[5] = md->watchdog();
  out[6] = md->cutoff();
}

// commands, rejected, stale, watchdog trips, longest gap (s)
void cogu_medula_stats(void* m, double* out) {
  const cogu::MedulaStats& s = static_cast<Medula*>(m)->stats();
  out[0] = s.commands;
  out[1] = s.rejected;
  out[2] = s.stale;
  out[3] = s.watchdog;
  out[4] = s.longest_gap;
}

// the policies alone, for the equivalence check against JAX
void cogu_policy_eval(int which, const float* obs, float* out) {
  static cogu::Mlp walk(cogu::walk_policy()), getup(cogu::getup_policy());
  (which == 0 ? walk : getup).eval(obs, out);
}

void* cogu_imu_new(float gain) { return new cogu::GravityFilter(gain); }
void cogu_imu_free(void* f) { delete static_cast<cogu::GravityFilter*>(f); }
void cogu_imu_update(void* f, const float* gyro, const float* accel, float dt, float* gravity) {
  auto* g = static_cast<cogu::GravityFilter*>(f);
  g->update(gyro, accel, dt);
  g->gravity(gravity);
}

}  // extern "C"
