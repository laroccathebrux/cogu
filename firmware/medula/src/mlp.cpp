#include "mlp.h"

#include <math.h>

namespace cogu {

void Mlp::eval(const float* obs, float* out) {
  const int n0 = s_.sizes[0];
  float* x = buf_[0];
  for (int i = 0; i < n0; ++i) x[i] = (obs[i] - s_.mean[i]) * s_.inv_std[i];
  for (int l = 0; l < s_.n_layers; ++l) {
    const int n_in = s_.sizes[l], n_out = s_.sizes[l + 1];
    const bool last = l == s_.n_layers - 1;
    float* y = last ? out : buf_[(l + 1) & 1];
    const float* w = s_.w[l];
    const float* b = s_.b[l];
    for (int o = 0; o < n_out; ++o) {
      const float* row = w + o * n_in;
      float acc = b[o];
      for (int i = 0; i < n_in; ++i) acc += row[i] * x[i];
      if (!last) acc = acc / (1.0f + expf(-acc));          // SiLU
      else if (s_.tanh_out) acc = tanhf(acc);
      y[o] = acc;
    }
    x = y;
  }
}

}  // namespace cogu
