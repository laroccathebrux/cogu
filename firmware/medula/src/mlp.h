// A small multilayer perceptron (SiLU hidden layers), the actor of a brax PPO policy: normalise the observation
// with the training statistics, run the layers, then tanh (tanh_normal) or nothing (normal) on the mean.
#pragma once

namespace cogu {

struct MlpSpec {
  int n_layers;                 // weight layers (hidden + output)
  const int* sizes;             // n_layers + 1: input, hidden..., output
  const float* mean;            // [input]
  const float* inv_std;         // [input]
  const float* const* w;        // per layer, row-major [out][in]
  const float* const* b;        // per layer, [out]
  bool tanh_out;
};

constexpr int MLP_MAX_WIDTH = 256;

class Mlp {
 public:
  explicit Mlp(const MlpSpec& spec) : s_(spec) {}
  int inputs() const { return s_.sizes[0]; }
  int outputs() const { return s_.sizes[s_.n_layers]; }
  void eval(const float* obs, float* out);

 private:
  const MlpSpec& s_;
  float buf_[2][MLP_MAX_WIDTH];
};

}  // namespace cogu
