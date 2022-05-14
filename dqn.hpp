#ifndef DQN_HPP
#define DQN_HPP
#include <torch/torch.h>

struct LocallyConnected2DImpl : torch::nn::Module {
  LocallyConnected2DImpl(const int64_t _in_channels,
                         const int64_t _out_channels,
                         const int64_t _output_size0,
                         const int64_t _output_size1,
                         const int64_t _kernel_size, const int64_t _stride)
      : m_in_channels(_in_channels), m_out_channels(_out_channels),
        m_output_size0(_output_size0), m_output_size1(_output_size1),
        m_kernel(_kernel_size), m_stride(_stride) {
    m_W = register_parameter(
        "m_W", torch::randn({1, _out_channels, _in_channels, _output_size0,
                             _output_size1, _kernel_size * _kernel_size}));
    m_b = register_parameter(
        "m_b", torch::randn({1, _out_channels, _output_size0, _output_size1}));
  }

  torch::Tensor forward(torch::Tensor x) {
    auto batches = x.size(0);
    x = x.unfold(2, m_kernel, m_stride).unfold(3, m_kernel, m_stride);
    x = x.contiguous().view(
        {batches, m_in_channels, m_output_size0, m_output_size1, -1});
    auto out = (x.unsqueeze(1) * m_W).sum({2, -1});
    out = out + m_b;
    return out;
  }

  int64_t m_in_channels;
  int64_t m_out_channels;
  int64_t m_output_size0;
  int64_t m_output_size1;
  int64_t m_kernel;
  int64_t m_stride;
  torch::Tensor m_W;
  torch::Tensor m_b;
};
TORCH_MODULE(LocallyConnected2D);

// TODO: coop with RandomEngine
struct NoisyLinearImpl : torch::nn::Module {
  NoisyLinearImpl(const int64_t _in_features, const int64_t _out_features,
                  const float _std_init)
      : m_in_features(_in_features), m_out_features(_out_features),
        m_std_init(_std_init),
        m_weight_mu(register_parameter(
            "m_weight_mu",
            torch::zeros({_out_features, _in_features},
                         torch::dtype(torch::kFloat32).requires_grad(true)))),
        m_weight_sigma(register_parameter(
            "m_weight_sigma",
            torch::zeros({_out_features, _in_features},
                         torch::dtype(torch::kFloat32).requires_grad(true)))),
        m_weight_epsilon(register_buffer(
            "m_weight_epsilon",
            torch::zeros({_out_features, _in_features},
                         torch::dtype(torch::kFloat32).requires_grad(false)))),
        m_bias_mu(register_parameter(
            "m_bias_mu",
            torch::zeros({_out_features},
                         torch::dtype(torch::kFloat32).requires_grad(true)))),
        m_bias_sigma(register_parameter(
            "m_bias_sigma",
            torch::zeros({_out_features},
                         torch::dtype(torch::kFloat32).requires_grad(true)))),
        m_bias_epsilon(register_buffer(
            "m_bias_epsilon",
            torch::zeros({_out_features},
                         torch::dtype(torch::kFloat32).requires_grad(false)))) {
    resetParameters();
    resetNoise();
  }

  torch::Tensor forward(torch::Tensor x) {
    if (is_training()) {
      auto weight = m_weight_mu + m_weight_sigma.mul(m_weight_epsilon);
      auto bias = m_bias_mu + m_bias_sigma.mul(m_bias_epsilon);
      return at::linear(x, weight, bias);
    } else {
      return at::linear(x, m_weight_mu, m_bias_mu);
    }
  }

  inline void resetParameters() {
    const float mu_range = 1.f / std::sqrt(m_weight_mu.size(1));

    m_weight_mu.data().uniform_(-mu_range, mu_range);
    m_weight_sigma.data().fill_(m_std_init / std::sqrt(m_weight_sigma.size(1)));

    m_bias_mu.data().uniform_(-mu_range, mu_range);
    m_bias_sigma.data().fill_(m_std_init / std::sqrt(m_bias_sigma.size(0)));
  }

  inline void resetNoise() {
    auto epsilon_in = scaleNoise(m_in_features);
    auto epsilon_out = scaleNoise(m_out_features);

    m_weight_epsilon.copy_(epsilon_out.ger(epsilon_in));
    m_bias_epsilon.copy_(epsilon_out);
  }

  static torch::Tensor scaleNoise(int size) {
    torch::Tensor ret = torch::randn(size);
    ret = ret.sign().mul(ret.abs().sqrt());
    return ret;
  }

  int64_t m_in_features;
  int64_t m_out_features;
  float m_std_init;
  torch::Tensor m_weight_mu;
  torch::Tensor m_weight_sigma;
  torch::Tensor m_weight_epsilon;
  torch::Tensor m_bias_mu;
  torch::Tensor m_bias_sigma;
  torch::Tensor m_bias_epsilon;
};
TORCH_MODULE(NoisyLinear);

static int compute_output_size(int w, int f, int s, int p) {
  return (w - f + 2 * p) / s + 1;
}

struct LinearOnlyDQN : torch::nn::Module {
  LinearOnlyDQN(const int64_t _in_channels, const int64_t _input_size,
                const int64_t _output_size, float)
      : linear1(register_module(
            "linear1",
            torch::nn::Linear(torch::nn::LinearOptions(
                                  _input_size * _input_size * _in_channels, 16)
                                  .bias(true)))),
        linear2(register_module(
            "linear2",
            torch::nn::Linear(
                torch::nn::LinearOptions(16, _output_size).bias(true)))) {}

  torch::Tensor forward(torch::Tensor geometric) {
    auto input = torch::sigmoid(linear1(geometric.view(
        {-1, geometric.size(1) * geometric.size(2) * geometric.size(3)})));
    return linear2(input);
  }

  void resetNoise() {}

  torch::nn::Linear linear1, linear2;
};

struct SmallDQN : torch::nn::Module {
  static constexpr int local_kernel = 3;
  static constexpr int local_stride = 1;
  static constexpr int local_pad = 0;
  static constexpr int local_channels = 2;
  SmallDQN(const int64_t _in_channels, const int64_t _input_size,
           const int64_t _output_size)
      : m_local_output_size(
            compute_output_size(_input_size, /*kernel*/ local_kernel,
                                /*stride*/ local_stride, /*pad*/ local_pad)),
        m_local1(register_module(
            "m_local1", LocallyConnected2D(
                            _in_channels, local_channels, m_local_output_size,
                            m_local_output_size, local_kernel, local_stride))),
        linear1(register_module(
            "linear1",
            torch::nn::Linear(torch::nn::LinearOptions(m_local_output_size *
                                                           m_local_output_size *
                                                           local_channels,
                                                       16)
                                  .bias(true)))),
        linear2(register_module(
            "linear2",
            torch::nn::Linear(
                torch::nn::LinearOptions(16, _output_size).bias(true)))) {}

  torch::Tensor forward(torch::Tensor geometric) {
    auto input = torch::sigmoid(m_local1(geometric));
    input = torch::sigmoid(linear1(
        input.view({-1, input.size(1) * input.size(2) * input.size(3)})));
    return linear2(input);
  }
  unsigned m_local_output_size = 0;
  LocallyConnected2D m_local1;
  torch::nn::Linear linear1, linear2;
};

struct ConvOnlyDQN : torch::nn::Module {
  ConvOnlyDQN(const int64_t _in_channels, const int64_t _input_size,
              const int64_t _output_size, float)
      : m_local_output_size(compute_output_size(_input_size, /*kernel*/ 3,
                                                /*stride*/ 1, /*pad*/ 1)),
        m_conv1(register_module(
            "m_conv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(_in_channels, 8, 3)
                                  .stride(1)
                                  .padding(1)
                                  .bias(true)))),

        linear1(register_module(
            "linear1",
            torch::nn::Linear(
                torch::nn::LinearOptions(
                    m_local_output_size * m_local_output_size * 8, 16)
                    .bias(true)))),
        linear2(register_module(
            "linear2",
            torch::nn::Linear(
                torch::nn::LinearOptions(16, _output_size).bias(true)))) {}

  torch::Tensor forward(torch::Tensor geometric) {
    auto input = torch::sigmoid(m_conv1(geometric));
    input = torch::sigmoid(linear1(
        input.view({-1, input.size(1) * input.size(2) * input.size(3)})));
    return linear2(input);
  }

  void resetNoise() {}

  unsigned m_local_output_size = 0;
  torch::nn::Conv2d m_conv1;
  torch::nn::Linear linear1, linear2;
};

struct BigDQN : torch::nn::Module {
  static constexpr int local_kernel = 1;
  static constexpr int local_stride = 1;
  static constexpr int local_pad = 0;
  static constexpr int local_channels = 4;

  BigDQN(const int64_t _in_channels, const int64_t _input_size,
         const int64_t _output_size, const float _std_init,
         const int64_t _atom_count, const float _v_min, const float _v_max)
      : m_local_output_size(compute_output_size(_input_size, local_kernel,
                                                local_stride, local_pad)),
        m_output_size(_output_size), m_atom_count(_atom_count), m_v_min(_v_min),
        m_v_max(_v_max),
        m_local1(register_module(
            "m_local1", LocallyConnected2D(
                            _in_channels, local_channels, m_local_output_size,
                            m_local_output_size, local_kernel, local_stride))),
        m_conv1(register_module(
            "m_conv1",
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(_in_channels, 8, 3).stride(1).padding(1).bias(
                    true)))),
        m_bn_1(register_module("m_bn_1", torch::nn::BatchNorm2d(8))),
        m_conv2(register_module(
            "m_conv2",
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(8, 16, 3).stride(1).padding(1).bias(
                    true)))),
        m_bn_2(register_module("m_bn_2", torch::nn::BatchNorm2d(16))),
        m_conv3(register_module(
            "m_conv3",
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(16, 32, 3).stride(1).padding(1).bias(
                    true)))),
        m_bn_3(register_module("m_bn_3", torch::nn::BatchNorm2d(32))),
        m_conv4(register_module(
            "m_conv4",
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(32, 64, 3).stride(1).padding(1).bias(
                    true)))),
        m_bn_4(register_module("m_bn_4", torch::nn::BatchNorm2d(64))),
        m_conv5(register_module(
            "m_conv5", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3)
                                             .stride(1)
                                             .padding(1)
                                             .bias(true)))),
        m_bn_5(register_module("m_bn_5", torch::nn::BatchNorm2d(128))),
        m_conv6(register_module(
            "m_conv6", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3)
                                             .stride(1)
                                             .padding(1)
                                             .bias(true)))),
        m_bn_6(register_module("m_bn_6", torch::nn::BatchNorm2d(256))),
        m_conv7(register_module(
            "m_conv7", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3)
                                             .stride(1)
                                             .padding(1)
                                             .bias(true)))),
        m_bn_7(register_module("m_bn_7", torch::nn::BatchNorm2d(512))),

				m_linear1_v(register_module(
            "m_linear1_v",
            NoisyLinear(16928, 512, _std_init))),
        m_linear2_v(register_module(
            "m_linear2_v",
            NoisyLinear(512, m_atom_count, _std_init))),

        m_linear1_a(register_module(
            "m_linear1_a", NoisyLinear(16928, 512, _std_init))),
        m_linear2_a(register_module(
            "m_linear2_a",
            NoisyLinear(512, _output_size * m_atom_count, _std_init))) {}
//        m_linear1_v(register_module(
//            "m_linear1_v",
//            torch::nn::Linear(
//                torch::nn::LinearOptions(16928, 512).bias(true)))),
//        m_linear2_v(register_module(
//            "m_linear2_v",
//            torch::nn::Linear(
//                torch::nn::LinearOptions(512, m_atom_count).bias(true)))),

  //          m_linear1_a(register_module("m_linear1_a",
  //          torch::nn::Linear(torch::nn::LinearOptions(3872 /*61952*/,
  //          512).bias(true)))), m_linear2_a(register_module("m_linear2_a",
  //          torch::nn::Linear(torch::nn::LinearOptions(512,
  //          _output_size*m_atom_count).bias(true)))) {}

  torch::Tensor forward(torch::Tensor geometric) {
    auto input = torch::relu(m_conv1(geometric));
    input = torch::relu(m_bn_2(m_conv2(input)));
    input = torch::relu(m_conv3(input));
    //        input = torch::relu(m_bn_4(m_conv4(input)));
    //        input = torch::relu(m_bn_5(m_conv5(input)));
    //        input = torch::relu(m_bn_6(m_conv6(input)));
    //        input = torch::relu(m_bn_7(m_conv7(input)));

    input = input.view({-1, input.size(1) * input.size(2) * input.size(3)});
    auto advantage = torch::relu(m_linear1_a(input));
    advantage = m_linear2_a(advantage);

    auto value = torch::relu(m_linear1_v(input));
    value = m_linear2_v(value);

    value = value.view({-1, 1, m_atom_count});
    advantage = advantage.view({-1, m_output_size, m_atom_count});

    auto output = value + advantage - advantage.mean(1, /*keepdim*/ true);
    output = torch::nn::functional::softmax(
        output, torch::nn::functional::SoftmaxFuncOptions(2));
		
    return output;
  }

  inline void resetNoise() {
    m_linear1_v->resetNoise();
    m_linear2_v->resetNoise();

    m_linear1_a->resetNoise();
    m_linear2_a->resetNoise();
  }

  unsigned m_local_output_size = 0;
  int64_t m_output_size, m_atom_count;
  float m_v_min, m_v_max;
  LocallyConnected2D m_local1;
  torch::nn::Conv2d m_conv1, m_conv2, m_conv3, m_conv4, m_conv5, m_conv6,
      m_conv7;
  torch::nn::BatchNorm2d m_bn_1, m_bn_2, m_bn_3, m_bn_4, m_bn_5, m_bn_6, m_bn_7;
//  torch::nn::Linear m_linear1_v, m_linear2_v; //,m_linear1_a,m_linear2_a;
  NoisyLinear m_linear1_v, m_linear2_v;
  NoisyLinear m_linear1_a, m_linear2_a;

};

#endif /* DQN_HPP */
