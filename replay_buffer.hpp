/*
 * replay_buffer.hpp
 *
 *  Created on: Apr 27, 2021
 *      Author: saul.ramirez
 */

#ifndef REPLAY_BUFFER_HPP_
#define REPLAY_BUFFER_HPP_

#include "board_config.hpp"
#include "data_objects.hpp"
#include "math_util.hpp"
#include "model_config.hpp"
#include "template_util.hpp"
#include <Eigen/Dense>
#include <torch/torch.h>

template <torch::DeviceType DeviceType, typename BatchType,
          typename ExampleType>
class ReplayBuffer {
public:
  ReplayBuffer(const unsigned _capacity, const unsigned _batch_size,
               const float _alpha, const float _beta, const float _beta_decay)
      : m_is_capacity_reached(false), m_capacity(_capacity),
        m_batch_size(_batch_size), m_alpha(_alpha), m_beta(_beta),
        m_beta_decay(_beta_decay), m_prios(Eigen::ArrayXf::Zero(_capacity)),
        m_cumsum(Eigen::ArrayXf::Zero(_capacity)), m_choices(_batch_size),
        m_buffer(_capacity), m_batch(_batch_size), m_pos(0),
        m_weights_on_cpu(
            torch::zeros({_batch_size}, torch::dtype(torch::kFloat32)
                                            .requires_grad(false)
                                            .device(torch::kCPU))) {}

  template <typename RandomEngine>
  const BatchType &sample(RandomEngine &random_engine_, const unsigned _frame) {
    const float max_prio = m_pos > 0 ? m_prios.maxCoeff() : 1;
    const auto prios = m_is_capacity_reached ? m_prios : m_prios.head(m_pos);
    Eigen::ArrayXf probs = prios.pow(m_alpha);
    probs /= probs.sum();
    choice(random_engine_, probs, m_cumsum.head(probs.size()), m_choices);

    std::cout << "REPLAY prios: " << std::endl;
    std::cout << prios.maxCoeff() << ", " << prios.minCoeff() << std::endl;

    for (int i = 0; i < m_batch_size; ++i) {
      auto const &sample = m_buffer[m_choices(i)];
      m_batch.set(i, sample);
    }

    const auto weights =
        (probs.size() * probs)
            .pow(-std::min(1.f, m_beta + _frame * (1 - m_beta) / m_beta_decay));
    auto weights_on_cpu_a = m_weights_on_cpu.accessor<float, 1>();
    for (int i = 0; i < m_batch_size; ++i) {
      weights_on_cpu_a[i] = weights(m_choices(i));
    }
    m_batch.m_weights.index_put_(
        {torch::indexing::Slice(0, m_batch_size, 1)},
        m_weights_on_cpu.index({torch::indexing::Slice(0, m_batch_size, 1)})
            .to(DeviceType, false, false));

    return m_batch;
  }

  void updatePrios(const torch::Tensor &_prios) {
    auto prios_cpu = _prios.cpu().squeeze();
    auto a = prios_cpu.accessor<float, 1>();
    for (int i = 0; i < m_batch_size; ++i) {
      m_prios(m_choices(i)) = a[i];
    }
  }

  void push(const ExampleType &_example) {

    const float max_prio = m_pos > 0 ? m_prios.maxCoeff() : 1;

    m_buffer[m_pos] = _example;
    m_prios(m_pos) = max_prio;
    m_pos = (m_pos + 1) % m_capacity;

    if (!m_is_capacity_reached) {
      m_is_capacity_reached = m_pos == (m_capacity - 1);
    }
  }

  void push(const BatchType &_batch, const std::size_t _obj_count) {
    const float max_prio = m_pos > 0 ? m_prios.maxCoeff() : 1;
    for (int i = 0; i < _obj_count; ++i) {

      auto &example = m_buffer[m_pos];
      example.m_state.m_geometric.index_put_(
          {torch::indexing::None}, _batch.m_state.m_geometric.index({i}));

      example.m_state.m_temporal.index_put_(
          {torch::indexing::None}, _batch.m_state.m_temporal.index({i}));

      example.m_action = _batch.m_action.index({i}).item().template to<int>();
      example.m_reward = _batch.m_reward.index({i}).item().template to<float>();
      example.m_is_non_terminal =
          _batch.m_is_non_terminal.index({i}).item().template to<bool>();

      example.m_next_state.m_geometric.index_put_(
          {torch::indexing::None}, _batch.m_next_state.m_geometric.index({i}));

      example.m_next_state.m_temporal.index_put_(
          {torch::indexing::None}, _batch.m_next_state.m_temporal.index({i}));

      m_prios(m_pos) = max_prio;
      m_pos = (m_pos + 1) % m_capacity;
      if (m_pos == (m_capacity - 1)) {
        std::cout << "capacity reset" << std::endl;
      }
      m_is_capacity_reached =
          m_is_capacity_reached ? true : m_pos == (m_capacity - 1);
    }
  }

private:
  bool m_is_capacity_reached;
  unsigned m_capacity;
  unsigned m_batch_size;
  float m_alpha;
  float m_beta;
  float m_beta_decay;
  Eigen::ArrayXf m_prios;
  Eigen::ArrayXf m_cumsum;
  Eigen::ArrayXi m_choices;
  std::vector<ExampleType> m_buffer;
  BatchType m_batch;
  unsigned m_pos;
  torch::Tensor m_weights_on_cpu;
};

#endif /* REPLAY_BUFFER_HPP_ */
