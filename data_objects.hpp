#pragma once

#include "board_config.hpp"
#include "math_util.hpp"
#include "model_config.hpp"
#include "template_util.hpp"
#include <Eigen/Dense>
#include <torch/torch.h>

template <torch::DeviceType DeviceType, std::size_t size,
          typename ModelConfig>
struct BatchRewardFeature {};

template <torch::DeviceType DeviceType, std::size_t size>
struct BatchRewardFeature<DeviceType, size, WorkerModelConfig> {
  BatchRewardFeature(const unsigned _batch_size) {}
};

template <torch::DeviceType DeviceType, std::size_t size>
struct BatchRewardFeature<DeviceType, size, CartModelConfig> {
  BatchRewardFeature(const unsigned _batch_size) {}
};

template <torch::DeviceType DeviceType, std::size_t size>
struct BatchRewardFeature<DeviceType, size, CityTileModelConfig> {
  BatchRewardFeature(const unsigned _batch_size) {}
};



template <torch::DeviceType DeviceType, std::size_t size,
          typename ModelConfig>
struct BatchStateFeature {
  BatchStateFeature(unsigned _batch_size)
      : m_batch_size(_batch_size),
        m_geometric(torch::zeros({_batch_size, ModelConfig::channels,
                                  size, size},
                                 torch::dtype(torch::kFloat32)
                                     .requires_grad(false)
                                     .device(DeviceType))),
        m_temporal(torch::zeros({_batch_size, ModelConfig::ts_ftr_count},
                                torch::dtype(torch::kFloat32)
                                    .requires_grad(false)
                                    .device(DeviceType))),
        m_reward_ftrs(_batch_size) {}

  BatchStateFeature(const BatchStateFeature &_other)
      : m_batch_size(_other.m_batch_size),
        m_geometric(_other.m_geometric.detach().clone()),
        m_temporal(_other.m_temporal.detach().clone()),
        m_reward_ftrs(_other.m_reward_ftrs) {}

  BatchStateFeature &operator=(const BatchStateFeature &_other) {
    if (this == &_other)
      return *this;
    m_batch_size = _other.m_batch_size;
    m_geometric = _other.m_geometric.detach().clone();
    m_temporal = _other.m_temporal.detach().clone();
    m_reward_ftrs = _other.m_reward_ftrs;
    return *this;
  }

  BatchStateFeature(BatchStateFeature &&other_)
      : m_batch_size(other_.m_batch_size),
        m_geometric(std::move(other_.m_geometric)),
        m_temporal(std::move(other_.m_temporal)),
        m_reward_ftrs(std::move(other_.m_reward_ftrs)) {}

  unsigned m_batch_size;
  torch::Tensor m_geometric;
  torch::Tensor m_temporal;
  BatchRewardFeature<DeviceType, size, ModelConfig> m_reward_ftrs;
};

template <torch::DeviceType DeviceType, std::size_t size,
          typename ModelConfig>
struct SingleStateFeature {
  SingleStateFeature()
      : m_geometric(torch::zeros(
            {ModelConfig::channels, size, size},
            torch::dtype(torch::kFloat32)
                .requires_grad(false)
                .device(DeviceType))),
        m_temporal(torch::zeros({ModelConfig::ts_ftr_count},
                                torch::dtype(torch::kFloat32)
                                    .requires_grad(false)
                                    .device(DeviceType))) {}


  SingleStateFeature(const SingleStateFeature &_other)
      : m_geometric(_other.m_geometric.detach().clone()),
        m_temporal(_other.m_temporal.detach().clone()) {}

  SingleStateFeature &operator=(const SingleStateFeature &_other) {
    if (this == &_other)
      return *this;
    m_geometric = _other.m_geometric.detach().clone();
    m_temporal = _other.m_temporal.detach().clone();
    return *this;
  }

  SingleStateFeature(SingleStateFeature &&other_)
      : m_geometric(std::move(other_.m_geometric)),
        m_temporal(std::move(other_.m_temporal)){}

  torch::Tensor m_geometric;
  torch::Tensor m_temporal;
};

template <torch::DeviceType DeviceType, std::size_t size,
          typename ModelConfig>
struct Transition {
  using state_t = SingleStateFeature<DeviceType, size, ModelConfig>;
  using action_t = int;
  using reward_t = float;
  using is_non_terminal_t = bool;
  Transition()
      : m_state(), m_action(0), m_reward(0), m_next_state(),
        m_is_non_terminal(true) {}

  SingleStateFeature<DeviceType, size, ModelConfig> m_state;
  int m_action;
  float m_reward;
  SingleStateFeature<DeviceType, size, ModelConfig> m_next_state;
  bool m_is_non_terminal;
};

template <torch::DeviceType DeviceType, std::size_t size,
          typename ModelConfig>
struct TrainingBatch {

  TrainingBatch(const unsigned _batch_size = size * size)
      : m_batch_size(_batch_size), m_state(_batch_size),
        m_action(torch::empty(_batch_size, torch::dtype(torch::kInt64)
                                               .requires_grad(false)
                                               .device(DeviceType))),
        m_reward(torch::empty(_batch_size, torch::dtype(torch::kFloat32)
                                               .requires_grad(false)
                                               .device(DeviceType))),
        m_next_state(_batch_size),
        m_is_non_terminal(torch::zeros(_batch_size, torch::dtype(torch::kBool)
                                                        .requires_grad(false)
                                                        .device(DeviceType))),
        m_weights(torch::empty(_batch_size, torch::dtype(torch::kFloat32)
                                                .requires_grad(false)
                                                .device(DeviceType))),
        m_reward_ftrs(_batch_size) {}

  inline void zero_() {
    m_state.m_geometric.zero_();
    m_state.m_temporal.zero_();
    m_next_state.m_geometric.zero_();
    m_next_state.m_temporal.zero_();
    m_action.zero_();
    m_reward.zero_();
  }

  void set(const int _index, const Transition<DeviceType, size, ModelConfig> &_example) {
    // b, c, h, w
    // geometric[_index]
    m_state.m_geometric.index_put_({_index}, _example.m_state.m_geometric);
    // ts[_index]
    m_state.m_temporal.index_put_({_index}, _example.m_state.m_temporal);

    // action
    m_action.index_put_({_index}, _example.m_action);

    // reward
    m_reward.index_put_({_index}, _example.m_reward);

    // next state
    m_next_state.m_geometric.index_put_({_index},
                                        _example.m_next_state.m_geometric);
    m_next_state.m_temporal.index_put_({_index},
                                       _example.m_next_state.m_temporal);

    // is terminal
    m_is_non_terminal.index_put_({_index}, _example.m_is_non_terminal);
  }

  unsigned m_batch_size;
  BatchStateFeature<DeviceType, size, ModelConfig> m_state;
  torch::Tensor m_action;
  torch::Tensor m_reward;
  BatchStateFeature<DeviceType, size, ModelConfig> m_next_state;
  torch::Tensor m_is_non_terminal;
  torch::Tensor m_weights;
  BatchRewardFeature<DeviceType, size, ModelConfig> m_reward_ftrs;
};
