/*
 * model_trainer.hpp
 *
 *  Created on: May 28, 2021
 *      Author: saul.ramirez
 */

#ifndef MODEL_LEARNER_HPP_
#define MODEL_LEARNER_HPP_

#include "dqn.hpp"
#include "hyper_parameters.hpp"
#include <chrono>
#include <torch/torch.h>

template <typename Model>
static void copy_state_dict(Model &to_, const Model &_from) {
  torch::autograd::GradMode::set_enabled(false);
  auto update_params = to_.named_parameters(true);
  auto update_buffers = to_.named_buffers(true);
  const auto copy_params = _from.named_parameters(true /*recurse*/);
  const auto copy_buffers = _from.named_buffers(true /*recurse*/);

  for (const auto &kv : copy_params) {
    auto *tensor = update_params.find(kv.key());
    if (tensor != nullptr) {
      tensor->copy_(kv.value());
    }
  }

  for (const auto &kv : copy_buffers) {
    auto *tensor = update_buffers.find(kv.key());
    if (tensor != nullptr) {
      tensor->copy_(kv.value());
    }
  }
  torch::autograd::GradMode::set_enabled(true);
}

template <typename DQN, torch::DeviceType DeviceType,
          std::size_t action_space_size>
class ModelLearner {
public:
  template <typename... Args>
  ModelLearner(DQN &dqn_dynamic_, Args &&... args)
      : m_batch_size(HyperParameters::m_replay_batch_size), m_vmin(HyperParameters::m_nn_v_min),
        m_vmax(HyperParameters::m_nn_v_max), m_atom_count(HyperParameters::m_nn_atom_count),
        m_target_model_update(HyperParameters::m_nn_target_model_update),
        m_gamma(std::pow(HyperParameters::m_nn_gamma,HyperParameters::m_nn_step_size)), m_lr(HyperParameters::m_nn_lr),
        m_momentum(HyperParameters::m_nn_momentum), m_weight_decay(HyperParameters::m_nn_weight_decay),
        m_delta_z((HyperParameters::m_nn_v_max - HyperParameters::m_nn_v_min) /
                  (HyperParameters::m_nn_atom_count - 1)),
        m_elapsed_ms(0), m_reps(0), m_dqn_dynamic(dqn_dynamic_),
        m_dqn_target(std::forward<Args>(args)...),
        m_optimizer(dqn_dynamic_.parameters(),
                    torch::optim::SGDOptions(m_lr)
                        .momentum(m_momentum)
                        .weight_decay(m_weight_decay)),
        m_support(torch::linspace(HyperParameters::m_nn_v_min, HyperParameters::m_nn_v_max,
                                  HyperParameters::m_nn_atom_count,
                                  torch::dtype(torch::kFloat32)
                                      .requires_grad(false)
                                      .device(DeviceType))),
        m_target_distribution(torch::zeros({m_batch_size, m_atom_count},
                                           torch::dtype(torch::kFloat32)
                                               .requires_grad(false)
                                               .device(DeviceType))),
        m_offset(torch::linspace(
                     0, static_cast<int>((m_batch_size - 1) * m_atom_count),
                     static_cast<int>(m_batch_size),
                     torch::dtype(torch::kInt32)
                         .device(DeviceType)
                         .requires_grad(false))
                     .unsqueeze(1)
                     .expand({m_batch_size, m_atom_count})) {
    m_dqn_target.to(DeviceType);
    updateTargetModel();
  };

  template <typename RandomEngine, typename ReplayBuffer>
  inline void train(const unsigned _frame, ReplayBuffer &_replay_buffer,
                    RandomEngine &random_engine_) {
    auto &mini_batch = _replay_buffer.sample(random_engine_, _frame);

    torch::Tensor forward =
        m_dqn_dynamic.forward(mini_batch.m_state.m_geometric);

    torch::Tensor current_dist =
        forward
            .gather(1, mini_batch.m_action.unsqueeze(1).unsqueeze(1).expand(
                           {m_batch_size, 1, m_atom_count}))
            .squeeze(1);

    computeTargetDistribution(mini_batch);

    torch::Tensor loss = -(m_target_distribution * current_dist.log()).sum(1);

    m_optimizer.zero_grad();

    // auto start = std::chrono::high_resolution_clock::now();
    (mini_batch.m_weights.unsqueeze(1) * loss).mean().backward();

    // auto end = std::chrono::high_resolution_clock::now();
    // m_elapsed_ms += end - start;
    // m_reps++;
    // if ((m_reps % 10)==0) report();

    m_optimizer.step();

    _replay_buffer.updatePrios(loss.detach());

    if ((m_reps + 1) % m_target_model_update == 0) {
      std::cout << "updated target model" << std::endl;
      updateTargetModel();
    }
    m_dqn_dynamic.resetNoise();
    m_dqn_target.resetNoise();
    m_reps++;
  }

private:
  template <typename BatchType>
  void inline computeTargetDistribution(const BatchType &_mini_batch) {
    torch::NoGradGuard no_grad;
    m_target_distribution.zero_();

    auto dynamic_probabilities =
        m_dqn_dynamic.forward(_mini_batch.m_next_state.m_geometric);

    auto dynamic_distribution = dynamic_probabilities * m_support;

    auto dynamic_selection = std::get<1>(dynamic_distribution.sum(2).max(1))
                                 .unsqueeze(1)
                                 .unsqueeze(1)
                                 .expand({m_batch_size, 1, m_atom_count});

    auto target_probabilities = m_dqn_target.forward(
        _mini_batch.m_next_state.m_geometric); // care for  * support bug here

    auto target_selection =
        target_probabilities.gather(1, dynamic_selection).squeeze(1);
    //		auto Tz =
    //			_mini_batch.m_reward.cpu().unsqueeze(1).expand_as(target_selection)
    //+
    //			_mini_batch.m_is_non_terminal.cpu().unsqueeze(1).expand_as(target_selection)
    //* m_gamma * m_support.unsqueeze(0).expand_as(target_selection);

    auto Tz = _mini_batch.m_reward.unsqueeze(1) +
              _mini_batch.m_is_non_terminal.unsqueeze(1) * m_gamma *
                  m_support.unsqueeze(0);

    Tz.clamp_(m_vmin, m_vmax);
		
		std::cout << "actions:" << std::endl;
		std::cout << _mini_batch.m_action << std::endl;

    std::cout << "rewards:" << std::endl;
    std::cout << _mini_batch.m_reward << std::endl;

    torch::Tensor b = (Tz - m_vmin) / m_delta_z;

    torch::Tensor l = b.floor().to(torch::kInt32);
    torch::Tensor u = b.ceil().to(torch::kInt32);


    torch::Tensor same_index_mask = l == b;
    torch::Tensor mask_for_lowers = (u > 0) * same_index_mask;
    torch::Tensor mask_for_uppers =
        (l < (static_cast<int>(m_atom_count) - 1)) * same_index_mask;
    l.index_put_({mask_for_lowers}, (l - 1).index({mask_for_lowers}));
    u.index_put_({mask_for_uppers}, (u + 1).index({mask_for_uppers}));


    m_target_distribution.view({-1}).index_add_(
        0, (l + m_offset).view({-1}),
        (target_selection * (u.to(torch::kFloat32) - b)).view({-1}));
    m_target_distribution.view({-1}).index_add_(
        0, (u + m_offset).view({-1}),
        (target_selection * (b - l.to(torch::kFloat32))).view({-1}));
  }

  void inline updateTargetModel() {
    copy_state_dict(m_dqn_target, m_dqn_dynamic);
  }

  void inline report() {
    std::cout << "reps: " << m_reps << " avg ms: "
              << m_elapsed_ms.count() / static_cast<double>(m_reps)
              << std::endl;
  }

private:
  std::size_t m_batch_size;
  std::size_t m_atom_count;
  float m_vmin;
  float m_vmax;
  std::size_t m_target_model_update;
  float m_gamma;
  float m_lr;
  float m_momentum;
  float m_weight_decay;
  float m_delta_z;
  std::chrono::duration<double, std::milli> m_elapsed_ms;
  std::size_t m_reps;

  DQN &m_dqn_dynamic;
  DQN m_dqn_target;
  torch::optim::SGD m_optimizer;
  torch::Tensor m_support;
  torch::Tensor m_offset;
  torch::Tensor m_target_distribution;
};

#endif /* MODEL_LEARNER_HPP_ */
