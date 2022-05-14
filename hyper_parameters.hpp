/*
 * hyper_parameters.hpp
 *
 *  Created on: May 15, 2021
 *      Author: saul.ramirez
 */

#ifndef HYPER_PARAMETERS_HPP_
#define HYPER_PARAMETERS_HPP_

namespace HyperParameters {
  constexpr std::size_t m_replay_capacity = 15000;
  constexpr std::size_t m_replay_batch_size = 32;
  constexpr float m_replay_alpha = 0.5;
  constexpr float m_replay_beta = 0.3;
  constexpr float m_replay_beta_decay = 50000;

  constexpr float m_actor_epsilon_decay = 15000;
  constexpr float m_actor_epsilon_start = 1;
  constexpr float m_actor_epsilon_end = 0;

  constexpr float m_nn_lr = .01;
  constexpr std::size_t m_nn_step_size = 2;
  constexpr float m_nn_momentum = .9;
  constexpr float m_nn_weight_decay = 5e-4;
  constexpr float m_nn_gamma = .99;
  constexpr float m_nn_std_init = 0.1;
  constexpr float m_nn_v_min = -200;
  constexpr float m_nn_v_max = 4000.;
  constexpr std::size_t m_nn_atom_count = 41;

  constexpr std::size_t m_nn_target_model_update = 5000;

  constexpr float m_reward_mine_beta = 1.;
  constexpr float m_reward_deposit_beta = 2.;   
  constexpr float m_reward_distance_beta = 1.; 
 
  constexpr float m_reward_discovery_beta = 0.; 
  constexpr float m_reward_mine_time_beta = .5;
  constexpr float m_reward_deposit_time_beta = .5;
  constexpr float m_reward_distance_time_beta = .5;
  constexpr float m_reward_discovery_time_beta = .25;
  constexpr float m_reward_mine_min_clip = 0.;
  constexpr float m_reward_mine_max_clip = 1000.;
  constexpr float m_reward_deposit_min_clip = 0.;
  constexpr float m_reward_deposit_max_clip = 1000.;
  constexpr float m_reward_distance_min_clip = 0.;
  constexpr float m_reward_distance_max_clip = 1000.;
  constexpr float m_reward_discovery_min_clip = 0.;
  constexpr float m_reward_discovery_max_clip = .25;

	constexpr float m_reward_night_min_clip = -1000;
	constexpr float m_reward_night_max_clip = 0;
	constexpr float m_reward_worker_death = -200;
	constexpr float m_reward_worker_death_min_clip = -1000;
	constexpr float m_reward_worker_death_max_clip = 0;

};

#endif /* HYPER_PARAMETERS_HPP_ */
