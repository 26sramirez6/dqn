/*
 * Trainer.hpp
 *
 *  Created on: May 15, 2021
 *      Author: saul.ramirez
 */

#ifndef TRAINER_HPP_
#define TRAINER_HPP_

#include <tuple>
#include "actions.hpp"
#include "template_util.hpp"
#include "hyper_parameters.hpp"
#include "actor.hpp"
#include "board_config.hpp"
#include "dqn.hpp"
#include "feature_builder.hpp"
#include "math_util.hpp"
#include "model_config.hpp"
#include "model_learner.hpp"
#include "random_engine.hpp"
#include "replay_buffer.hpp"
#include "reward_engine.hpp"

template <std::size_t ActorCount,
          torch::DeviceType DeviceType, typename RandomEngine>
class Trainer {
public:
	static constexpr int augmented_size = BoardConfig::size + BoardConfig::size - 1;
  using WorkerBatch = DynamicBatch<DeviceType, augmented_size, WorkerModelConfig>;
  using CityTileBatch =
      DynamicBatch<DeviceType, augmented_size, CityTileModelConfig>;
  using WorkerTransition = Transition<DeviceType, augmented_size, WorkerModelConfig>;
  using CityTileTransition =
      Transition<DeviceType, augmented_size, CityTileModelConfig>;
  using WorkerReplayBuffer = ReplayBuffer<DeviceType, WorkerBatch, WorkerTransition>;
  using CityTileReplayBuffer =
      ReplayBuffer<DeviceType, CityTileBatch, CityTileTransition>;

	using ActionReturn = std::tuple<
		const Eigen::Ref<const Eigen::ArrayXi>,
		const Eigen::Ref<const Eigen::ArrayXi>>;


	using WorkerDQN = BigDQN;
	using CityTileDQN = SmallDQN;

  template <std::size_t ActorId>
  using ActorType =
      Actor<ActorId, DeviceType, BoardConfig, WorkerModelConfig,
            CityTileModelConfig, WorkerFeatureBuilder, CityTileFeatureBuilder,
            WorkerRewardEngine<DeviceType>, CityTileRewardEngine<DeviceType>, WorkerReplayBuffer,
            CityTileReplayBuffer, RandomEngine>;

  using Actors = std::tuple<ActorType<0>>;
  static_assert(ActorCount == 1, "Unsupported");

	Trainer() : 
		m_worker_dqn(WorkerModelConfig::channels, augmented_size,
								static_cast<uint64_t>(WorkerActions::Count),
								HyperParameters::m_nn_std_init,
								HyperParameters::m_nn_atom_count,
								HyperParameters::m_nn_v_min, HyperParameters::m_nn_v_max),
	  m_citytile_dqn(CityTileModelConfig::channels, augmented_size,
                   static_cast<uint64_t>(CityTileActions::Count)),
    m_worker_model_learner(
        m_worker_dqn, unsigned(WorkerModelConfig::channels),
        unsigned(augmented_size), static_cast<uint64_t>(WorkerActions::Count),
        HyperParameters::m_nn_std_init, HyperParameters::m_nn_atom_count,
        HyperParameters::m_nn_v_min, HyperParameters::m_nn_v_max),
    m_citytile_model_learner(m_citytile_dqn,
                           unsigned(CityTileModelConfig::channels),
                           unsigned(augmented_size),
                           static_cast<uint64_t>(CityTileActions::Count)),
		
    m_worker_replay_buffer(
        HyperParameters::m_replay_capacity,
        HyperParameters::m_replay_batch_size, HyperParameters::m_replay_alpha,
        HyperParameters::m_replay_beta, HyperParameters::m_replay_beta_decay),

    m_citytile_replay_buffer(
        HyperParameters::m_replay_capacity,
        HyperParameters::m_replay_batch_size, HyperParameters::m_replay_alpha,
        HyperParameters::m_replay_beta, HyperParameters::m_replay_beta_decay),

    m_worker_reward_engine(),
    m_citytile_reward_engine(),

		m_actors(tuple_builder<Actors>::create(
        HyperParameters::m_actor_epsilon_decay,
        HyperParameters::m_actor_epsilon_start,
        HyperParameters::m_actor_epsilon_end,
        HyperParameters::m_nn_atom_count, HyperParameters::m_nn_v_min,
        HyperParameters::m_nn_v_max, HyperParameters::m_nn_step_size,
        HyperParameters::m_nn_gamma, HyperParameters::m_replay_batch_size))

 	{
		m_worker_dqn.to(DeviceType);
		m_citytile_dqn.to(DeviceType);
	}
	
  inline ActionReturn processEpisode(const kit::Agent& _agent, RandomEngine& random_engine_,
		 const std::size_t _frame, const std::size_t _episode) {

		auto &actor0 = std::get<0>(m_actors);
		actor0.processEpisode(_agent, m_worker_dqn, m_citytile_dqn, m_worker_replay_buffer,
													m_citytile_replay_buffer, m_worker_reward_engine,
													m_citytile_reward_engine, random_engine_);
		
		if (_frame > HyperParameters::m_replay_capacity) {
			m_worker_model_learner.train(_frame, m_worker_replay_buffer, random_engine_);
			// citytile_model_trainer.train(frame, citytile_replay_buffer,
			// random_engine_);
		}
		return ActionReturn(
			actor0.getBestWorkerActions(), 
			actor0.getBestCityTileActions());
  }
	
	inline void resetState() {
		std::get<0>(m_actors).resetState();
		m_worker_reward_engine.onNewGame();
	}
	
	inline float getWorkerReward() const {
		return m_worker_reward_engine.getRewardSum();
	}

private:
	WorkerDQN m_worker_dqn;
	CityTileDQN m_citytile_dqn;

  ModelLearner<WorkerDQN, DeviceType,
                 static_cast<std::size_t>(WorkerActions::Count)>
    m_worker_model_learner;
  ModelLearner<CityTileDQN, DeviceType,
                 static_cast<std::size_t>(CityTileActions::Count)>
    m_citytile_model_learner;

  WorkerReplayBuffer m_worker_replay_buffer;
  CityTileReplayBuffer m_citytile_replay_buffer;
  WorkerRewardEngine<DeviceType> m_worker_reward_engine;
  CityTileRewardEngine<DeviceType> m_citytile_reward_engine;
  Actors m_actors;
};

#endif /* TRAINER_HPP_ */
