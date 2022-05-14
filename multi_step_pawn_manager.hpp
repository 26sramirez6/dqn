#ifndef MULTISTEP_MANAGER_HPP_
#define MULTISTEP_MANAGER_HPP_

#include <chrono>
#include <cmath>
#include <random>
#include <type_traits>

#include "board.hpp"
#include "feature_builder.hpp"
#include "model_config.hpp"
#include "replay_buffer.hpp"



template <std::size_t ActorId, torch::DeviceType DeviceType,
          typename FeatureBuilder, typename RewardEngine, typename ReplayBuf, 
					typename ModelConfig>
class MultiStepPawnManager {
public:
	using SingleStateFeatures = SingleStateFeature<DeviceType, BoardConfig, ModelConfig>;
	using FinalTrainingBatch = TrainingBatch<DeviceType, BoardConfig, ModelConfig>;
  MultiStepPawnManager(const std::size_t _multi_step_n, const float _gamma,
                       const std::size_t _batch_size)
      : m_n(_multi_step_n), m_gamma(_gamma), m_batch_size(_batch_size),
        m_retained_id_indices(torch::zeros(
            {BoardConfig::size * BoardConfig::size}, torch::dtype(torch::kInt64)
                                                         .requires_grad(false)
                                                         .device(torch::kCPU))),
        m_destroyed_id_indices(torch::zeros(
            {BoardConfig::size * BoardConfig::size}, torch::dtype(torch::kInt64)
                                                         .requires_grad(false)
                                                         .device(torch::kCPU))),
        m_multi_step_pawn_ids(), m_multi_step_actions(),
        m_multi_step_features(), m_multi_step_rewards(_multi_step_n),
        m_latest_pawn_count(0), m_nth_rewards_prior_cursor(0), 
				m_nth_ids_prior_size(0), m_final_batch()

  {}

  inline const FinalTrainingBatch& getFinalTrainingBatch() const { return m_final_batch; }

	inline std::size_t getLatestPawnCount() const { 	return m_latest_pawn_count; }

  inline std::size_t getQueueSize() const { return m_multi_step_actions.size(); }

	inline const SingleStateFeatures& getLatestStateFeatures() const { 
		return m_multi_step_features.back(); 
	}
	
	inline void printRewardState() const {
		std::cout << "reward cursor: " << m_nth_rewards_prior_cursor << std::endl;
		int i = 0;
		for (auto& map : m_multi_step_rewards) {
			std::cout << i << ": {";
			for (auto& kv : map) {
				std::cout << kv.first << ": " << kv.second << ", ";
			}
			std::cout << "}" << std::endl;
		}
	}	
	
  template <typename FeatureBuilder, typename Board>
  void updatePawnStates(const Board &_current_board,
                        RewardEngine &reward_engine_,
                        const torch::Tensor &_latest_actions) {
    // TODO: make generic for any pawn
    const auto &pawn_map = _current_board.getShipMap();
		m_latest_pawn_count = pawn_map.size(); // don't rely on multi_step_pawn_ids
		pushLatestFeatureState<FeatureBuilder>(_current_board);

		if (_latest_actions.size(0) > 0) {
			pushLatestPawns(pawn_map);
			pushLatestRewards(_current_board, reward_engine_);
			pushLatestActions(_latest_actions);
		}
    if (_current_board.m_step > m_n && m_multi_step_pawn_ids.size() == m_n) {
      std::vector<int> nth_retained_ids_prior;
      const auto &nth_ids_prior = m_multi_step_pawn_ids.front();
      nth_retained_ids_prior.reserve(nth_ids_prior.size());

      m_nth_ids_prior_size = nth_ids_prior.size();
			if (!m_nth_ids_prior_size) {
				return;
			}

      auto retained_accessor = m_retained_id_indices.accessor<int64_t, 1>();
      auto destroyed_accessor = m_destroyed_id_indices.accessor<int64_t, 1>();
      int retained_count = 0;
      int destroyed_count = 0;
			const auto& latest_ids = m_multi_step_pawn_ids.back();
      for (int i = 0; i < nth_ids_prior.size(); ++i) {
        const int id_prior = nth_ids_prior[i];
        if (id_prior == latest_ids[retained_count]) {
          retained_accessor[retained_count++] = i;
          nth_retained_ids_prior.push_back(id_prior);
        } else {
          destroyed_accessor[destroyed_count++] = i;
        }
      }
			m_nth_ids_prior_size = nth_ids_prior.size();
		
      updateFinalTrainingBatchFeatureState<FeatureBuilder>(nth_ids_prior.size(),
                                                   retained_count);
      updateFinalTrainingBatchNonTerminal(retained_count);
      updateFinalTrainingBatchActions(nth_ids_prior.size());
      updateFinalTrainingBatchRewards(_current_board, nth_retained_ids_prior,
                              reward_engine_);

      m_multi_step_pawn_ids.pop();
    }
  }

	inline bool pushTransitions(ReplayBuf& replay_buf_) const {
		if (m_multi_step_pawn_ids.size() == m_n-1) {
			replay_buf_.push(m_final_batch, m_nth_ids_prior_size);
			return true;
		}

		return false;
	}

	void inline resetState() {
		m_retained_id_indices.zero_();
		m_destroyed_id_indices.zero_();

		std::queue<std::vector<int>>().swap(m_multi_step_pawn_ids);
		std::queue<torch::Tensor>().swap(m_multi_step_actions);
		std::queue<SingleStateFeatures>().swap(m_multi_step_features);
		m_latest_pawn_count = 0;
		m_nth_rewards_prior_cursor = 0;
		m_nth_ids_prior_size = 0;
	}

private:
	template<typename PawnMap>
	void inline pushLatestPawns(const PawnMap& _pawn_map) {
    std::vector<int> latest_ids;
    latest_ids.reserve(_pawn_map.size());
    std::transform(_pawn_map.begin(), _pawn_map.end(),
                   std::back_inserter(latest_ids),
                   [](const auto &pair) { return pair.first; });

    m_multi_step_pawn_ids.push(std::move(latest_ids));
	}	

  void inline pushLatestActions(const torch::Tensor &_latest_actions) {
		std::cout << "pushing action: " << _latest_actions.index({0}).item().to<float>() << std::endl;
    m_multi_step_actions.push(_latest_actions.clone());
  }

  template <typename FeatureBuilder, typename Board>
  void inline pushLatestFeatureState(const Board &_current_board) {
    SingleStateFeatures feature_state(m_batch_size);
    FeatureBuilder::template setStateFeatures<BoardConfig>(_current_board,
                                                           feature_state);
    m_multi_step_features.push(std::move(feature_state));
  }

  template <typename Board>
  void inline pushLatestRewards(const Board &_current_board,
                                RewardEngine &reward_engine_) {
		std::unordered_map<int, float> reward_map;
    reward_engine_.template computeRewards<ActorId>(_current_board,
                                                    reward_map);
		std::cout << "pushing rewards: ";
		for(auto& kv : reward_map) {
			std::cout << kv.first << ", " << kv.second << std::endl; 
		}
		std::cout << std::endl;

    m_multi_step_rewards[m_nth_rewards_prior_cursor] = std::move(reward_map);
    m_nth_rewards_prior_cursor = (m_nth_rewards_prior_cursor + 1) % m_n;
  }

  void inline updateFinalTrainingBatchNonTerminal(const std::size_t _up_to_retained) {
    auto up_to_retained = torch::indexing::Slice(0, _up_to_retained, 1);
    m_final_batch.m_is_non_terminal.index_put_({torch::indexing::None}, false);
    m_final_batch.m_is_non_terminal.index_put_(
        {m_retained_id_indices.index({up_to_retained})}, true);
  }

  template <typename FeatureBuilder>
  void inline updateFinalTrainingBatchFeatureState(
      const std::size_t _nth_pawn_count_prior,
      const std::size_t _retained_pawn_count) {
    auto up_to_retained = torch::indexing::Slice(0, _retained_pawn_count, 1);
    const auto up_to_prior_pawn_count =
        torch::indexing::Slice(0, _nth_pawn_count_prior, 1);
    const auto &nth_features_prior = m_multi_step_features.front();

    m_final_batch.m_state.m_geometric.index_put_(
        {up_to_prior_pawn_count},
        nth_features_prior.m_geometric.index({up_to_prior_pawn_count}));

    m_final_batch.m_next_state.m_geometric.index_put_(
        {m_retained_id_indices.index({up_to_retained})},
        m_multi_step_features.back().m_geometric.index({up_to_retained}));

    // pop at end to avoid invalidating the reference
		// only pop when queue is 1 size greater
		// i.e. S1 -> S5 with m_n==4 will require a queue holding 5 states
		if (m_multi_step_features.size() == m_n + 1) {
    	m_multi_step_features.pop();
		}
  }

  void inline updateFinalTrainingBatchActions(const std::size_t _nth_pawn_count_prior) {
    // the latest actions have been pushed by this point
    const auto &nth_actions_prior = m_multi_step_actions.front();
    const auto up_to_prior_pawn_count =
        torch::indexing::Slice(0, _nth_pawn_count_prior, 1);
    // nth_actions_prior won't be on gpu, so must use to(Device) before
    // index_put
	
    m_final_batch.m_action.index_put_(
        {up_to_prior_pawn_count},
        nth_actions_prior.index({up_to_prior_pawn_count}).to(DeviceType));
    m_multi_step_actions.pop();
  }

  template <typename Board>
  void updateFinalTrainingBatchRewards(const Board &_current_board,
                               const std::vector<int> &_nth_retained_ids_prior,
                               RewardEngine &reward_engine_) {
		// copy ctor called here
    auto cumulative_rewards_map = m_multi_step_rewards[m_nth_rewards_prior_cursor];
		
		std::cout << "cum rewards before: ";
		for (auto& kv : cumulative_rewards_map) {
			std::cout << kv.first << ": " << kv.second <<", ";
		}		
		std::cout << std::endl;


		for (auto id : _nth_retained_ids_prior) {
			if (cumulative_rewards_map.find(id) == cumulative_rewards_map.end()) { // doesn't contain
				cumulative_rewards_map.insert({id, 0});
			}
		}		

    for (int i = 1; i < m_n; ++i) {
      const int index = (i + m_nth_rewards_prior_cursor) % m_n;
      const auto &next_reward_map = m_multi_step_rewards[index];
      const float gamma = std::pow(m_gamma, i);
      for (auto &kv : cumulative_rewards_map) {
        if (next_reward_map.find(kv.first) !=
            next_reward_map.end()) { // contains
          kv.second += gamma*next_reward_map.at(kv.first);
        }
      }
    }	

		std::cout << "cum rewards after: ";
		for (auto& kv : cumulative_rewards_map) {
			std::cout << kv.first << ": " << kv.second <<", ";
		}		
		std::cout << std::endl;

    auto retained_accessor = m_retained_id_indices.accessor<int64_t, 1>();

		std::cout << "nth retained ids prior: ";
		for (auto i : _nth_retained_ids_prior) std::cout << i << ",";
		std::cout << std::endl;
    for (int i = 0; i < _nth_retained_ids_prior.size(); ++i) {
      m_final_batch.m_reward.index_put_(
          {i}, cumulative_rewards_map.at(_nth_retained_ids_prior[i]));
    }
  }

private:
  const std::size_t m_n;
  const float m_gamma;
  const std::size_t m_batch_size;


  torch::Tensor m_retained_id_indices;
  torch::Tensor m_destroyed_id_indices;

  std::queue<std::vector<int>> m_multi_step_pawn_ids;
  std::queue<torch::Tensor> m_multi_step_actions;
  std::queue<SingleStateFeatures> m_multi_step_features;
  std::vector<std::unordered_map<int, float>> m_multi_step_rewards;

	std::size_t m_latest_pawn_count;
  std::size_t m_nth_rewards_prior_cursor;
	std::size_t m_nth_ids_prior_size;

  FinalTrainingBatch m_final_batch;
};

