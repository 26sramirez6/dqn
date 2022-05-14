#ifndef MULTI_STEP_PAWN_MANAGER_HPP_
#define MULTI_STEP_PAWN_MANAGER_HPP_



template <std::size_t ActorId, torch::DeviceType DeviceType,
					typename FeatureBuilder,
          typename RewardEngine, typename ReplayBuf, 
					typename ModelConfig>
class MultiStepPawnManager {
public:
	using SingleStateFeatures = SingleStateFeature<DeviceType, BoardConfig::size, ModelConfig>;
	using FinalBatch = TrainingBatch<DeviceType, BoardConfig::size, ModelConfig>;

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
				m_one_step_prior_pawns(),
        m_latest_pawn_count(0), m_nth_rewards_prior_cursor(0), 
				m_nth_ids_prior_size(0), m_final_batch()

  {}

  inline const FinalBatch& getFinalBatch() const { return m_final_batch; }

	inline std::size_t getLatestPawnCount() const { 	return m_latest_pawn_count; }

  inline std::size_t getQueueSize() const { return m_multi_step_actions.size(); }

	inline const SingleStateFeature& getLatestStateFeature() const { return m_multi_step_features.back(); }
	
	inline void printRewardState() const {
		std::cout << "reward cursor: " << m_nth_rewards_prior_cursor << std::endl;
		int i = 0;
		for (auto& map : m_multi_step_rewards) {
			std::cout << i << ": {";
			for (auto& kv : map) {
				std::cout << kv.first << ": " << kv.second << ", ";
			
			std::cout << "}" << std::endl;
		}
	}	
	
  template <typename Env>
  void updatePawnStates(const Env &_env,
                        RewardEngine &reward_engine_,
                        const torch::Tensor &_latest_actions) {
		pushLatestFeatureState(_env);
		pushLatestRewards(_env, reward_engine_);
		pushLatestActions(_latest_actions);

    if (_env.turn > m_n && m_multi_step_pawn_ids.size() == m_n) {
//      std::vector<int> nth_retained_ids_prior;
//      const auto &nth_ids_prior = m_multi_step_pawn_ids.front();
//      nth_retained_ids_prior.reserve(nth_ids_prior.size());
//
//      m_nth_ids_prior_size = nth_ids_prior.size();
//			if (!m_nth_ids_prior_size) {
//				return;
//			}

      auto retained_accessor = m_retained_id_indices.accessor<int64_t, 1>();
      auto destroyed_accessor = m_destroyed_id_indices.accessor<int64_t, 1>();
      int retained_count = 0;
      int destroyed_count = 0;
			const auto& latest_ids = m_multi_step_pawn_ids.back();
      for (int i = 0; i < nth_ids_prior.size(); ++i) {
        const int id_prior = nth_ids_prior[i];
        if (m_latest_pawn_count > 0 && id_prior == latest_ids[retained_count]) {
          retained_accessor[retained_count++] = i;
          nth_retained_ids_prior.push_back(id_prior);
        } else {
          destroyed_accessor[destroyed_count++] = i;
        }
      }
			m_nth_ids_prior_size = nth_ids_prior.size();
		
      updateFinalBatchFeatureState(nth_ids_prior.size(), retained_count);
      updateFinalBatchNonTerminal(retained_count);
      updateFinalBatchActions(nth_ids_prior.size());
      updateFinalBatchRewards(_env, nth_retained_ids_prior,
                              reward_engine_);

      m_multi_step_pawn_ids.pop();
    }

		m_one_step_prior_pawns = std::move(latest_pawn_map); 
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
		m_one_step_prior_pawns.clear();
		std::queue<std::vector<int>>().swap(m_multi_step_pawn_ids);
		std::queue<torch::Tensor>().swap(m_multi_step_actions);
		std::queue<SingleStateFeatures>().swap(m_multi_step_features);
		m_latest_pawn_count = 0;
		m_nth_rewards_prior_cursor = 0;
		m_nth_ids_prior_size = 0;
	}

private:

  void inline pushLatestActions(const torch::Tensor &_latest_actions) {
		std::cout << "pushing action: " << _latest_actions.index({0}).item().to<float>() << std::endl;
    m_multi_step_actions.push(_latest_actions.clone());
  }

  template <typename Env>
  void inline pushLatestFeatureState(const Env &_env) {
    SingleStateFeatures feature_state(m_batch_size);
    FeatureBuilder::template setStateFeatures(_env, feature_state);
    m_multi_step_features.push(std::move(feature_state));
  }

  template <typename Env>
  void inline pushLatestRewards(const Env &_env,
																const auto& _latest_pawn_map,
                                RewardEngine &reward_engine_) {
		std::unordered_map<int, float> reward_map;
    reward_engine_.template computeRewards<ActorId>(
			_env, _latest_pawn_map, m_one_step_prior_pawns, reward_map);

		std::cout << "pushing rewards: ";
		for(auto& kv : reward_map) {
			std::cout << kv.first << ": " << kv.second << std::endl; 
		}
		std::cout << std::endl;

    m_multi_step_rewards[m_nth_rewards_prior_cursor] = std::move(reward_map);
    m_nth_rewards_prior_cursor = (m_nth_rewards_prior_cursor + 1) % m_n;
  }

  void inline updateFinalBatchNonTerminal(const std::size_t _up_to_retained) {
    auto up_to_retained = torch::indexing::Slice(0, _up_to_retained, 1);
    m_final_batch.m_is_non_terminal.index_put_({torch::indexing::None}, false);
    m_final_batch.m_is_non_terminal.index_put_(
        {m_retained_id_indices.index({up_to_retained})}, true);
  }

  
  void inline updateFinalBatchFeatureState(
      const std::size_t _nth_pawn_count_prior,
      const std::size_t _retained_pawn_count) {
    auto up_to_retained = torch::indexing::Slice(0, _retained_pawn_count, 1);
    const auto up_to_prior_pawn_count =
        torch::indexing::Slice(0, _nth_pawn_count_prior, 1);
    const auto &nth_features_prior = m_multi_step_features.front();

    m_final_batch.m_state.m_geometric.index_put_(
        {up_to_prior_pawn_count},
        nth_features_prior.m_geometric.index({up_to_prior_pawn_count}));

		const auto& latest_features = m_multi_step_features.back();
 
    m_final_batch.m_next_state.m_geometric.index_put_(
        {m_retained_id_indices.index({up_to_retained})},
        latest_features.m_geometric.index({up_to_retained}));

    // pop at end to avoid invalidating the reference
		// only pop when queue is 1 size greater
		// i.e. S1 -> S5 with m_n==4 will require a queue holding 5 states
		if (m_multi_step_features.size() == m_n + 1) {
    	m_multi_step_features.pop();
		}
  }

  void inline updateFinalBatchActions(const std::size_t _nth_pawn_count_prior) {
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

  template <typename Env>
  void updateFinalBatchRewards(const Env &_env,
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
	
	std::unordered_map<int, typename PawnType::type> m_one_step_prior_pawns;

	std::size_t m_latest_pawn_count;
  std::size_t m_nth_rewards_prior_cursor;
	std::size_t m_nth_ids_prior_size;

  FinalBatch m_final_batch;
};

