/*
 * random_engine.hpp
 *
 *  Created on: May 20, 2021
 *      Author: saul.ramirez
 */

#ifndef RANDOM_ENGINE_HPP_
#define RANDOM_ENGINE_HPP_

#include <Eigen/Dense>
#include <chrono>
#include <cstdint>
#include <random>
#include <type_traits>

template <typename T, uint64_t SpecifiedSeed = 0> class RandomEngine {
  using ArrayT = Eigen::Array<T, Eigen::Dynamic, 1>;

public:
  static RandomEngine &getInstance() {
    static RandomEngine instance(SpecifiedSeed);
    return instance;
  }

  inline T uniform() { return m_uniform(m_rng); }

  inline ArrayT uniform(const unsigned _rows) {
    return ArrayT::NullaryExpr(_rows, 1, [&]() { return m_uniform(m_rng); });
  }

  inline std::mt19937 &getGenerator() { return m_rng; }

private:
  RandomEngine(uint64_t seed)
      : m_seed(seed == 0 ? std::chrono::high_resolution_clock::now()
                               .time_since_epoch()
                               .count()
                         : seed),
        m_uniform(0, 1) {
    m_rng.seed(seed);
  }

  RandomEngine(RandomEngine const &) = delete;
  void operator=(RandomEngine const &) = delete;

  std::mt19937 m_rng;
  uint64_t m_seed;
  std::uniform_real_distribution<T> m_uniform;
};

#endif /* RANDOM_ENGINE_HPP_ */
