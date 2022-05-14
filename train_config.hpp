/*
 * train_config.hpp
 *
 *  Created on: May 20, 2021
 *      Author: saul.ramirez
 */

#ifndef TRAIN_CONFIG_HPP_
#define TRAIN_CONFIG_HPP_
#include <cstdint>
#include <torch/torch.h>
#ifndef USE_CUDA
#define DEVICE torch::kCPU
#else
#define DEVICE torch::kCUDA
#endif

struct TrainConfig {
  static constexpr uint64_t train_seed = 0;
  static constexpr unsigned chunk_size = 100;
  static constexpr unsigned game_iterations = 10000;
  static constexpr torch::DeviceType device = DEVICE;
};

#endif /* TRAIN_CONFIG_HPP_ */
