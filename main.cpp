/*
 * main.cpp
 *
 *  Created on: May 21, 2021
 *      Author: saul.ramirez
 */

#include <Eigen/Dense>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <torch/torch.h>
#include <tuple>
#include <vector>

#include "actor.hpp"
#include "board_config.hpp"
#include "dqn.hpp"
#include "feature_builder.hpp"
#include "hyper_parameters.hpp"
#include "model_config.hpp"
#include "random_engine.hpp"
#include "replay_buffer.hpp"
#include "ship.hpp"
#include "train_config.hpp"
#include "trainer.hpp"

int main() {
  auto &random_engine =
      RandomEngine<float, TrainConfig::train_seed>::getInstance();
  HyperParameters hyper_parameters;
  auto device = TrainConfig::device;
  Trainer<TrainConfig::game_iterations, TrainConfig::chunk_size,
          BoardConfig::player_count, TrainConfig::device,
          decltype(random_engine)>::train(hyper_parameters, random_engine);
}
