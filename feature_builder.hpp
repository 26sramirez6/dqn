#ifndef FEATURE_BUILDER_HPP
#define FEATURE_BUILDER_HPP

#include "board_config.hpp"
#include "math_util.hpp"
#include "model_config.hpp"
#include <torch/torch.h>


struct FeatureBuilder {

  template <std::size_t size>
  static inline const torch::Tensor &getSpatialTensor() {
    static torch::Tensor spatial(construct_spatial(size));
    return spatial;
  }

	template<typename Env>
  static void setStateFeatures(const Env &_env, SingleStateFeatures &ftrs_) {
				
  }
};



#endif /* FEATURE_BUILDER_HPP */
