/*
 * math_util.hpp
 *
 *  Created on: May 22, 2021
 *      Author: saul.ramirez
 */

#ifndef MATH_UTIL_HPP_
#define MATH_UTIL_HPP_

#include <Eigen/Dense>
#include <algorithm>
#include <torch/torch.h>

static inline int manhattan(const int _x1, const int _y1, const int _x2,
                            const int _y2) {
  return std::abs(_x2 - _x1) + std::abs(_y2 - _y1);
}

static inline torch::Tensor manhattan(const torch::Tensor &_spatial,
                                      const torch::Tensor &_positions) {
  auto diff = _spatial - _positions;
  return diff.abs().sum(1);
}

template <int WrapAround>
static inline int manhattan(const int _x1, const int _y1, const int _x2,
                            const int _y2) {
  return std::min(std::abs((_x2 - _x1) % WrapAround),
                  std::abs((_x1 - _x2) % WrapAround)) +
         std::min(std::abs((_y2 - _y1) % WrapAround),
                  std::abs((_y1 - _y2) % WrapAround));
}

template <int WrapAround>
static inline torch::Tensor manhattan(const torch::Tensor &_spatial,
                                      const torch::Tensor &_positions) {
  auto diff = _spatial - _positions;
  return diff.remainder(WrapAround).min((-diff).remainder(WrapAround)).sum(1);
}

static inline torch::Tensor construct_spatial(const int _size) {
  torch::Tensor ret(torch::zeros({_size * _size, 2}));
  auto a = ret.accessor<float, 2>();
  for (int i = 0; i < _size * _size; i++) {
    a[i][0] = i % _size;
    a[i][1] = std::floor(i / _size);
  }

  return ret;
}

template <typename Scalar>
static inline void min_max_norm(torch::Tensor tensor_,
                                const Scalar _fill_value_if_no_diff,
                                const bool _zero_to_one) {
  const auto max = tensor_.max();
  const auto min = tensor_.min();
  const auto diff = max - min;
  if (!diff.is_nonzero()) {
    tensor_.fill_(_fill_value_if_no_diff);
    return;
  }
  tensor_.sub_(min);
  tensor_.div_(diff);
  if (!_zero_to_one) {
    tensor_.mul_(2);
    tensor_.sub_(1);
  }
}

static inline int rand_in_range(int lower, int upper) {
  return (rand() % (upper - lower + 1)) + lower;
}

static inline int
searchSortedRight(const Eigen::Ref<const Eigen::ArrayXf> &_sorted,
                  const float _value) {
  auto result =
      std::lower_bound(_sorted.data(), _sorted.data() + _sorted.size(), _value);
  return std::distance(_sorted.data(), result);
}

static inline void
searchSortedRight(const Eigen::Ref<const Eigen::ArrayXf> &_sorted,
                  const Eigen::Ref<const Eigen::ArrayXf> &_values,
                  Eigen::Ref<Eigen::ArrayXi> &indices_) {
  for (int i = 0; i < _values.size(); ++i) {
    indices_(i) = searchSortedRight(_sorted, _values(i));
  }
}

template <typename Derived1, typename Derived2>
static inline void cumsum(const Derived1 &_a, Derived2 &cumsum_) {
  cumsum_(0) = _a(0);
  for (int i = 1; i < _a.size(); ++i) {
    cumsum_(i) = _a(i) + cumsum_(i - 1);
  }
}

template <typename Derived> static inline void cumsum(Derived &cumsum_) {
  for (int i = 1; i < cumsum_.size(); ++i) {
    cumsum_(i) = cumsum_(i) + cumsum_(i - 1);
  }
}

template <typename RandomEngine>
static inline void choice(RandomEngine &random_engine_,
                          const Eigen::Ref<const Eigen::ArrayXf> &_p,
                          Eigen::Ref<Eigen::ArrayXf> cumsum_,
                          Eigen::Ref<Eigen::ArrayXi> indices_) {
  cumsum(_p, cumsum_);
  cumsum_(cumsum_.size() - 1) = 1.;
  const auto draws = random_engine_.uniform(indices_.size());
  searchSortedRight(cumsum_, draws, indices_);
}

template <typename RandomEngine>
static inline void choice(RandomEngine &random_engine_,
                          Eigen::Ref<Eigen::ArrayXf> _p,
                          Eigen::Ref<Eigen::ArrayXi> indices_) {
  cumsum(_p);
  const auto draws = random_engine_.uniform(indices_.size());
  searchSortedRight(_p, draws, indices_);
}

template <typename RandomEngine>
static inline void choice(RandomEngine &random_engine_,
                          const std::size_t _sample_space_size,
                          Eigen::Ref<Eigen::ArrayXi> indices_) {
  auto p = Eigen::ArrayXf::LinSpaced(_sample_space_size + 1, 0, 1);
  const auto draws = random_engine_.uniform(indices_.size());
  searchSortedRight(p, draws, indices_);
  indices_ -= 1;
}

template <typename T> static inline T clip(T value, T lower, T upper) {
  return value < lower ? lower : value > upper ? upper : value;
}

template <typename ScalarType, typename Derived>
static inline void eigen_to_tensor(const Derived &_eigen,
                                   torch::Tensor &tensor_) {
  std::memcpy(tensor_.data_ptr(), _eigen.data(),
              sizeof(ScalarType) * _eigen.size());
}

template <typename ScalarType, typename Derived>
static inline void tensor_to_eigen(const torch::Tensor &_tensor,
                                   Derived &eigen_) {
  std::memcpy(eigen_.data(), _tensor.data_ptr(),
              sizeof(ScalarType) * _tensor.size(0));
}

static inline void choice(const std::size_t _sample_space_size,
                          torch::Tensor &p_, torch::Tensor &draws_,
                          torch::Tensor &indices_) {
  torch::linspace_out(p_, 0, 1, _sample_space_size + 1);
  torch::rand_out(draws_, {indices_.size(0)});
  torch::searchsorted_out(indices_, p_, draws_, true, true);
  // torch::bucketize_out(indices_, p_, draws_, true, true);
  indices_.sub_(1);
}
#endif /* MATH_UTIL_HPP_ */
