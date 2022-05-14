/*
 * template_util.hpp
 *
 *  Created on: May 22, 2021
 *      Author: saul.ramirez
 */

#ifndef TEMPLATE_UTIL_HPP_
#define TEMPLATE_UTIL_HPP_

template <class T> using remove_cv_t = typename std::remove_cv<T>::type;

template <class T>
using remove_reference_t = typename std::remove_reference<T>::type;

template <typename T> using remove_cv_ref = remove_cv_t<remove_reference_t<T>>;

template <typename T, typename U>
using is_equiv = std::is_same<T, remove_cv_ref<U>>;

// the following breaks for visual c++ compilers (tested on version VS15)
#ifndef _MSC_VER
namespace detail {
	template <bool...> struct bool_pack;
	template <bool... bs>
	using all_true = std::is_same<bool_pack<bs..., true>, bool_pack<true, bs...>>;
} // namespace detail
template <typename... Ts> using all_true = detail::all_true<Ts::value...>;

template <typename TTuple, typename UTuple> struct are_equiv;

template <typename... Ts, typename... Us>
struct are_equiv<std::tuple<Ts...>, std::tuple<Us...>>
    : all_true<is_equiv<Ts, Us>...> {};
#endif

template <typename Collection> struct remove_references {
  using type = remove_reference_t<Collection>;
};

template <template <typename...> class Collection, typename... Types>
struct remove_references<Collection<Types...> &> {
  using type = Collection<typename std::remove_reference<Types>::type...>;
};

template <template <typename...> class Collection, typename... Types>
struct remove_references<Collection<Types...>> {
  using type = Collection<typename std::remove_reference<Types>::type...>;
};

template <typename T>
using remove_references_t = typename remove_references<T>::type;

template <typename T> struct tuple_builder {};

template <typename... T> struct tuple_builder<std::tuple<T...>> {
  template <typename... Args>
  static inline std::tuple<T...> create(Args &&... args) {
    return std::forward_as_tuple(T(args...)...);
  }
};

template <typename... input_t>
using tuple_cat_t = decltype(std::tuple_cat(std::declval<input_t>()...));

template <std::size_t I, typename T> struct tuple_n {
  template <typename... Args>
  using type = typename tuple_n<I - 1, T>::template type<T, Args...>;
};

template <typename T> struct tuple_n<0, T> {
  template <typename... Args> using type = std::tuple<Args...>;
};

template <std::size_t I, typename T>
using tuple_n_t = typename tuple_n<I, T>::template type<>;

template <unsigned int N, typename T> struct tuple_n_t_obj {
  typedef decltype(std::tuple_cat(
      std::tuple<T>(), typename tuple_n_t_obj<N - 1, T>::type())) type;
};

template <typename T> struct tuple_n_t_obj<0, T> {
  typedef decltype(std::tuple<>()) type;
};

template <class Functor, class Tuple, std::size_t I = 0, typename... FuncArgs>
inline typename std::enable_if<I == std::tuple_size<Tuple>::value, void>::type
tuple_for_each_type(FuncArgs &&... func_args) {}

template <class Functor, class Tuple, std::size_t I = 0, typename... FuncArgs>
    inline
    typename std::enable_if < I<std::tuple_size<Tuple>::value, void>::type
                              tuple_for_each_type(FuncArgs &&... func_args) {
  Functor::template calc<I, typename std::tuple_element<I, Tuple>::type>(
      std::forward(func_args)...);
  tuple_for_each_type<Functor, Tuple, I + 1>(std::forward(func_args)...);
}

template <std::size_t I = 0, typename FuncT, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type
tuple_for_each_obj(std::tuple<Tp...> &, FuncT) {}

template <std::size_t I = 0, typename FuncT, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), void>::type tuple_for_each_obj(std::tuple<Tp...> &t,
                                                    FuncT f) {
  f(std::get<I>(t));
  tuple_for_each_obj<I + 1, FuncT, Tp...>(t, f);
}

#endif /* TEMPLATE_UTIL_HPP_ */
