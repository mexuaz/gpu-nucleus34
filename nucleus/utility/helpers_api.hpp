#pragma once

#include <string> // memcpy
#include <tuple>
#include <array>
#include <vector>
#include <algorithm> // remove, remove_if, sort
#include <numeric> // inclusive_scan(..), partial_sum(..), reduce(..)
#include <iomanip> // setprecision(..)
#include <omp.h>

#include "def_types.hpp"
#include "defs.hpp"

// TODO: remove mem_release and mem_bytes

/**
 * @brief mem_release Free up the memory occupied by vector
 * @tparam T Type of vector elements
 * @param v Input vector
 */
template<typename T>
void mem_release(std::vector<T>& v);

/**
 * @brief mem_bytes Calculate memory space occupied by data content of vector
 * @tparam T Type of vector elements
 * @param v Input vector
 * @return memory in bytes occupied by data content of vector
 */
template<typename T>
size_t mem_bytes(const std::vector<T>& v);

/**
 * @brief reserve_ex memory allocation for vectors with
 * message showing the required memory in case of error
 * @param v
 * @param size
 */
template <typename T>
void reserve_ex(std::vector<T>& v, const size_t& size);

template <typename T>
void resize_ex(std::vector<T>& v, const size_t& size);

template <typename T>
void resize_ex(std::vector<T>& v, const size_t& size, const T& value);

/**
 * @brief hr_cap change the given value to human readable format for capacity
 * @param val
 * @param precision
 * @return
 */
template <typename T>
std::string hr_cap(T val, int precision = 2);

std::string hr_cap(const std::string& val, int precision = 2);


/**
 * @brief nonzero Return non-zero elements of vector v
 */
template <class ExecutionPolicy, typename T>
auto nonzero(ExecutionPolicy&& policy, const std::vector<T>& v);


/**
 * @brief nonzero Filter elements of vector v based on nonzero elements of vector _pred
 */
template <class ExecutionPolicy, typename T, typename U>
auto nonzero(ExecutionPolicy&& policy,
	     const std::vector<T>& v,
	     const std::vector<U> _pred);



template <class ExecutionPolicy, typename T, typename U, typename L = MTYPE>
auto multi_arrange_ep(ExecutionPolicy&& policy, const std::vector<T>& start, const std::vector<U>& count);

template <class ExecutionPolicy, typename T, typename U, typename L = MTYPE>
auto multi_arange_omp(ExecutionPolicy&& policy, const std::vector<T>& start, const std::vector<U>& count);

/**
 * @brief take
 * @param vec
 * @param idx
 * @return vec[idx]
 */
template <typename T, typename U>
auto take(const std::vector<T>& vec,
	  const std::vector<U>& idx);

template <typename T, typename U, size_t SIZE>
auto take_par(const std::vector<T>& vec,
	  const std::vector<std::array<U, SIZE>>& idx,
	      size_t first, size_t last);

template <typename T, typename U, size_t SIZE>
auto take_seq(const std::vector<T>& vec,
	  const std::vector<std::array<U, SIZE>>& idx,
	      size_t first, size_t last);

/**
 * @brief dtake retrieve [first, last) of vec1 and vec2 of those indices in array of vectors in dst
 * @param vec1
 * @param vec2
 * @param dst array with length of SIZE of vectors used as indices
 * @param first
 * @param last not included in the range
 */
template <typename T, typename U, typename L, size_t SIZE>
auto dtake_par(const std::vector<T>& vec1,
	   const std::vector<U>& vec2,
	   const std::vector<std::array<L, SIZE>>& dst,
	   size_t first, size_t last);

template <typename T, typename U, typename L, size_t SIZE>
auto dtake_seq(const std::vector<T>& vec1,
	   const std::vector<U>& vec2,
	   const std::vector<std::array<L, SIZE>>& dst,
	   size_t first, size_t last);


template <typename V, size_t SIZE>
void cpy(const std::vector<std::array<V, SIZE>>& s,
	  std::vector<V>& d, size_t start = 0);

template <typename V, size_t SIZE>
void cpy(const std::vector<V>& s,
		std::vector<std::array<V, SIZE>>& d);


/**
 * @brief sort sort the given input vector of array
 * @tparam T datatype of array
 * @tparam SIZE size of array
 * @param vec Input vector of array
 * @param depth Depth of the array we want to sort
 */
template <class ExecutionPolicy, typename T, size_t SIZE>
void sort(ExecutionPolicy&& policy, std::vector<std::array<T, SIZE>>& vec, size_t depth = SIZE);

/**
 * @brief Sort first innder dimension (across depth of array) and then 
 * across the vector
 * refer to test units for some examples
 * 
 * @tparam ExecutionPolicy 
 * @tparam T 
 * @tparam SIZE 
 * @param policy 
 * @param vec 
 * @param depth 
 */
template <class ExecutionPolicy, typename T, size_t SIZE>
void sort2d(ExecutionPolicy&& policy, std::vector<std::array<T, SIZE>>& vec, size_t depth = SIZE);


/**
 * @brief sort_permutation produce sorted indices of the given vector
 * @param policy
 * @param cbegin Iterator to the beginning of the input vector
 * @param cend Iterator to the position after last element of the input vector
 */
template <class ExecutionPolicy, typename RandomIt>
auto sort_permutation(ExecutionPolicy&& policy, RandomIt cbegin, RandomIt cend, bool ascending = true);

/**
 * @brief sort_permutation Produces indices relevant to paired elements of
 * two vectors which maps them to sorted vectors.
 * The second vector's length should be equal or bigger than the first vector
 * @param policy
 * @param cbegin1 Iterator to the beginning of the first vector
 * @param cend1 Iterator to the position after last element of first vector
 * @param cbegin2 Iterator to the beginning of the second vector
 */
template <class ExecutionPolicy, typename RandomIt1, typename RandomIt2>
auto sort_permutation(ExecutionPolicy&& policy,
		      RandomIt1 cbegin1, RandomIt1 cend1,
		      RandomIt2 cbegin2);



/**
 * @brief unique_indices produce indices of unique elements of a sorted vector
 * @param cbegin Iterator to the begining of the first vector
 * @param cend Iterator to the position after the last element of the vector
 */
template <typename RandomIt>
auto unique_indices(RandomIt cbegin, RandomIt cend);

/**
 * @brief unique_indices Produces indices relevant to paired elements of
 * two stable sorted vectors where their. The second vector's length should be equal
 * or bigger than the first vector
 * @param cbegin1 Iterator to the beginning of the first vector
 * @param cend1 Iterator to the position after last element of first vector
 * @param cbegin2 Iterator to the beginning of the second vector
 */
template <typename RandomIt1, typename RandomIt2>
auto unique_indices(RandomIt1 cbegin1, RandomIt1 cend1,
		    RandomIt2 cbegin2);

/**
 * @brief extract_column extract a specific column
 * @param cbegin input iterator begin
 * @param cend input iterator end
 * @param output Output iterator
 * @param col column to extract
 */
template<typename RandomIt, typename OutputIt>
void extract_column(RandomIt cbegin, RandomIt cend, OutputIt output, size_t col);
