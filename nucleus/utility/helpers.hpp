#pragma once

#include "helpers_api.hpp"

template<typename T>
void mem_release(std::vector<T>& v) {
	std::vector<T>().swap(v);
}

template<typename T>
size_t mem_bytes(const std::vector<T>& v) {
	return sizeof(T)* v.capacity();
}



template <typename T>
void reserve_ex(std::vector<T>& v, const size_t& size) {
	try {
		v.reserve(size);
	} catch (std::bad_alloc& e) {
		std::throw_with_nested(std::runtime_error("Could n't allocate "
							  + std::to_string(size*sizeof(T))
							  + " Bytes: "
							  + e.what()));
	}
}

template <typename T>
void resize_ex(std::vector<T>& v, const size_t& size) {
	try {
		v.resize(size);
	} catch (std::bad_alloc& e) {
		std::throw_with_nested(std::runtime_error("Could n't allocate "
							  + std::to_string(size*sizeof(T))
							  + " Bytes: "
							  + e.what()));
	}
}

template <typename T>
void resize_ex(std::vector<T>& v, const size_t& size, const T& value) {
	try {
		v.resize(size, value);
	} catch (std::bad_alloc& e) {
		std::throw_with_nested(std::runtime_error("Could n't allocate "
							  + std::to_string(size*sizeof(T))
							  + " Bytes: "
							  + e.what()));
	}
}


template <typename T>
std::string hr_cap(T val, int precision /*= 2*/) {
	std::array<std::string, 5> millnames = {"", " K", " M", " G", " T"};
	auto millid = std::max(0U,
			       std::min(static_cast<unsigned>(millnames.size())-1U, val==0 ? 0U : static_cast<unsigned>(std::log10(   std::abs(static_cast<double>(val))  )/3.) )
				);

	std::ostringstream oss;
	oss << std::fixed << std::setprecision(precision) << static_cast<T>(val/std::pow(10., 3*millid)) << millnames[millid];

	return oss.str();
}

template <class ExecutionPolicy, typename T>
auto nonzero(ExecutionPolicy&& policy, const std::vector<T>& v) {
	std::vector<T> t(v);
	t.erase(std::remove(policy, t.begin(), t.end(), 0), t.end());
	return t;
}


template <class ExecutionPolicy, typename T, typename U>
auto nonzero(ExecutionPolicy&& policy,
	     const std::vector<T>& v,
	     const std::vector<U> _pred) {
	auto sz = std::min(v.size(), _pred.size());
	std::vector<T> t(v.cbegin(), v.cbegin()+sz);
	auto p ( _pred.cbegin() );
	auto b ( t.data() );
	t.erase(std::remove_if(policy,
			t.begin(), t.end(),
			[&p, &b](T& val) {return *(p+(&val-b))==0;}),
			t.end());
	return t;
}



template <class ExecutionPolicy, typename T, typename U, typename L /*= MTYPE*/>
auto multi_arrange_ep(ExecutionPolicy&& policy, const std::vector<T>& start, const std::vector<U>& count) {

	ASERT(start.size()==count.size(), "Start and Count sizes should match.");

	std::pair<std::vector<L>, std::vector<T>> tp;
	auto& [incr, neighbors] = tp;

	// Strip zero values from start and count vectors
	auto ct = nonzero(policy, count);

	if (ct.empty()) { // Nothing to do
		return tp;
	}

	auto st = nonzero(policy, start, count);

	neighbors.resize(count.size()+1, 0);
	std::inclusive_scan(policy, count.cbegin(), count.cend(),
			    neighbors.begin()+1,
			    std::plus<>());

	// building reset indexes
	// neighbors[:-1] at nonzero positions are reset indexes
	auto ri = nonzero(policy, neighbors, count);

	// building incremental indices
	// This vector require to be signed type to handle negative numbers properly
	auto arr_len(std::reduce(policy, ct.cbegin(), ct.cend()));
	resize_ex(incr, arr_len, static_cast<L>(1));

	incr[ri[0]] = st[0];

	//size_t i = 1;
	std::for_each(policy, ri.cbegin()+1, ri.cend(),
				  [&incr = incr,
				   &bg = std::as_const(ri[0]),
				   &st,
				   &ct](const auto& r) {

		auto i = static_cast<size_t>(std::distance(&bg, &r));
		incr[r] = st[i] + static_cast<L>(1) -
				static_cast<L>(st[i-1]+ct[i-1]);
	});

	std::inclusive_scan(policy, incr.cbegin(), incr.cend(),
			    incr.begin(),
			    std::plus<>());

	return tp;
}

template <class ExecutionPolicy, typename T, typename U, typename L /*= MTYPE*/>
auto multi_arange_omp(ExecutionPolicy&& policy, const std::vector<T>& start, const std::vector<U>& count) {

	ASERT(start.size()==count.size(), "Start and Count sizes should match.");

	std::pair<std::vector<L>, std::vector<T>> tp;
	auto& [inc, neighbors] = tp;
	//  to be able to use structure binding problem inside omp section
	auto& incr = inc;

	// Strip zero values from start and count vectors
	auto ct = nonzero(policy, count);

	if (ct.empty()) { // Nothing to do
		return tp;
	}

	auto st = nonzero(policy, start, count);

	neighbors.resize(count.size()+1, 0);
	std::inclusive_scan(policy, count.cbegin(), count.cend(),
						neighbors.begin()+1,
						std::plus<>());

	// building reset indexes
	// neighbors[:-1] at nonzero positions are reset indexes
	auto ri = nonzero(policy, neighbors, count);

	// building incremental indices
	// This vector requires to be signed type to handle negative numbers properly
	auto arr_len(std::reduce(policy, ct.cbegin(), ct.cend()));
	resize_ex(incr, arr_len, static_cast<L>(1));

	incr[ri[0]] = st[0];

	#pragma omp parallel for default(none) shared(incr, ri, st, ct)
	for (size_t i = 1; i < ri.size(); i++) {
		incr[ri[i]] = st[i] + static_cast<L>(1) -
				static_cast<L>(st[i-1]+ct[i-1]);
	}

	std::inclusive_scan(policy, incr.cbegin(), incr.cend(),
						incr.begin(),
						std::plus<>());

	return tp;
}

template <typename T, typename U>
auto take(const std::vector<T>& vec,
	  const std::vector<U>& idx) {
	std::vector<T> dst(idx.size());
	#pragma omp parallel for default(none) shared(dst, vec, idx)
	for(size_t i = 0; i < idx.size(); i++) {
		dst[i] = vec[idx[i]];
	}
	return dst;
}

template <typename T, typename U, size_t SIZE>
auto take_par(const std::vector<T>& vec,
	  const std::vector<std::array<U, SIZE>>& idx,
	      size_t first, size_t last) {
	auto sz = last - first;
	std::array<std::vector<T>, SIZE> dst;
	for(size_t s = 0; s < SIZE; s++) {
		dst[s].resize(sz);
	}
	#pragma omp parallel for default (none) shared (dst, vec, idx, first, sz)
	for(size_t i = 0; i < sz; i++) {
		for(size_t s = 0; s < SIZE; s++) {
			dst[s][i] = vec[idx[i+first][s]];
		}
	}
	return dst;
}

template <typename T, typename U, size_t SIZE>
auto take_seq(const std::vector<T>& vec,
	  const std::vector<std::array<U, SIZE>>& idx,
	      size_t first, size_t last) {
	auto sz = last - first;
	std::array<std::vector<T>, SIZE> dst;
	for(size_t s = 0; s < SIZE; s++) {
		dst[s].resize(sz);
	}
	for(size_t i = 0; i < sz; i++) {
		for(size_t s = 0; s < SIZE; s++) {
			dst[s][i] = vec[idx[i+first][s]];
		}
	}
	return dst;
}

template <typename T, typename U, typename L, size_t SIZE>
auto dtake_par(const std::vector<T>& vec1,
	   const std::vector<U>& vec2,
	   const std::vector<std::array<L, SIZE>>& dst,
	   size_t first, size_t last) {

	auto sz = last - first;
	std::array<std::pair<std::vector<T>, std::vector<U>>, SIZE> tp;
	for(size_t s = 0; s < SIZE; s++) {
		tp[s].first.resize(sz);
		tp[s].second.resize(sz);
	}

	#pragma omp parallel for default (none) shared( tp, dst, vec1, vec2, sz, first)
	for(size_t i = 0; i < sz; i++) {
		for(size_t s = 0; s < SIZE; s++) {
			tp[s].first[i] = vec1[dst[i+first][s]];
			tp[s].second[i] = vec2[dst[i+first][s]];
		}
	}

	return tp;
}

template <typename T, typename U, typename L, size_t SIZE>
auto dtake_seq(const std::vector<T>& vec1,
	   const std::vector<U>& vec2,
	   const std::vector<std::array<L, SIZE>>& dst,
	   size_t first, size_t last) {

	auto sz = last - first;
	std::array<std::pair<std::vector<T>, std::vector<U>>, SIZE> tp;
	for(size_t s = 0; s < SIZE; s++) {
		tp[s].first.resize(sz);
		tp[s].second.resize(sz);
	}

	for(size_t i = 0; i < sz; i++) {
		for(size_t s = 0; s < SIZE; s++) {
			tp[s].first[i] = vec1[dst[i+first][s]];
			tp[s].second[i] = vec2[dst[i+first][s]];
		}
	}

	return tp;
}


template <typename V, size_t SIZE>
void cpy(const std::vector<std::array<V, SIZE>>& s,
	  std::vector<V>& d, size_t start /*= 0*/) {
		for(size_t i = 0; i < s.size(); i++) {
			for (size_t j = 0; j < SIZE; j++) {
				d[SIZE*i+j+start] = s[i][j];
			}
		}
}

template <typename V, size_t SIZE>
void cpy(const std::vector<V>& s,
		std::vector<std::array<V, SIZE>>& d) {
		for(size_t i = 0; i < s.size(); i++) {
			d[i/SIZE][i%SIZE] = s[i];
		}
}


template <class ExecutionPolicy, typename T, size_t SIZE>
void sort(ExecutionPolicy&& policy, std::vector<std::array<T, SIZE>>& vec, size_t depth /*= SIZE*/) {
	ASERT(depth <= SIZE, "Sorting depth should be lower or equal to SIZE!");
	std::sort(policy, vec.begin(), vec.end(),
		  [&](const auto& t1, const auto& t2)->bool {
			for(size_t s = 0; s < depth; s++) {
				if(t1[s] != t2[s]) {
					return t1[s] < t2[s];
				}
			}
			return false; // sort requires strict weak ordering
		  });
}

template <class ExecutionPolicy, typename T, size_t SIZE>
void sort2d(ExecutionPolicy&& policy, std::vector<std::array<T, SIZE>>& vec, size_t depth /*= SIZE*/) {
	ASERT(depth <= SIZE, "Sorting depth should be lower or equal to SIZE!");

	// First sort inner dimension	
	for(auto& v: vec) {
			for(size_t s = 0; s < depth; s++) {
				std::sort(policy, &v[0], &v[0]+depth);
			}
	}

	// Then sort external dimension
	std::sort(policy, vec.begin(), vec.end(),
		  [&](const auto& t1, const auto& t2)->bool {
			for(size_t s = 0; s < depth; s++) {
				if(t1[s] != t2[s]) {
					return t1[s] < t2[s];
				}
			}
			return false; // sort requires strict weak ordering
		  });
}


template <class ExecutionPolicy, typename RandomIt>
auto sort_permutation(ExecutionPolicy&& policy, RandomIt cbegin, RandomIt cend, bool ascending /*= true*/) {
	auto len = std::distance(cbegin, cend);
	typedef typename std::iterator_traits<RandomIt>::value_type value_type;
	std::vector<value_type> perm(len);
	std::iota(perm.begin(), perm.end(), 0U);
	std::sort(policy, perm.begin(), perm.end(),
		  [&](const size_t& a, const size_t& b)
		  {return (ascending ? (*(cbegin+a) < *(cbegin+b)) : (*(cbegin+a) > *(cbegin+b)));});
	return perm;
}

template <class ExecutionPolicy, typename RandomIt1, typename RandomIt2>
auto sort_permutation(ExecutionPolicy&& policy,
		      RandomIt1 cbegin1, RandomIt1 cend1,
		      RandomIt2 cbegin2) {
	auto len = std::distance(cbegin1, cend1);
	typedef typename std::iterator_traits<RandomIt1>::value_type value_type;
	std::vector<value_type> perm(len);
	std::iota(perm.begin(), perm.end(), 0U);
	std::sort(policy, perm.begin(), perm.end(),
		  [&](const size_t& a, const size_t& b)
		  {return *(cbegin1+a) < *(cbegin1+b) ||
				(  *(cbegin1+a) == *(cbegin1+b) && *(cbegin2+a) < *(cbegin2+b)  );
		   });
	return perm;
}

template <typename RandomIt>
auto unique_indices(RandomIt cbegin, RandomIt cend) {
	auto len = std::distance(cbegin, cend);
	std::vector<size_t> id(len);
	std::iota(id.begin(), id.end(), 0U);
	auto it = std::unique(id.begin(), id.end(),
			      [&](const size_t& a, const size_t& b)
			      {return *(cbegin+a) == *(cbegin+b);});
	id.resize(std::distance(id.begin(), it));
	return id;
}

template <typename RandomIt1, typename RandomIt2>
auto unique_indices(RandomIt1 cbegin1, RandomIt1 cend1,
		    RandomIt2 cbegin2) {
	std::vector<size_t> id(std::distance(cbegin1, cend1));
	std::iota(id.begin(), id.end(), 0U);
	auto it = std::unique(id.begin(), id.end(),
			      [&](const size_t& a, const size_t& b)
			      {return *(cbegin1+a) == *(cbegin1+b) && *(cbegin2+a) == *(cbegin2+b) ;});
	id.resize(std::distance(id.begin(), it));
	return id;
}

template<typename RandomIt, typename OutputIt>
void extract_column(RandomIt cbegin, RandomIt cend, OutputIt output, size_t col) {
	for(auto it = cbegin; it!=cend; it++, output++) {
		*output = (*it)[col];
	}
}
