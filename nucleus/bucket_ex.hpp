#ifndef __BUCKET_EX__H
#define __BUCKET_EX__H
#include "graph/graph.hpp"

#include "utility/def_system.hpp"

template <typename T, typename L>
class bucket_ex
{
	T min_id = 0;
	L last_popped_value = 0;

	std::vector<L> values;	// original values
	std::vector<T> perm;	// actual perm, size num of elements
	std::vector<T> perm_ex; // perm of perm
	std::vector<T> offset;	// start points of values, size (max-value)

	// Stable snapshot of the current peeling batch used by peeling_par().
	// in_batch[k] is non-zero while key k is being peeled in the current
	// level. Unlike the live `values`, it does not change when other keys are
	// transiently decreased down to the current level, so it gives a reliable
	// ownership tie-break between concurrent peelers.
	std::vector<char> in_batch;

public:
	/**
	 * @brief constructor Given the initial values vector, construct the
	 * bucket datastructures
	 * @param vals initial support vector datatype (usually from degree vector of graph)
	 */
	bucket_ex(const std::vector<T> &vals)
	{
		if (vals.empty())
		{
			return;
		}
		const auto max_value = *std::max_element(vals.cbegin(), vals.cend());
		std::vector<T> cnt(max_value + 1);
		for (size_t i = 0; i < vals.size(); i++)
		{
			const auto &v = vals[i];
			auto &e = cnt[v];
			__sync_fetch_and_add(&e, 1);
		}

		offset.resize(cnt.size());
		std::inclusive_scan(EXE_POL, cnt.cbegin(), cnt.cend(), offset.begin());

		perm = sort_permutation(EXE_POL, vals.cbegin(), vals.cend());
		perm_ex = sort_permutation(EXE_POL, perm.cbegin(), perm.cend());

		// Copy, Typecast and replace zeros with no_val
		values.resize(vals.size());
		std::replace_copy(vals.begin(), vals.end(), values.begin(), 0, -1);

		// Move min_id to the first non-zero value
		for (; min_id < perm.size(); min_id++)
		{
			if (vals[perm[min_id]])
			{
				break;
			}
		}
	}
	~bucket_ex() = default;

	/**
	 * @brief copy constructor
	 */
	bucket_ex(const bucket_ex &) = delete;

	/**
	 * @brief  assignment operator
	 */
	bucket_ex &operator=(const bucket_ex &) = delete;

	const L no_val = -1;

	template <typename U, typename V>
	friend std::ostream &operator<<(std::ostream &out,
									const bucket_ex<U, V> &bkt);

	/**
	 * @brief support Adds marker to the values vector and return that as
	 * support vector.
	 * Since we deduct one in order to keep track of peeled elements, the default
	 * is one.
	 * @param marker The value that needs to be added to the values vector to
	 * get the correct support vector
	 * @return support vector
	 */
	std::vector<L> support(const L &marker = 1) const
	{
		std::vector<L> sup(values.size());
		std::transform(values.cbegin(), values.cend(), sup.begin(),
					   [&](const auto &x)
					   { return x + marker; });
		return sup;
	}

	/**
	 * @brief mem_bytes return the approximate amount of the money that
	 * variable occupies
	 * @return bytes of memory occupied
	 */
	[[nodiscard]] size_t bytes() const
	{
		size_t m = mem_bytes(values) +
				   mem_bytes(perm) +
				   mem_bytes(perm_ex) +
				   mem_bytes(offset) +
				   sizeof(min_id) +
				   sizeof(last_popped_value);
		return m;
	}

	/**
	 * @brief release Free up memory occupied by internal vectors
	 * We don't release the values as they are support value results
	 */
	[[maybe_unused]] void release()
	{
		mem_release(perm);
		mem_release(perm_ex);
		mem_release(offset);
	}

	/**
	 * @brief decrease Decrease the value of element for given key
	 * @param key
	 */
	void decrease(const size_t &key)
	{
		auto &val = values[key];
		const auto &pos = perm_ex[key];
		auto &off = offset[val - 1];
		auto &v1 = perm[pos];
		auto &v2 = perm[off];
		auto &w1 = perm_ex[v1];
		auto &w2 = perm_ex[v2];

		std::swap(v1, v2);
		std::swap(w1, w2);
		off++;
		val--;
	}

	template <typename Iter>
	void peel(Iter FIterKey1, Iter FIterKey2)
	{
		for (auto it = FIterKey1; it != FIterKey2; it++)
		{
			if (values[*it] < last_popped_value)
			{
				return; // peeled before
			}
		}
		for (Iter it = FIterKey1; it != FIterKey2; it++)
		{
			if (values[*it] != last_popped_value)
				decrease(*it);
		}
	}

	// Parallel counting pass for a single clique on behalf of source vertex id.
	// Exactly one source vertex from the current batch owns a still-active
	// clique (the smallest id among current-batch members). The owner records
	// decrement requests for non-batch members only.
	template <typename Iter>
	void peel_count(const size_t &id,
					 Iter FIterKey1,
					 Iter FIterKey2,
					 std::vector<int> &dec_req,
					 std::vector<T> &local_touched)
	{
		for (auto it = FIterKey1; it != FIterKey2; it++)
		{
			if (values[*it] < last_popped_value)
			{
				return; // clique already destroyed in an earlier level
			}
		}
		for (auto it = FIterKey1; it != FIterKey2; it++)
		{
			if (in_batch[*it] && static_cast<size_t>(*it) < id)
			{
				return; // a smaller-id batch member owns this clique
			}
		}

		for (auto it = FIterKey1; it != FIterKey2; it++)
		{
			const auto key = *it;
			if (!in_batch[key])
			{
				if (__sync_fetch_and_add(&dec_req[key], 1) == 0)
				{
					local_touched.push_back(key);
				}
			}
		}
	}

	/**
	 * @brief PopMin pops the minimum value and relevant id from bucket
	 * @param id
	 * @param val
	 * @return false if bucket is empty
	 */
	bool pop_min(T &key)
	{
		if (min_id < perm.size())
		{
			// const std::lock_guard<std::mutex> lock(mutex);
			key = perm[min_id++];
			last_popped_value = values[key];
			return true;
		}
		return false; // if the bucket is empty
	}

	bool pop_min(std::vector<T> &keys)
	{
		if (min_id >= perm.size())
		{
			return false; // The bucket is empty
		}

		last_popped_value = values[perm[min_id]];
		keys.resize(offset[last_popped_value] - min_id);
		for (size_t i = 0; i < keys.size(); i++, min_id++)
		{
			keys[i] = perm[min_id];
		}
		return true;
	}

	/**
	 * @brief peeling sequential peeling approach
	 * @tparam E Input graph offset data type
	 * @tparam V Input graph vertices data type
	 * @tparam DIM Input graph Dimension
	 * @param graph input undirected graph
	 * @return last last_popped_value value (maximum support)
	 */
	template <typename E, typename V, size_t DIM>
	auto peeling_seq(const graph_t<E, V, DIM> &graph)
	{
		V id;
		while (pop_min(id))
		{
			const auto &off = graph.O[id];
			const auto &deg = graph.D[id];
			for (size_t d = 0; d < deg; d++)
			{
				const auto it_b = graph.E[off + d].cbegin();
				peel(it_b, it_b + DIM);
			} // for all neighbors of id
			// We deduct to keep track of peeled key/values
			// please note the condition in peel method
			values[id]--;
		} // bucket pop loop
		return last_popped_value;
	}

	/**
	 * @brief peeling multi-threaded peeling approach
	 * @tparam E Input graph offset data type
	 * @tparam V Input graph vertices data type
	 * @tparam DIM Input graph Dimension
	 * @param graph input undirected graph
	 * @return last last_popped_value value (maximum support)
	 */
	template <typename E, typename V, size_t DIM>
	auto peeling_par(const graph_t<E, V, DIM> &graph)
	{
		in_batch.assign(values.size(), 0);
		std::vector<int> dec_req(values.size(), 0);
		std::vector<V> ids;
		while (pop_min(ids))
		{
			// For small batches, OpenMP setup and atomic accounting cost more
			// than useful parallel work. Use the direct sequential peel path.
			if (ids.size() < 256)
			{
				for (const auto &id : ids)
				{
					const auto &off = graph.O[id];
					const auto &deg = graph.D[id];
					for (size_t d = 0; d < deg; d++)
					{
						const auto it_b = graph.E[off + d].cbegin();
						peel(it_b, it_b + DIM);
					}
					values[id]--;
				}
				continue;
			}

			// Publish the stable membership snapshot for this level before any
			// value gets modified, so peel_safe() can resolve clique ownership
			// without being fooled by members transiently reaching the level.
			for (const auto &id : ids)
			{
				in_batch[id] = 1;
			}

			std::vector<T> touched;
#pragma omp parallel default(none) shared(ids, graph, dec_req, touched)
			{
				std::vector<T> local_touched;
#pragma omp for schedule(static)
				for (size_t i = 0; i < ids.size(); i++)
				{
					const auto id = ids[i];
					const auto &off = graph.O[id];
					const auto &deg = graph.D[id];
					for (size_t d = 0; d < deg; d++)
					{
						const auto it_b = graph.E[off + d].cbegin();
						peel_count(id, it_b, it_b + DIM, dec_req, local_touched);
					}
				}
#pragma omp critical
				touched.insert(touched.end(),
							   local_touched.cbegin(),
							   local_touched.cend());
			}

			// Apply requested decrements once per touched key. We clamp by
			// current level so values never drop below last_popped_value.
			for (const auto &key : touched)
			{
				auto &req = dec_req[key];
				if (!req)
				{
					continue;
				}
				const auto allowed = values[key] - last_popped_value;
				if (allowed > 0)
				{
					const auto times = std::min(static_cast<size_t>(req),
											   static_cast<size_t>(allowed));
					for (size_t k = 0; k < times; k++)
					{
						decrease(key);
					}
				}
				req = 0;
			}

			// Mark the peeled ids only after decrement application.
			for (const auto &id : ids)
			{
				values[id]--;
				in_batch[id] = 0;
			}
		} // bucket pop loop
		return last_popped_value;
	}
};

/**
 * @brief operator << overloads printing for bucket datastructures
 * print the bucket value1 (key11 key12 ...), value 2 (key21 key22 ...)
 * @tparam T bucket datatype
 * @param out
 * @param db
 * @return
 */
template <typename T, typename L>
std::ostream &operator<<(std::ostream &out, const bucket_ex<T, L> &bkt)
{

	T st = 0;
	for (size_t i = 0; i < bkt.offset.size(); i++)
	{
		const auto &ed = bkt.offset[i];
		out << i << ":(";
		for (auto j = st; j < ed; j++)
		{
			const auto &k = bkt.perm[j];
			out << k;
			if (j != ed - 1)
			{
				out << ' ';
			}
		}
		out << ')';
		if (i < bkt.offset.size() - 1)
		{
			out << ',';
		}
	}
	return out;
}

#endif // __BUCKET_EX__H
