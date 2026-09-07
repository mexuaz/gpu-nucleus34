#ifndef CLIQUES_HPP_
#define CLIQUES_HPP_

#include "graph/graph.hpp"
#include <iostream>
#include <memory>
#include <numeric>

template <typename U, typename V, size_t DIM>
auto cliques_tri_seq(const graph_t<U, V, DIM> &graph)
{

	const auto &O = graph.O;
	const auto &D = graph.D;
	const auto &E = graph.E;

	/**
	 * OE = O[E], DE = D[E]
	 * @brief dt Tuple containing both OE and DE
	 */
	auto dt = dtake_par(O, D, E, 0, graph.size_edges());
	const auto &OE = dt[0].first;
	const auto &DE = dt[0].second;

	auto tp = multi_arange_omp(EXE_POL, OE, DE);

	/**
	 * @brief M: Generated indices
	 * @brief N: Offset of generated indices
	 */
	auto &[M, N] = tp;

	std::vector<std::array<V, DIM + 2>> tris;

	if (M.empty())
	{ // No triangles here
		return tris;
	}

	reserve_ex(tris, M.size()); // Very long array

	auto do_intersections = [&N = N,
							 &M = M,
							 &E,
							 &tris](const U &d1, const U &d2)
	{
		// loop E[i1:i2]
		for (auto d = d1; d < d2; d++)
		{
			auto i1 = d1;
			auto j1 = N[d];
			const auto &j2 = N[d + 1];

			// loop EM[j1:j2]
			// intersection(E[d1..d..d2], EM[N[d]..j1..N[d+1])
			// find indices of intersection
			while (i1 < d2 && j1 < j2)
			{
				const auto &p = E[i1][0];
				const auto &o = static_cast<V>(M[j1]);
				const auto &q = E[o][0];
				if (p < q)
				{
					i1++;
				}
				else
				{
					if (!(q < p))
					{
						/**
						 * @note S[i1], p, S[o] are
						 *  triangle vertices, but we
						 *  store edges ids that have
						 *  those vertices
						 */
						tris.push_back({i1, d, o});
						i1++;
					}
					j1++;
				}
			} // end of intersection loop
		} // end of for-loop
	};

	// Extend neighbors list to include ranges
	for (size_t i = 0; i < O.size() - 1; i++)
	{
		do_intersections(O[i], O[i + 1]);
	}

	// Do it for last range which is not included in nei vector
	do_intersections(O.back(), graph.size_edges());
	tris.shrink_to_fit();

	return tris;
}

template <typename U, typename V, size_t DIM>
auto cliques_tri_par(const graph_t<U, V, DIM> &graph)
{

	//// @todo TODO: investigate weather to create a copy for local thread or to share between threads
	const auto &O = graph.O;
	const auto &D = graph.D;
	const auto &E = graph.E;

	if (!graph.split_parts.size())
	{
		throw std::runtime_error("split_parts vector not initialized for parallel calls.");
	}

	std::vector<std::array<V, DIM + 2>> final_tris;

	// Explicitly disable dynamic teams
	omp_set_dynamic(0);

	// Use n threads for all consecutive parallel regions
	omp_set_num_threads(graph.split_parts.size());

#pragma omp parallel default(none) shared(graph, O, D, E, final_tris, EXE_POL)
	{

		auto th_id = static_cast<size_t>(omp_get_thread_num());
		auto bg = graph.split_parts[th_id];
		auto ed = graph.O.size();
		auto dst_bg = O[bg];
		auto dst_ed = graph.size_edges();
		if (th_id < graph.split_parts.size() - 1)
		{
			ed = graph.split_parts[th_id + 1];
			dst_ed = O[ed];
		}

		/**
		 * OE = O[E], DE = D[E]
		 * @brief dt Tuple containing both OE and DE
		 */
		auto dt = dtake_par(O, D, E, dst_bg, dst_ed);
		const auto &OE = dt[0].first;
		const auto &DE = dt[0].second;

		auto tp = multi_arange_omp(EXE_POL, OE, DE);

		/**
		 * @brief M: Generated indices
		 * @brief N: Offset of generated indices
		 */
		auto &[M, N] = tp;

		if (M.empty())
		{			 // No triangle in this portion
			bg = ed; // end the block without doing the intersection
		}

		decltype(final_tris) tris;
		reserve_ex(tris, M.size());

		auto do_intersections = [&N = N,
								 &M = M,
								 &E,
								 &tris](const U &d1,
										const U &d2,
										size_t neiOffset)
		{
			// dst[d1:d2]
			for (U j = d1; j < d2; j++)
			{
				auto i1 = d1;
				auto j1 = N[j - neiOffset];
				auto j2 = N[j + 1 - neiOffset];

				// intersection(dst[i1:d2], dn[j1:j2])
				while (i1 < d2 && j1 < j2)
				{
					auto &p = E[i1][0];
					const auto &o = static_cast<V>(M[j1]);
					auto &q = E[o][0];
					ASERT(M.size() > j1, "Out of bound index!");
					if (p < q)
					{
						i1++;
					}
					else
					{
						if (!(q < p))
						{
							// S[i1], p, S[o] are triangle vertices
							// But we store edges ids that have these vertices
							// following three edges are required to build triangles
							// however we need only first two to build the graph
							// Note: inorder to find only triangles j is not required
							tris.push_back({i1, j, o});
							i1++;
						}
						j1++;
					}
				} // end of intersection loop

			} // end of for-loop
		};

		for (size_t i = bg; i < ed; i++)
		{
			do_intersections(O[i],
							 (i < O.size() - 1) ? O[i + 1] : graph.size_edges(),
							 dst_bg);
		}

		// tris.shrink_to_fit(); // Not necessary
		size_t last;
#pragma omp critical
		{
			last = final_tris.size();
			if (!last)
			{
				// On the first thread try to estimate and reserve all the required memory
				final_tris.reserve(tris.size() * graph.split_parts.size());
			}
			final_tris.resize(last + tris.size());
		}
		std::copy(tris.cbegin(), tris.cend(), final_tris.begin() + last);
		// We do this instead of inserting in locked mode
	} // end of parallel block

	return final_tris;
}

template <typename U, typename V, size_t DIM>
auto cliques_tri_par_parts(const graph_t<U, V, DIM> &graph)
{

	//// @todo TODO: investigate weather to create a copy for local thread or to share between threads
	const auto &O = graph.O;
	const auto &D = graph.D;
	const auto &E = graph.E;

	if (!graph.split_parts.size())
	{
		throw std::runtime_error("split_parts vector not initialized for parallel calls.");
	}

	std::vector<std::vector<std::array<V, DIM + 2>>> parts_tris(graph.split_parts.size());

	// Explicitly disable dynamic teams
	omp_set_dynamic(0);

	// Use n threads for all consecutive parallel regions
	omp_set_num_threads(graph.split_parts.size());

#pragma omp parallel default(none) shared(graph, O, D, E, parts_tris, EXE_POL)
	{

		auto th_id = static_cast<size_t>(omp_get_thread_num());
		auto bg = graph.split_parts[th_id];
		auto ed = graph.O.size();
		auto dst_bg = O[bg];
		auto dst_ed = graph.size_edges();
		if (th_id < graph.split_parts.size() - 1)
		{
			ed = graph.split_parts[th_id + 1];
			dst_ed = O[ed];
		}

		/**
		 * OE = O[E], DE = D[E]
		 * @brief dt Tuple containing both OE and DE
		 */
		auto dt = dtake_par(O, D, E, dst_bg, dst_ed);
		const auto &OE = dt[0].first;
		const auto &DE = dt[0].second;

		auto tp = multi_arange_omp(EXE_POL, OE, DE);

		/**
		 * @brief M: Generated indices
		 * @brief N: Offset of generated indices
		 */
		auto &[M, N] = tp;

		if (M.empty())
		{			 // No triangle in this portion
			bg = ed; // end the block without doing the intersection
		}

		auto &tris = parts_tris[th_id];
		reserve_ex(tris, M.size());

		auto do_intersections = [&N = N,
								 &M = M,
								 &E,
								 &tris](const U &d1,
										const U &d2,
										size_t neiOffset)
		{
			// dst[d1:d2]
			for (U j = d1; j < d2; j++)
			{
				auto i1 = d1;
				auto j1 = N[j - neiOffset];
				auto j2 = N[j + 1 - neiOffset];

				// intersection(dst[i1:d2], dn[j1:j2])
				while (i1 < d2 && j1 < j2)
				{
					auto &p = E[i1][0];
					const auto &o = static_cast<V>(M[j1]);
					auto &q = E[o][0];
					ASERT(M.size() > j1, "Out of bound index!");
					if (p < q)
					{
						i1++;
					}
					else
					{
						if (!(q < p))
						{
							// S[i1], p, S[o] are triangle vertices
							// But we store edges ids that have these vertices
							// following three edges are required to build triangles
							// however we need only first two to build the graph
							// Note: inorder to find only triangles j is not required
							tris.push_back({i1, j, o});
							i1++;
						}
						j1++;
					}
				} // end of intersection loop

			} // end of for-loop
		};

		for (size_t i = bg; i < ed; i++)
		{
			do_intersections(O[i],
							 (i < O.size() - 1) ? O[i + 1] : graph.size_edges(),
							 dst_bg);
		}
	} // end of parallel block

	return parts_tris;
}

/**
 * cliques_tri_par_for for-loop (per vertex) version of cliques_tri_par
 * @tparam U
 * @tparam V
 * @tparam DIM
 * @param graph
 * @return
 */
template <typename U, typename V, size_t DIM>
auto cliques_tri_par_for(const graph_t<U, V, DIM> &graph)
{

	//// @todo TODO: investigate weather to create a copy for local thread or to share between threads
	const auto &O = graph.O;
	const auto &D = graph.D;
	const auto &E = graph.E;

	if (!graph.split_parts.size())
	{
		throw std::runtime_error("split_parts vector not initialized for parallel calls.");
	}

	std::vector<std::array<V, DIM + 2>> final_tris;

#pragma omp parallel for default(none) shared(graph, O, D, E, final_tris)
	for (size_t v = 0; v < O.size(); v++)
	{

		auto dst_bg = O[v];
		auto dst_ed = graph.size_edges();
		if (v < O.size() - 1)
		{
			dst_ed = O[v + 1];
		}

		/**
		 * OE = O[E], DE = D[E]
		 * @brief dt Tuple containing both OE and DE
		 */
		auto dt = dtake_par(O, D, E, dst_bg, dst_ed);
		const auto &OE = dt[0].first;
		const auto &DE = dt[0].second;

		auto tp = multi_arange_omp(EXE_POL, OE, DE);

		/**
		 * @brief M: Generated indices
		 * @brief N: Offset of generated indices
		 */
		auto &[M, N] = tp;

		decltype(final_tris) tris;
		reserve_ex(tris, M.size());

		auto do_intersections = [&N = N,
								 &M = M,
								 &E,
								 &tris](const U &d1, const U &d2, size_t neiOffset)
		{
			// dst[d1:d2]
			for (U j = d1; j < d2; j++)
			{
				auto i1 = d1;
				auto j1 = N[j - neiOffset];
				auto j2 = N[j + 1 - neiOffset];

				// intersection(dst[i1:d2], dn[j1:j2])
				while (i1 < d2 && j1 < j2)
				{
					auto &p = E[i1][0];
					const auto &o = static_cast<V>(M[j1]);
					auto &q = E[o][0];
					ASERT(M.size() > j1, "Out of bound index!");
					if (p < q)
					{
						i1++;
					}
					else
					{
						if (!(q < p))
						{
							// S[i1], p, S[o] are triangle vertices
							// But we store edges ids that have these vertices
							// following three edges are required to build triangles
							// however we need only first two to build the graph
							// Note: inorder to find only triangles j is not required
							tris.push_back({i1, j, o});
							i1++;
						}
						j1++;
					}
				} // end of intersection loop

			} // end of for-loop
		};

		if (!M.empty())
		{ // No triangle in this portion
			do_intersections(dst_bg, dst_ed, dst_bg);
		}

		// tris.shrink_to_fit(); // Not necessary
		size_t last;
#pragma omp critical
		{
			last = final_tris.size();
			if (!last)
			{
				// On the first thread try to estimate and reserve all the required memory
				final_tris.reserve(tris.size() * graph.split_parts.size());
			}
			final_tris.resize(last + tris.size());
		}
		std::copy(tris.cbegin(), tris.cend(), final_tris.begin() + last);
		// We do this instead of inserting in locked mode
	} // end of parallel block

	return final_tris;
}

template <typename U, typename V, size_t SIZE>
auto cliques_qud_seq(const std::vector<U> &O,
					 const std::vector<V> &D,
					 const std::vector<std::array<V, SIZE>> &E)
{

	// nds = O[E], dds = D[E]
	// OE_(b,c) = O[E], DE_(b,c) = D[E]
	auto dt = dtake_seq(O, D, E, 0, E.size());
	const auto &OE_b = dt[0].first;
	const auto &DE_b = dt[0].second;
	const auto &OE_c = dt[1].first;
	const auto &DE_c = dt[1].second;

	/*!
	 * \brief M: Generated indices
	 * \brief N: Offset of generated indices
	 */

	auto tp0 = multi_arrange_ep(EXE_POL, OE_b, DE_b);
	auto &[M0, N0] = tp0;

	auto tp1 = multi_arrange_ep(EXE_POL, OE_c, DE_c);
	auto &[M1, N1] = tp1;

	std::vector<std::array<V, SIZE + 2>> cliques;

	if (M0.empty())
	{ // No cliques here
		return cliques;
	}

	reserve_ex(cliques, M0.size());

	auto do_intersections = [
									&M0 = M0,
									&M1 = M1,
									&N0 = N0,
									&N1 = N1,
									&E,
									&cliques](const U &d1, const U &d2)
	{
		// E[d1:d2]
		for (U j = d1; j < d2; j++)
		{
			auto i1 = d1;
			auto i2 = d2;
			auto j1 = N0[j];
			auto j2 = N0[j + 1];

			// intersection(E[i1:i2], EM[j1:j2])
			while (i1 < i2 && j1 < j2)
			{
				ASERT(E.size() > i1, "Out of boundary E[i1][0]!");
				auto &p = E[i1][0];
				ASERT(M0.size() > j1, "Out of boundary M0[j1]!");
				const auto &o1 = static_cast<V>(M0[j1]);
				ASERT(E.size() > o1, "Out of boundary E[o1][0]!");
				auto &q = E[o1][0];

				if (p < q)
				{
					i1++;
				}
				else
				{
					if (!(q < p))
					{
						auto &k1 = N1[i1];
						auto &k2 = N1[i1 + 1];
						auto k = k1;
						for (; k < k2; k++)
						{
							const auto &o2 = static_cast<V>(M1[k]);
							if (E[j][1] == E[o2][1])
							{
								cliques.push_back({i1, o1, j, o2});
								i1++;
								break;
							}
						}
						ASERT(k < k2, "Could n't find 4th element!");
					}
					j1++;
				}
			} // end of intersection loop
		} // end of for-loop
	};

	for (size_t i = 0; i < O.size() - 1; i++)
	{
		do_intersections(O[i], O[i + 1]);
	}

	// Do it for last range for edges which is not included in O vector
	do_intersections(O.back(), E.size());

	cliques.shrink_to_fit();
	return cliques;
}

template <typename U, typename V, size_t DIM>
auto cliques_qud_seq(const graph_t<U, V, DIM> &graph)
{
	return cliques_qud_seq<U, V, DIM>(graph.O, graph.D, graph.E);
}

template <typename U, typename V, size_t SIZE>
auto cliques_qud_par(const graph_t<U, V, SIZE> &graph)
{

	const auto &O = graph.O;
	const auto &D = graph.D;
	const auto &E = graph.E;

	if (!graph.split_parts.size())
	{
		throw std::runtime_error("split_parts vector not initialized for parallel calls.");
	}

	std::vector<std::array<V, SIZE + 2>> final_cliques;

	// Explicitly disable dynamic teams
	omp_set_dynamic(0);

	// Use n threads for all consecutive parallel regions
	omp_set_num_threads(graph.split_parts.size());

#pragma omp parallel default(none) shared(graph, O, D, E, final_cliques, EXE_POL)
	{
		auto th_id = static_cast<size_t>(omp_get_thread_num());
		auto bg = graph.split_parts[th_id];
		auto ed = graph.O.size();
		auto dst_bg = O[bg];
		auto dst_ed = graph.size_edges();
		if (th_id < graph.split_parts.size() - 1)
		{
			ed = graph.split_parts[th_id + 1];
			dst_ed = O[ed];
		}

		// nds = O[E], dds = D[E]
		auto dt = dtake_par(O, D, E, dst_bg, dst_ed);
		const auto &nds0 = dt[0].first;
		const auto &dds0 = dt[0].second;
		const auto &nds1 = dt[1].first;
		const auto &dds1 = dt[1].second;

		/**
		 * @brief M: Generated indices
		 * @brief N: Offset of generated indices
		 */

		auto tp0 = multi_arange_omp(EXE_POL, nds0, dds0);
		auto &[M0, N0] = tp0;

		auto tp1 = multi_arange_omp(EXE_POL, nds1, dds1);
		auto &[M1, N1] = tp1;

		if (M0.empty())
		{			 // No triangle in this portion
			bg = ed; // end the block without doing the intersection
		}

		decltype(final_cliques) cliques;
		reserve_ex(cliques, M0.size());

		auto do_intersections = [
										&N0 = N0,
										&M0 = M0,
										&N1 = N1,
										&M1 = M1,
										&E = graph.E,
										&cliques](const U &d1,
												  const U &d2,
												  size_t neiOffset)
		{
			// dst[d1:d2]
			for (U j = d1; j < d2; j++)
			{
				auto i1 = d1;
				auto i2 = d2;
				auto j1 = N0[j - neiOffset];
				auto j2 = N0[j + 1 - neiOffset];

				// intersection(E[i1:i2], EM[j1:j2])
				while (i1 < i2 && j1 < j2)
				{
					ASERT(E.size() > i1, "Out of boundary E[i1][0]!");
					auto &p = E[i1][0];
					ASERT(M0.size() > j1, "Out of boundary M0[j1]!");
					const auto &o1 = static_cast<V>(M0[j1]);
					ASERT(E.size() > o1, "Out of boundary E[o1][0]!");
					auto &q = E[o1][0];

					if (p < q)
					{
						i1++;
					}
					else
					{
						if (!(q < p))
						{
							auto &k1 = N1[i1 - neiOffset];
							auto &k2 = N1[i1 + 1 - neiOffset];
							auto k = k1;
							for (; k < k2; k++)
							{
								const auto &o2 = static_cast<V>(M1[k]);
								if (E[j][1] == E[o2][1])
								{
									cliques.push_back({i1, o1, j, o2});
									i1++;
									break;
								}
							}
							ASERT(k < k2, "Could n't find 4th element!");
						}
						j1++;
					}
				} // end of intersection loop
			} // end of for-loop
		};

		for (size_t i = bg; i < ed; i++)
		{
			do_intersections(O[i],
							 (i < O.size() - 1) ? O[i + 1] : graph.size_edges(),
							 dst_bg);
		}

		// tris.shrink_to_fit(); // Not necessary
		size_t last;
#pragma omp critical
		{
			last = final_cliques.size();
			if (!last)
			{
				// On the first thread try to estimate and reserve all the required memory
				reserve_ex(final_cliques, cliques.size() * graph.split_parts.size());
			}
			resize_ex(final_cliques, last + cliques.size());
		}
		// We do this instead of inserting in locked mode
		std::copy(cliques.cbegin(), cliques.cend(), final_cliques.begin() + last);
	} // end of parallel block

	return final_cliques;
}

template <typename U, typename V, size_t SIZE>
auto cliques_qud_par_parts(const graph_t<U, V, SIZE> &graph)
{

	const auto &O = graph.O;
	const auto &D = graph.D;
	const auto &E = graph.E;

	if (!graph.split_parts.size())
	{
		throw std::runtime_error("split_parts vector not initialized for parallel calls.");
	}

	std::vector<std::vector<std::array<V, SIZE + 2>>> parts_cliques(graph.split_parts.size());

	// Explicitly disable dynamic teams
	omp_set_dynamic(0);

	// Use n threads for all consecutive parallel regions
	omp_set_num_threads(graph.split_parts.size());

#pragma omp parallel default(none) shared(graph, O, D, E, parts_cliques, EXE_POL)
	{
		auto th_id = static_cast<size_t>(omp_get_thread_num());
		auto bg = graph.split_parts[th_id];
		auto ed = graph.O.size();
		auto dst_bg = O[bg];
		auto dst_ed = graph.size_edges();
		if (th_id < graph.split_parts.size() - 1)
		{
			ed = graph.split_parts[th_id + 1];
			dst_ed = O[ed];
		}

		// nds = O[E], dds = D[E]
		auto dt = dtake_par(O, D, E, dst_bg, dst_ed);
		const auto &nds0 = dt[0].first;
		const auto &dds0 = dt[0].second;
		const auto &nds1 = dt[1].first;
		const auto &dds1 = dt[1].second;

		/**
		 * @brief M: Generated indices
		 * @brief N: Offset of generated indices
		 */

		auto tp0 = multi_arange_omp(EXE_POL, nds0, dds0);
		auto &[M0, N0] = tp0;

		auto tp1 = multi_arange_omp(EXE_POL, nds1, dds1);
		auto &[M1, N1] = tp1;

		if (M0.empty())
		{			 // No triangle in this portion
			bg = ed; // end the block without doing the intersection
		}

		auto &cliques = parts_cliques[th_id];
		reserve_ex(cliques, M0.size());

		auto do_intersections = [
										&N0 = N0,
										&M0 = M0,
										&N1 = N1,
										&M1 = M1,
										&E = graph.E,
										&cliques](const U &d1,
												  const U &d2,
												  size_t neiOffset)
		{
			// dst[d1:d2]
			for (U j = d1; j < d2; j++)
			{
				auto i1 = d1;
				auto i2 = d2;
				auto j1 = N0[j - neiOffset];
				auto j2 = N0[j + 1 - neiOffset];

				// intersection(E[i1:i2], EM[j1:j2])
				while (i1 < i2 && j1 < j2)
				{
					ASERT(E.size() > i1, "Out of boundary E[i1][0]!");
					auto &p = E[i1][0];
					ASERT(M0.size() > j1, "Out of boundary M0[j1]!");
					const auto &o1 = static_cast<V>(M0[j1]);
					ASERT(E.size() > o1, "Out of boundary E[o1][0]!");
					auto &q = E[o1][0];

					if (p < q)
					{
						i1++;
					}
					else
					{
						if (!(q < p))
						{
							auto &k1 = N1[i1 - neiOffset];
							auto &k2 = N1[i1 + 1 - neiOffset];
							auto k = k1;
							for (; k < k2; k++)
							{
								const auto &o2 = static_cast<V>(M1[k]);
								if (E[j][1] == E[o2][1])
								{
									cliques.push_back({i1, o1, j, o2});
									i1++;
									break;
								}
							}
							ASERT(k < k2, "Could n't find 4th element!");
						}
						j1++;
					}
				} // end of intersection loop
			} // end of for-loop
		};

		for (size_t i = bg; i < ed; i++)
		{
			do_intersections(O[i],
							 (i < O.size() - 1) ? O[i + 1] : graph.size_edges(),
							 dst_bg);
		}
	} // end of parallel block

	return parts_cliques;
}

/**
 * @brief partition_graph splits a DIM=1 graph into several independent DIM=1
 * graphs such that no edge connects vertices that end up in different parts.
 *
 * In the input graph the source of each edge is implied by the vertex ranges of
 * the offset/degree vectors (first column "S" in the dataset) while E holds the
 * destinations (second column "E"). The graph is first decomposed into its
 * connected components (treating every edge as undirected), and whole
 * components are then distributed across the requested number of parts. Because
 * a component is never split, every edge of a returned graph has both endpoints
 * inside the same part: there is no edge or connection between different parts.
 *
 * Each returned graph keeps the original vertex id space (destination ids are
 * left untouched); only the vertices belonging to the part carry their original
 * degree and edges, every other vertex has degree zero. Therefore the union of
 * the edge sets of all returned graphs reproduces the edge set of the input
 * graph.
 *
 * The component-to-part assignment is balanced by edge count, mirroring the
 * behaviour of graph_t::make_split_parts. Because parts are made of whole
 * components, the number of returned graphs can never exceed the number of
 * connected components.
 *
 * @tparam U neighbor-list address type (edge id type)
 * @tparam V vertex id / degree type
 * @param graph input graph of dimension one
 * @param parts desired number of parts. When 0 (default) the graph is broken
 * into the maximum number of parts possible, i.e. one connected component per
 * part. The value is capped at the number of connected components.
 * @return a vector of owning pointers to the parted graphs. graph_t is
 * non-copyable and non-movable, hence the indirection.
 */
template <typename U, typename V>
auto partition_graph(const graph_t<U, V, 1> &graph, size_t parts = 0)
	-> std::vector<std::unique_ptr<graph_t<U, V, 1>>>
{
	const auto &D = graph.D;
	const auto &E = graph.E;

	const size_t vertex_count = graph.size_vertices();
	const size_t edge_count = graph.size_edges();

	std::vector<std::unique_ptr<graph_t<U, V, 1>>> result;

	if (vertex_count == 0)
	{
		return result;
	}

	// Local offset (prefix sum of degrees) so the input graph is not mutated.
	// offset[v] = index in E where the edges of source vertex v begin.
	std::vector<U> offset;
	if (graph.O.size() == vertex_count)
	{
		offset = graph.O; // reuse already materialized offsets
	}
	else
	{
		offset.resize(vertex_count);
		std::exclusive_scan(EXE_POL, D.cbegin(), D.cend(),
							offset.begin(), U{0}, std::plus<>());
	}

	// --- Connected components via union-find (edges treated as undirected) ---
	std::vector<size_t> parent(vertex_count);
	std::iota(parent.begin(), parent.end(), size_t{0});

	auto find = [&parent](size_t x) -> size_t
	{
		while (parent[x] != x)
		{
			parent[x] = parent[parent[x]]; // path halving
			x = parent[x];
		}
		return x;
	};
	auto unite = [&](size_t a, size_t b)
	{
		const size_t ra = find(a);
		const size_t rb = find(b);
		if (ra != rb)
		{
			parent[ra] = rb;
		}
	};

	for (size_t v = 0; v < vertex_count; ++v)
	{
		const U bg = offset[v];
		const U ed = (v + 1 < vertex_count) ? offset[v + 1]
											: static_cast<U>(edge_count);
		for (U e = bg; e < ed; ++e)
		{
			unite(v, static_cast<size_t>(E[e][0]));
		}
	}

	// Relabel component roots to contiguous ids in order of first appearance so
	// the component containing the smallest vertex id becomes component 0.
	constexpr size_t NONE = static_cast<size_t>(-1);
	std::vector<size_t> root_label(vertex_count, NONE);
	std::vector<size_t> vcomp(vertex_count);
	size_t component_count = 0;
	for (size_t v = 0; v < vertex_count; ++v)
	{
		const size_t r = find(v);
		if (root_label[r] == NONE)
		{
			root_label[r] = component_count++;
		}
		vcomp[v] = root_label[r];
	}

	// Number of edges owned by each component.
	std::vector<size_t> comp_edges(component_count, 0);
	for (size_t v = 0; v < vertex_count; ++v)
	{
		comp_edges[vcomp[v]] += static_cast<size_t>(D[v]);
	}

	// Number of parts: 0 means maximum possible (one component per part).
	size_t segments = (parts == 0) ? component_count : parts;
	segments = std::min(segments, component_count);

	// Exclusive prefix sum of component edge counts for balancing.
	std::vector<size_t> comp_off(component_count, 0);
	std::exclusive_scan(comp_edges.cbegin(), comp_edges.cend(),
						comp_off.begin(), size_t{0});

	// First component id of each part, balanced by edge count over whole
	// components (mirrors graph_t::make_split_parts).
	std::vector<size_t> seg_begin_comp(segments, component_count);
	seg_begin_comp[0] = 0;
	if (segments == component_count)
	{
		for (size_t i = 0; i < segments; ++i)
		{
			seg_begin_comp[i] = i;
		}
	}
	else if (edge_count > 0)
	{
		for (size_t c = 0, j = 1; c < component_count && j < segments; ++c)
		{
			const auto lhs = static_cast<unsigned __int128>(comp_off[c]) *
							 static_cast<unsigned __int128>(segments);
			const auto rhs = static_cast<unsigned __int128>(j) *
							 static_cast<unsigned __int128>(edge_count);
			if (lhs >= rhs)
			{
				seg_begin_comp[j++] = c;
			}
		}
	}
	else
	{
		// With no edges, balance by component count instead of edge count.
		for (size_t j = 1; j < segments; ++j)
		{
			seg_begin_comp[j] = (j * component_count) / segments;
		}
	}

	// Map each component to its part index.
	std::vector<size_t> comp_part(component_count, 0);
	for (size_t s = 0, c = 0; s < segments; ++s)
	{
		const size_t end_comp =
			(s + 1 < segments) ? seg_begin_comp[s + 1] : component_count;
		for (; c < end_comp; ++c)
		{
			comp_part[c] = s;
		}
	}

	// Bucket vertices by part (ascending vertex order is preserved).
	std::vector<std::vector<size_t>> part_vertices(segments);
	for (size_t v = 0; v < vertex_count; ++v)
	{
		part_vertices[comp_part[vcomp[v]]].push_back(v);
	}

	// Materialize each part as an independent DIM=1 graph.
	result.reserve(segments);
	for (size_t s = 0; s < segments; ++s)
	{
		auto part = std::make_unique<graph_t<U, V, 1>>();

		// Degree vector keeps the full vertex space; only the part's vertices
		// retain their original degree, everything else is zero.
		part->D.assign(vertex_count, 0);
		size_t part_edges = 0;
		for (const size_t v : part_vertices[s])
		{
			part->D[v] = D[v];
			part_edges += static_cast<size_t>(D[v]);
		}

		// Offset must index into the sliced edge list (depends only on D).
		part->build_offset();

		// Gather the part's edges in ascending vertex order.
		part->E.reserve(part_edges);
		for (const size_t v : part_vertices[s])
		{
			const U bg = offset[v];
			const U ed = (v + 1 < vertex_count) ? offset[v + 1]
												: static_cast<U>(edge_count);
			part->E.insert(part->E.end(), E.cbegin() + bg, E.cbegin() + ed);
		}

		result.push_back(std::move(part));
	}

	return result;
}

/**
 * @brief extract_bases if provided with edge graph and triangle cliques it will
 * extract triangle vertices, if provided with triangle graph and four-cliques it will provide
 * vertices of four cliques
 * @param graph
 * @param cliques
 */
template <size_t SIZE, typename U, typename V, size_t DIM, size_t CL_SIZE>
auto extract_bases(const graph_t<U, V, DIM> &graph,
				   const std::vector<std::array<V, CL_SIZE>> &cliques)
{

	std::vector<std::array<V, SIZE>> r;
	r.reserve(cliques.size());

	for (const auto &c : cliques)
	{ // loop list of cliques
		std::array<V, (DIM + 1) * CL_SIZE> arr;
		for (size_t cz = 0; cz < CL_SIZE; cz++)
		{ // loop per clique size
			// write source value
			arr[(DIM + 1) * cz] = graph.at_source(c[cz]);

			for (size_t d = 0; d < DIM; d++)
			{ // loop graph dimension
				arr[(DIM + 1) * cz + d + 1] = graph.E[c[cz]][d];
			}
		} // loop per clique size
		// arr has at most (DIM+1)*CL_SIZE elements (<= 12 in practice);
		// a sequential sort avoids the parallel-policy dispatch overhead per clique.
		std::sort(arr.begin(), arr.end());
		auto it = std::unique(arr.begin(), arr.end());
		if (std::distance(arr.begin(), it) != SIZE)
		{
			std::stringstream oss;
			oss << "Error: unique items "
				<< std::distance(arr.begin(), it)
				<< " doesn't match cliques size! "
				<< SIZE << std::endl
				<< "arr: "
				<< arr << std::endl;
			throw std::runtime_error(oss.str());
		}
		// TODO: initialize arr2 using the following link
		// TODO: https://stackoverflow.com/a/10930078
		std::array<V, SIZE> arr2;
		for (size_t s = 0; s < SIZE; s++)
		{
			arr2[s] = arr[s];
		}
		r.push_back(arr2);
	}

	return r;
}

#endif // CLIQUES_HPP_
