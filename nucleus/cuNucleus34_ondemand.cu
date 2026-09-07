#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <omp.h>
#include <stdexcept>
#include <sstream>

#include <thrust/iterator/permutation_iterator.h>

#include "bucket_ex.hpp"
#include "cliques.hpp"
#include "cpuinfo.hpp"
#include "graph/graph.hpp"
#include "ioutils.hpp"
#include "json.h"
#include "utility/defs.hpp"
#include "utility/def_system.hpp"
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include "utils.cuh"

using namespace json;

#define APP_NAME "cuda-nucleus34-ondemand"
#define APP_VER "0.1.0"

/**
 * (3,4)-nucleus peel that never materialises the four-cliques.
 *
 * cuNucleus34_direct stores one qud_t per four-clique plus a 4-entry incidence
 * CSR and a claim flag: ~36 bytes per four-clique, and ~84 at the
 * thrust::sort_by_key peak that builds the incidence. soc-LiveJournal1 has
 * 9,933,532,019 four-cliques, so that is ~358 GB resident and ~834 GB at the
 * peak against an H100's 80 GB -- and the count does not even fit the 32-bit
 * d_global_num_4cliques counter. No amount of segmenting fixes this: the
 * segmented four-clique pass bounds the *scratch*, but its output is still the
 * whole clique list.
 *
 * The observation that removes the clique list entirely:
 *
 *   Triangle t has vertices u<v<w (in orientation rank) and base edges
 *   a=(u,w), b=(u,v), c=(v,w). A four-clique containing t is a vertex x
 *   adjacent to all of u, v and w -- equivalently, x appears as the THIRD
 *   VERTEX in the triangle lists of all three of t's edges, because (u,v,x),
 *   (u,w,x) and (v,w,x) must all be triangles.
 *
 * So if we build an "edge -> incident triangles" CSR whose entries are keyed by
 * third vertex, a 3-way zipper over the lists of t's three edges enumerates
 * exactly t's four-cliques -- and each matching entry already carries the
 * partner's triangle index, so the three partners come out of the intersection
 * for free. No hash table, no binary search back into the triangle list.
 *
 * That CSR holds 3 entries per triangle (each triangle is listed under each of
 * its three edges), not one per four-clique: 857M entries for LiveJournal
 * instead of 9.9e9 four-cliques. Memory becomes O(triangles), ~14 GB, and the
 * same zipper serves both the support-counting pass and the peel, where a dying
 * triangle re-derives its cliques on demand. Each four-clique is re-enumerated
 * once per member death, so the peel does ~4x the work of one enumeration --
 * paid in time, which we have, instead of memory, which we do not.
 */

// 64-bit four-clique total. The existing d_global_num_4cliques is `unsigned`,
// which cannot represent LiveJournal's 9.93e9 four-cliques.
__device__ unsigned long long d_num_4cliques_64;
// Number of triangles that died in the current round, and the size of the
// next round's alive-list compaction.
__device__ unsigned d_dying_count;

namespace nucleus34_ondemand
{
	__global__ void reset_counters()
	{
		d_num_4cliques_64 = 0ULL;
		d_dying_count = 0U;
	}

	/**
	 * Recovers the three vertices of each triangle and emits its three
	 * incidence entries.
	 *
	 * tri_t is a triple of oriented-edge indices: a=(u,w), b=(u,v), c=(v,w)
	 * (see cu_cliques_tris, which sets a to the matching (u,w) edge, b to the
	 * (u,v) edge it started from and c to the (v,w) edge that closed the
	 * triangle). So u=src[b], v=dst[b], w=dst[c], and the third vertex under
	 * each edge is the one not on that edge.
	 */
	__global__ void build_incidence_entries(const tri_t *__restrict__ tris,
											const VERTEX_T *__restrict__ src,
											const VERTEX_T *__restrict__ E,
											size_t nTri,
											unsigned long long *__restrict__ keys,
											EDGE_T *__restrict__ vals)
	{
		size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		if (i >= nTri)
			return;

		const tri_t t = tris[i];
		const VERTEX_T u = src[t.b];
		const VERTEX_T v = E[t.b];
		const VERTEX_T w = E[t.c];

		// key = (edge << 32) | third vertex, so one sort orders the entries by
		// edge and, within an edge, by the third vertex the zipper matches on.
		keys[3 * i + 0] = (static_cast<unsigned long long>(t.b) << 32) | w; // (u,v) : x = w
		keys[3 * i + 1] = (static_cast<unsigned long long>(t.a) << 32) | v; // (u,w) : x = v
		keys[3 * i + 2] = (static_cast<unsigned long long>(t.c) << 32) | u; // (v,w) : x = u
		vals[3 * i + 0] = static_cast<EDGE_T>(i);
		vals[3 * i + 1] = static_cast<EDGE_T>(i);
		vals[3 * i + 2] = static_cast<EDGE_T>(i);
	}

	__global__ void split_keys(const unsigned long long *__restrict__ keys,
							   size_t n,
							   VERTEX_T *__restrict__ edge_of,
							   VERTEX_T *__restrict__ third_of)
	{
		size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		if (i >= n)
			return;
		edge_of[i] = static_cast<VERTEX_T>(keys[i] >> 32);
		third_of[i] = static_cast<VERTEX_T>(keys[i] & 0xFFFFFFFFULL);
	}

	/**
	 * 3-way zipper over the incidence lists of a triangle's three edges. Every
	 * common third vertex is one four-clique on this triangle; the callback
	 * receives the three partner triangle indices, which the lists carry
	 * directly.
	 */
	template <typename F>
	__device__ inline void for_each_clique_of(const tri_t &t,
											  const EDGE_T *__restrict__ incO,
											  const VERTEX_T *__restrict__ incThird,
											  const EDGE_T *__restrict__ incTri,
											  F fn)
	{
		EDGE_T i = incO[t.b], iE = incO[t.b + 1]; // (u,v,x) -> partner (u,v,x)
		EDGE_T j = incO[t.a], jE = incO[t.a + 1]; // (u,w,x) -> partner (u,w,x)
		EDGE_T k = incO[t.c], kE = incO[t.c + 1]; // (v,w,x) -> partner (v,w,x)

		while (i < iE && j < jE && k < kE)
		{
			const VERTEX_T x1 = incThird[i];
			const VERTEX_T x2 = incThird[j];
			const VERTEX_T x3 = incThird[k];
			if (x1 == x2 && x2 == x3)
			{
				fn(incTri[i], incTri[j], incTri[k]);
				++i;
				++j;
				++k;
			}
			else
			{
				const VERTEX_T mx = max(x1, max(x2, x3));
				if (x1 < mx)
					++i;
				if (x2 < mx)
					++j;
				if (x3 < mx)
					++k;
			}
		}
	}

	/**
	 * Support = number of four-cliques on each triangle. Storage-free: the
	 * cliques are counted through the zipper and thrown away.
	 */
	__global__ void count_support(const tri_t *__restrict__ tris,
								  const EDGE_T *__restrict__ incO,
								  const VERTEX_T *__restrict__ incThird,
								  const EDGE_T *__restrict__ incTri,
								  MTYPE *__restrict__ DD,
								  size_t nTri)
	{
		size_t v = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		if (v >= nTri)
			return;

		unsigned cnt = 0;
		for_each_clique_of(tris[v], incO, incThird, incTri,
						   [&cnt](EDGE_T, EDGE_T, EDGE_T) { ++cnt; });
		DD[v] = static_cast<MTYPE>(cnt);
		// Each four-clique is counted by each of its 4 triangles.
		if (cnt)
			atomicAdd(&d_num_4cliques_64, static_cast<unsigned long long>(cnt));
	}

	/**
	 * Round phase 1: over the compacted alive list, mark every triangle whose
	 * support has fallen to the current level as dying and append it to the
	 * dying list. Runs to completion before phase 2, so the V / round stamps
	 * phase 2 reads are stable -- that kernel boundary is what lets the
	 * ownership test below resolve same-round deaths without a claim flag.
	 */
	__global__ void mark_dying(const EDGE_T *__restrict__ alive,
							   size_t nAlive,
							   MTYPE *__restrict__ V,
							   const MTYPE *__restrict__ DD,
							   unsigned *__restrict__ round_of,
							   unsigned round,
							   EDGE_T *__restrict__ dying)
	{
		size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		if (i >= nAlive)
			return;
		const EDGE_T t = alive[i];
		const MTYPE cur = static_cast<MTYPE>(d_2peel_current);
		if (V[t] != -1 || DD[t] > cur)
			return;
		if (atomicCAS(&V[t], -1, cur) != -1)
			return; // another thread took it (alive list may repeat under compaction races)
		round_of[t] = round;
		const unsigned slot = warp_aggregated_add(&d_dying_count);
		dying[slot] = t;
		warp_aggregated_add(&d_peeled);
	}

	/**
	 * Round phase 2: each dying triangle re-derives its four-cliques and debits
	 * the survivors.
	 *
	 * A four-clique must be destroyed exactly once. cuNucleus34_direct spends an
	 * atomicCAS claim flag per clique for this; with no clique array there is
	 * nothing to flag, so ownership is decided from the death stamps instead:
	 * the clique belongs to the FIRST of its four triangles to die, and among
	 * triangles dying in the same round, to the lowest index. Both are readable
	 * because phase 1 already finished.
	 */
	__global__ void debit_partners(const EDGE_T *__restrict__ dying,
								   size_t nDying,
								   const tri_t *__restrict__ tris,
								   const EDGE_T *__restrict__ incO,
								   const VERTEX_T *__restrict__ incThird,
								   const EDGE_T *__restrict__ incTri,
								   const MTYPE *__restrict__ V,
								   const unsigned *__restrict__ round_of,
								   MTYPE *__restrict__ DD,
								   unsigned round)
	{
		size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		if (i >= nDying)
			return;
		const EDGE_T t = dying[i];

		for_each_clique_of(tris[t], incO, incThird, incTri,
						   [&](EDGE_T p1, EDGE_T p2, EDGE_T p3)
						   {
							   const EDGE_T p[3] = {p1, p2, p3};

							   // Own the clique only if no partner died earlier,
							   // and no same-round partner has a lower index.
							   for (int s = 0; s < 3; ++s)
							   {
								   if (V[p[s]] == -1)
									   continue; // still alive
								   if (round_of[p[s]] < round)
									   return;   // clique already destroyed
								   if (p[s] < t)
									   return;   // same round, partner owns it
							   }

							   for (int s = 0; s < 3; ++s)
							   {
								   if (V[p[s]] == -1)
									   atomicSub(&DD[p[s]], 1);
							   }
						   });
	}

	/**
	 * The level advances by one per round unless something is already at or
	 * below it, so the next level is the smallest surviving support. Computing
	 * it from the alive list keeps the peel from spinning through empty levels:
	 * LiveJournal's K_max is 351 but the supports are far sparser than that
	 * range, and a round that peels nothing still costs a full pass.
	 */
	__global__ void min_alive_support(const EDGE_T *__restrict__ alive,
									  size_t nAlive,
									  const MTYPE *__restrict__ V,
									  const MTYPE *__restrict__ DD,
									  unsigned *__restrict__ out)
	{
		size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		if (i >= nAlive)
			return;
		const EDGE_T t = alive[i];
		if (V[t] != -1)
			return;
		const MTYPE d = DD[t];
		atomicMin(out, static_cast<unsigned>(d < 0 ? 0 : d));
	}

	__global__ void set_level(const unsigned *__restrict__ next_level)
	{
		const unsigned nl = *next_level;
		d_2peel_current = (nl > d_2peel_current) ? nl : d_2peel_current;
	}
} // namespace nucleus34_ondemand

int main(int argc, char **argv)
{
	if (argc < 2)
	{
		std::cerr << "Usage: "
				  << strip_path(argv[0])
				  << " <dataset file (.mtx or ."
				  << graph_t<EDGE_T, VERTEX_T>::ext << ")>"
				  << " [results output file (optional)]"
				  << std::endl;
		return EXIT_FAILURE;
	}

	std::string dataset_file(argv[1]);
	std::string output_file;
	if (argc > 2)
	{
		output_file = argv[2];
	}

	std::string step("");

	SET_STEP("0.1 Environment info collection");
	auto cpu_info{cpuinfo()};
	auto gpu_info{gpuinfo()};
	int nProcs = omp_get_num_procs();

	std::ostringstream jsonSS;
	g_jsonSS = &jsonSS;

	beginObject(true, "environment");
	field(true, "app", APP_NAME);
	field(false, "version", APP_VER);
	field(false, "g++", CXX_VER);
	field(false, "cuda", CUDART_VERSION);
	field(false, "thrust", thrust_version_string());
	field(false, "policy", std::string(EXE_POL_STR));
	field(false, "procs", nProcs);
	field(false, "cpu_core_count", cpu_info.size());
	field(false, "cpu_model", cpu_info.size() ? cpu_info[0]["model name"] : std::string("Unknown"));

	for (const auto &gi : gpu_info)
	{
		for (const auto &p : gi)
		{
			std::string gpu_key = "gpu_" + p.first;
			std::replace(gpu_key.begin(), gpu_key.end(), ' ', '_');
			field(false, gpu_key, p.second);
		}
	}

	field(false, "host", hostname());
	endObject();

	beginObject(false, "dataset");
	field(true, "file", strip_path(dataset_file));

	try
	{
		SET_STEP("0.2 Data loading and graph construction");
		graph_t<EDGE_T, VERTEX_T> oriented_graph;
		auto ext = get_ext(dataset_file);

		auto t = hrc::now();
		if (ext == ".mtx")
		{
			oriented_graph.from_edges(dataset_file, true);
			oriented_graph.make_oriented();
		}
		else if (ext == ".edges" || ext == ".txt")
		{
			oriented_graph.from_edges(dataset_file, false);
			oriented_graph.make_oriented();
		}
		else if (ext == graph_t<EDGE_T, VERTEX_T>::ext)
		{
			oriented_graph.deserialize(dataset_file);
			oriented_graph.build_offset();
		}
		else
		{
			throw std::runtime_error("Unknown dataset format: " + ext);
		}
		oriented_graph.make_split_parts(nProcs);
		seconds t_data = hrc::now() - t;

		const auto nVert = oriented_graph.size_vertices();
		const auto nEdge = oriented_graph.size_edges();

		field(false, "vertex_count", nVert);
		field(false, "edge_count", nEdge * 2);
		{
			std::ostringstream os;
			os << std::fixed << std::setprecision(2) << t_data.count();
			raw_field(false, "data_loading_time_sec", os.str());
		}

		std::unordered_map<std::string, seconds> tms;

		SET_STEP("0.3 GPU Kernel Initialization");
		initialize_kernel<<<1, 1>>>();
		nucleus34_ondemand::reset_counters<<<1, 1>>>();
		CU_ERR(cudaPeekAtLastError());
		CU_ERR(cudaDeviceSynchronize());

		SET_STEP("1.0 Allocating memory for triangle phase");
		thrust::device_vector<VERTEX_T> d_D(oriented_graph.D);
		thrust::device_vector<EDGE_T> d_O(oriented_graph.O);
		auto e_ptr = reinterpret_cast<const VERTEX_T *>(oriented_graph.E.data());
		thrust::device_vector<VERTEX_T> d_E(e_ptr, e_ptr + nEdge);

		auto path_count = thrust::reduce(
			thrust::make_permutation_iterator(d_D.begin(), d_E.begin()),
			thrust::make_permutation_iterator(d_D.begin(), d_E.end()),
			std::size_t{0}, thrust::plus<std::size_t>());

		thrust::device_vector<VERTEX_T> d_DE(nEdge, 0);
		thrust::device_vector<EDGE_T> d_OE(nEdge, 0);
		thrust::device_vector<EDGE_T> d_N(nEdge + 1, 0);

		const size_t tri_single_shot_bytes =
			path_count * sizeof(MTYPE) + path_count * sizeof(tri_t);
		size_t tri_free_bytes = 0, tri_total_bytes = 0;
		cudaMemGetInfo(&tri_free_bytes, &tri_total_bytes);
		const size_t tri_budget =
			static_cast<size_t>(static_cast<double>(tri_free_bytes) * cu::GPU_SCRATCH_MEM_SAFETY);
		const bool tri_segmented = (tri_budget > 0 && tri_single_shot_bytes > tri_budget);

		thrust::device_vector<MTYPE> d_M;
		thrust::device_vector<tri_t> d_tris;
		if (tri_segmented)
		{
			std::cerr << "[tri] single-shot triangle scratch ~" << (tri_single_shot_bytes >> 20)
					  << " MiB exceeds budget ~" << (tri_budget >> 20)
					  << " MiB; using path-count-aware segmented pass" << std::endl;
		}
		else
		{
			d_M.assign(path_count, 1);
			d_tris.resize(path_count);
		}

		thrust::device_vector<VERTEX_T> d_trisD(nEdge, 0);
		thrust::device_vector<VERTEX_T> d_trisO(nEdge, 0);

		CU_ERR(cudaPeekAtLastError());
		CU_ERR(cudaDeviceSynchronize());

		SET_STEP("1.1 Triangle counting");
		size_t triangle_count = 0;
		t = hrc::now();
		if (tri_segmented)
		{
			cu::compute_triangles_segmented(d_D, d_O, d_E, 1, d_tris);
		}
		else
		{
			cu::compute_triangles(d_D, d_O, d_E, d_DE, d_OE, d_N, d_M, d_tris);
		}
		tms["Triangles"] = hrc::now() - t;
		cudaMemcpyFromSymbol(&triangle_count, d_global_num_triangles,
							 sizeof(d_global_num_triangles), 0, cudaMemcpyDeviceToHost);
		field(false, "triangle_time_sec", tms["Triangles"].count());
		field(false, "triangle_count", triangle_count);

		CU_ERR(cudaPeekAtLastError());
		CU_ERR(cudaDeviceSynchronize());

		if (!triangle_count)
		{
			endObject();
			lastObject();
			g_jsonSS = nullptr;
			std::cout << jsonSS.str() << std::endl;
			return EXIT_SUCCESS;
		}

		// Free the triangle-phase scratch; only d_tris and the base CSR survive.
		thrust::device_vector<VERTEX_T>().swap(d_DE);
		thrust::device_vector<EDGE_T>().swap(d_OE);
		thrust::device_vector<EDGE_T>().swap(d_N);
		thrust::device_vector<MTYPE>().swap(d_M);
		thrust::device_vector<VERTEX_T>().swap(d_trisD);
		thrust::device_vector<VERTEX_T>().swap(d_trisO);
		d_tris.resize(triangle_count);
		d_tris.shrink_to_fit();

		// The triangle list must be sorted so the incidence entries (and the
		// four-clique identities derived from them) are canonical.
		SET_STEP("2.0 Sorting triangles");
		t = hrc::now();
		thrust::sort(d_tris.begin(), d_tris.end());
		tms["Triangles-sort"] = hrc::now() - t;

		SET_STEP("2.1 Building edge -> triangle incidence (3 entries per triangle)");
		t = hrc::now();
		const size_t nInc = 3 * triangle_count;
		if (nInc > static_cast<size_t>(std::numeric_limits<EDGE_T>::max()))
		{
			std::ostringstream os;
			os << "incidence size " << nInc << " exceeds the 32-bit EDGE_T index range;"
			   << " rebuild with -DEdgeType=unsigned_long";
			throw std::runtime_error(os.str());
		}

		// src vertex of each oriented edge, needed to recover triangle vertices.
		thrust::device_vector<VERTEX_T> d_src(nEdge);
		thrust::upper_bound(d_O.cbegin() + 1, d_O.cend(),
							thrust::counting_iterator<EDGE_T>(0),
							thrust::counting_iterator<EDGE_T>(static_cast<EDGE_T>(nEdge)),
							d_src.begin());

		thrust::device_vector<EDGE_T> d_incTri(nInc);
		thrust::device_vector<VERTEX_T> d_incThird(nInc);
		thrust::device_vector<EDGE_T> d_incO(nEdge + 1);
		{
			thrust::device_vector<unsigned long long> d_keys(nInc);
			nucleus34_ondemand::build_incidence_entries<<<blocks(triangle_count), MAX_THRD_BLK>>>(
				thrust::raw_pointer_cast(d_tris.data()),
				thrust::raw_pointer_cast(d_src.data()),
				thrust::raw_pointer_cast(d_E.data()),
				triangle_count,
				thrust::raw_pointer_cast(d_keys.data()),
				thrust::raw_pointer_cast(d_incTri.data()));
			CU_ERR(cudaPeekAtLastError());
			CU_ERR(cudaDeviceSynchronize());

			thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_incTri.begin());

			thrust::device_vector<VERTEX_T> d_incEdge(nInc);
			nucleus34_ondemand::split_keys<<<blocks(nInc), MAX_THRD_BLK>>>(
				thrust::raw_pointer_cast(d_keys.data()), nInc,
				thrust::raw_pointer_cast(d_incEdge.data()),
				thrust::raw_pointer_cast(d_incThird.data()));
			CU_ERR(cudaPeekAtLastError());
			CU_ERR(cudaDeviceSynchronize());
			thrust::device_vector<unsigned long long>().swap(d_keys);

			thrust::lower_bound(d_incEdge.cbegin(), d_incEdge.cend(),
								thrust::counting_iterator<VERTEX_T>(0),
								thrust::counting_iterator<VERTEX_T>(static_cast<VERTEX_T>(nEdge + 1)),
								d_incO.begin());
		}
		thrust::device_vector<VERTEX_T>().swap(d_src);
		CU_ERR(cudaDeviceSynchronize());
		tms["Incidence"] = hrc::now() - t;
		field(false, "incidence_time_sec", tms["Incidence"].count());
		field(false, "incidence_entries", nInc);

		{
			size_t fb = 0, tb = 0;
			cudaMemGetInfo(&fb, &tb);
			std::cerr << "[mem] incidence entries=" << nInc
					  << " bytes=" << ((nInc * (sizeof(EDGE_T) + sizeof(VERTEX_T))) >> 20)
					  << " MiB | gpu_free=" << (fb >> 20) << "/" << (tb >> 20) << " MiB" << std::endl;
		}

		SET_STEP("3.0 Counting four-clique support (no clique storage)");
		thrust::device_vector<MTYPE> d_DD(triangle_count);
		t = hrc::now();
		nucleus34_ondemand::count_support<<<blocks(triangle_count), MAX_THRD_BLK>>>(
			thrust::raw_pointer_cast(d_tris.data()),
			thrust::raw_pointer_cast(d_incO.data()),
			thrust::raw_pointer_cast(d_incThird.data()),
			thrust::raw_pointer_cast(d_incTri.data()),
			thrust::raw_pointer_cast(d_DD.data()),
			triangle_count);
		CU_ERR(cudaPeekAtLastError());
		CU_ERR(cudaDeviceSynchronize());
		tms["Support"] = hrc::now() - t;

		unsigned long long incidence_total = 0ULL;
		cudaMemcpyFromSymbol(&incidence_total, d_num_4cliques_64,
							 sizeof(d_num_4cliques_64), 0, cudaMemcpyDeviceToHost);
		const unsigned long long fourclique_count = incidence_total / 4ULL;
		field(false, "four_clique_time_sec", tms["Support"].count());
		field(false, "four_clique_count", fourclique_count);
		std::cerr << "[mem] support four_cliques=" << fourclique_count
				  << " incidences=" << incidence_total
				  << " time=" << std::fixed << std::setprecision(1) << tms["Support"].count() << "s"
				  << std::endl;

		if (!fourclique_count)
		{
			endObject();
			lastObject();
			g_jsonSS = nullptr;
			std::cout << jsonSS.str() << std::endl;
			return EXIT_SUCCESS;
		}

		SET_STEP("4.0 Peeling (on-demand clique re-enumeration)");
		thrust::device_vector<MTYPE> d_kv(triangle_count, -1);
		thrust::device_vector<unsigned> d_round(triangle_count, 0xFFFFFFFFu);
		thrust::device_vector<EDGE_T> d_alive(triangle_count);
		thrust::device_vector<EDGE_T> d_alive_next(triangle_count);
		thrust::device_vector<EDGE_T> d_dying(triangle_count);
		thrust::device_vector<unsigned> d_next_level(1);
		thrust::sequence(d_alive.begin(), d_alive.end());
		size_t nAlive = triangle_count;

		t = hrc::now();
		cu::initialize<<<1, 1>>>(0U);
		CU_ERR(cudaPeekAtLastError());
		CU_ERR(cudaDeviceSynchronize());

		unsigned peeled = 0U;
		unsigned round = 0U;
		unsigned k_max_gpu = 0U;
		size_t last_compact = triangle_count;

		while (peeled < triangle_count)
		{
			// Advance the level to the smallest surviving support so empty
			// levels cost nothing.
			thrust::fill(d_next_level.begin(), d_next_level.end(), 0xFFFFFFFFu);
			nucleus34_ondemand::min_alive_support<<<blocks(nAlive), MAX_THRD_BLK>>>(
				thrust::raw_pointer_cast(d_alive.data()), nAlive,
				thrust::raw_pointer_cast(d_kv.data()),
				thrust::raw_pointer_cast(d_DD.data()),
				thrust::raw_pointer_cast(d_next_level.data()));
			CU_ERR(cudaPeekAtLastError());
			nucleus34_ondemand::set_level<<<1, 1>>>(
				thrust::raw_pointer_cast(d_next_level.data()));
			CU_ERR(cudaPeekAtLastError());

			unsigned zero = 0U;
			cudaMemcpyToSymbol(d_dying_count, &zero, sizeof(zero), 0, cudaMemcpyHostToDevice);

			nucleus34_ondemand::mark_dying<<<blocks(nAlive), MAX_THRD_BLK>>>(
				thrust::raw_pointer_cast(d_alive.data()), nAlive,
				thrust::raw_pointer_cast(d_kv.data()),
				thrust::raw_pointer_cast(d_DD.data()),
				thrust::raw_pointer_cast(d_round.data()), round,
				thrust::raw_pointer_cast(d_dying.data()));
			CU_ERR(cudaPeekAtLastError());
			CU_ERR(cudaDeviceSynchronize());

			unsigned nDying = 0U;
			cudaMemcpyFromSymbol(&nDying, d_dying_count, sizeof(nDying), 0, cudaMemcpyDeviceToHost);

			if (nDying)
			{
				nucleus34_ondemand::debit_partners<<<blocks(nDying), MAX_THRD_BLK>>>(
					thrust::raw_pointer_cast(d_dying.data()), nDying,
					thrust::raw_pointer_cast(d_tris.data()),
					thrust::raw_pointer_cast(d_incO.data()),
					thrust::raw_pointer_cast(d_incThird.data()),
					thrust::raw_pointer_cast(d_incTri.data()),
					thrust::raw_pointer_cast(d_kv.data()),
					thrust::raw_pointer_cast(d_round.data()),
					thrust::raw_pointer_cast(d_DD.data()), round);
				CU_ERR(cudaPeekAtLastError());
				CU_ERR(cudaDeviceSynchronize());
			}

			cudaMemcpyFromSymbol(&peeled, d_peeled, sizeof(d_peeled), 0, cudaMemcpyDeviceToHost);
			++round;

			// Recompact the alive list once it has halved. Phase 1 scans this
			// list every round, so letting it keep the dead would reproduce the
			// full-array rescan this design exists to avoid.
			if (peeled < triangle_count && (triangle_count - peeled) * 2 <= last_compact)
			{
				// Out-of-place: thrust::copy_if does not permit the output range
				// to alias the input.
				auto Vp = thrust::raw_pointer_cast(d_kv.data());
				auto end = thrust::copy_if(
					d_alive.begin(), d_alive.begin() + nAlive, d_alive_next.begin(),
					[Vp] __device__(EDGE_T x) -> bool { return Vp[x] == -1; });
				nAlive = static_cast<size_t>(end - d_alive_next.begin());
				d_alive.swap(d_alive_next);
				last_compact = nAlive;
			}
		}
		CU_ERR(cudaDeviceSynchronize());
		tms["Peeling"] = hrc::now() - t;

		cudaMemcpyFromSymbol(&k_max_gpu, d_2peel_current, sizeof(d_2peel_current), 0,
							 cudaMemcpyDeviceToHost);

		std::vector<MTYPE> h_kv(triangle_count);
		thrust::copy(d_kv.cbegin(), d_kv.cend(), h_kv.begin());

		// Every triangle must carry a peel value; a surviving -1 sentinel means
		// the peel exited with work left, which the loop condition alone cannot
		// prove. Reported rather than asserted so a bad run is still diagnosable.
		const size_t unpeeled =
			static_cast<size_t>(thrust::count(d_kv.cbegin(), d_kv.cend(), static_cast<MTYPE>(-1)));
		field(false, "unpeeled_count", unpeeled);
		if (unpeeled)
		{
			std::cerr << "[warn] " << unpeeled << " triangles left unpeeled" << std::endl;
		}

		field(false, "peeling_mode", std::string("ondemand-reenumerate"));
		field(false, "peeling_time_sec", tms["Peeling"].count());
		field(false, "peeling_iterations", round);
		field(false, "K_max", k_max_gpu);

		tms["all"] = seconds::zero();
		for (const auto &tm : tms)
		{
			if (tm.first != "all")
				tms["all"] += tm.second;
		}
		field(false, "total_time_sec", tms["all"].count());

		SET_STEP("4.3 Finalizing results on host");
		field(false, "K_count", h_kv.size());
		{
			auto n_dbg = std::min<std::size_t>(PRINT_CAP, h_kv.size());
			array_field(false, "K", h_kv.cbegin(), h_kv.cbegin() + n_dbg,
						[](std::ostream &os, const auto &v) { os << v; });
		}

		if (!output_file.empty())
		{
			std::vector<tri_t> h_tris(triangle_count);
			thrust::copy(d_tris.cbegin(), d_tris.cend(), h_tris.begin());
			std::ofstream out(output_file);
			if (!out)
				throw std::runtime_error("Failed to open output file: " + output_file);
			for (size_t i = 0; i < triangle_count; ++i)
				out << h_tris[i] << '\t' << h_kv[i] << '\n';
		}

		// Accumulate in long long: without an explicit init, std::reduce deduces
		// the accumulator from the iterator (MTYPE = int), and LiveJournal's
		// 285.7M peel values averaging ~35 sum to ~1e10 -- which wraps int32 and
		// reports a negative average.
		auto k_avg_gpu = static_cast<double>(
							 std::reduce(EXE_POL, h_kv.cbegin(), h_kv.cend(),
										 static_cast<long long>(0))) /
						 static_cast<double>(h_kv.size());
		field(false, "K_avg", k_avg_gpu);

		endObject();
		lastObject();
		g_jsonSS = nullptr;
		std::cout << jsonSS.str() << std::endl;
	}
	catch (thrust::system_error &e)
	{
		std::cerr << "Allocation failed: " << e.what() << std::endl;
	}
	catch (...)
	{
		auto expPtr = std::current_exception();
		try
		{
			if (expPtr)
				std::rethrow_exception(expPtr);
		}
		catch (const std::exception &e)
		{
			std::cerr
				<< "Dataset: " << dataset_file << std::endl
				<< "Step: " << step << std::endl
				<< "Error: " << e.what() << std::endl
				<< "NVIDIA SMI: " << std::endl
				<< cu::nvidiaSmi() << std::endl;

			if (!jsonSS.str().empty())
			{
				field(false, "errorstep", step);
				field(false, "error", std::string(e.what()));
				endObject();
				lastObject();
				g_jsonSS = nullptr;
				std::cout << jsonSS.str() << std::endl;
			}
		}
	}
	g_jsonSS = nullptr;
	return EXIT_SUCCESS;
}
