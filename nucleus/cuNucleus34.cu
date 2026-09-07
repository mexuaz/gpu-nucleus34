#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <limits>
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
#include <thrust/sort.h>
#include "utils.cuh"

using namespace json;

#define APP_NAME "cuda-nucleus34-direct"
#define APP_VER "0.9.3"

// Offsets into the triangle -> four-clique incidence array. This MUST be wider
// than EDGE_T. The array holds 4 entries per four-clique, so the final offset
// is 4 * four_clique_count; with EDGE_T = unsigned int (-DEDGE_U) that wraps
// once there are more than 2^30 four-cliques, and the peel then walks garbage
// ranges and reports a wrong K_max WITHOUT ERRORING. Measured on degree-prefix
// subgraphs of com-Orkut: at 1.22e9 four-cliques K_max came back 432 against a
// true 59 (arb-nucleus-decomp and cuNucleus34_ondemand agree on 59).
// The four-clique IDs stored in the array stay EDGE_T; only the offsets widen.
#define INCOFF_T unsigned long long


// Portable wrappers around cudaMemAdvise. CUDA 12.2 introduced (and CUDA 13
// made the default) a signature that takes a cudaMemLocation struct instead of
// an int device id. These helpers compile on both the old and new toolkits.
static inline void mem_advise_preferred_host(const void *ptr, size_t bytes)
{
#if CUDART_VERSION >= 12020
	cudaMemLocation loc{};
	loc.type = cudaMemLocationTypeHost;
	loc.id = 0;
	cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, loc);
#else
	cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
#endif
}

static inline void mem_advise_accessed_by_device(const void *ptr, size_t bytes, int device)
{
#if CUDART_VERSION >= 12020
	cudaMemLocation loc{};
	loc.type = cudaMemLocationTypeDevice;
	loc.id = device;
	cudaMemAdvise(ptr, bytes, cudaMemAdviseSetAccessedBy, loc);
#else
	cudaMemAdvise(ptr, bytes, cudaMemAdviseSetAccessedBy, device);
#endif
}

// Milestone note on stderr, next to the [phase] lines. The counts that decide
// whether this pipeline fits on one device (path counts, triangle count,
// four-clique count) are otherwise only visible in the JSON written at the very
// end, which a run that dies of an allocation failure never reaches. Printing
// them as they become known is what makes an out-of-memory run diagnosable.
static void mem_note(const char *tag, const std::string &detail)
{
	size_t free_bytes = 0, total_bytes = 0;
	cudaMemGetInfo(&free_bytes, &total_bytes);
	std::cerr << "[mem] " << tag << ' ' << detail
			  << " | gpu_free=" << (free_bytes >> 20) << "/" << (total_bytes >> 20)
			  << " MiB" << std::endl;
}

namespace nucleus34_direct
{
	/**
	 * (3,4)-nucleus peel without a four-cliques graph.
	 *
	 * A four-clique is a FOUR-way relation over triangles: when one of its
	 * triangles dies the clique dies once, and the other three must each lose
	 * exactly one support immediately.
	 *
	 * cuNucleus34.cu flattens every DIM=3 slot into 3 directed dbl_t edges (12
	 * per four-clique), scales degree by 3 and divides the peel values by 3 at
	 * the end. That subtracts 1 per slot entry, so a partner loses 1 of the 3
	 * it is owed at the clique's death and collects the rest only as the other
	 * partners die -- the same split-debit that inflated cuKtruss's trussness.
	 * (The host bucket_ex path does NOT have this problem: bucket_ex::peel
	 * tests the whole slot and debits all survivors at once, which is why it
	 * applies no /3 rescaling.)
	 *
	 * Here the adjacency carries four-clique IDS, so a dying triangle claims
	 * the clique with an atomicCAS. Exactly one claim wins even when several of
	 * its triangles die in the same round, and the winner debits all surviving
	 * partners in full. Support is a plain four-clique count, so there is no 3x
	 * slot scaling and no division afterwards -- and the 12-ids-per-clique
	 * flattened edge list is never built, which is what let the peel stay
	 * device-resident here without the memory-mode selection.
	 */
	__global__ void peel_fourcliques(const INCOFF_T *__restrict__ O,
									 const EDGE_T *__restrict__ INC,
									 const qud_t *__restrict__ QUD,
									 MTYPE *__restrict__ V,
									 MTYPE *__restrict__ DD,
									 int *__restrict__ claimed,
									 size_t nTri)
	{
		size_t v = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		if (v >= nTri)
			return;
		const MTYPE cur = static_cast<MTYPE>(d_2peel_current);
		if (V[v] != -1 || DD[v] > cur)
			return;

		V[v] = cur;
		warp_aggregated_add(&d_peeled);

		for (INCOFF_T i = O[v]; i < O[v + 1]; ++i)
		{
			const EDGE_T q = INC[i];
			if (atomicCAS(&claimed[q], 0, 1) != 0)
				continue;                     // another triangle already killed it
			const qud_t x = QUD[q];
			const VERTEX_T members[4] = {x.a, x.b, x.c, x.d};
			for (int s = 0; s < 4; ++s)
			{
				if (static_cast<size_t>(members[s]) == v)
					continue;
				const MTYPE old = atomicSub(&DD[members[s]], 1);
				if (old == cur + 1)           // just fell to the current level
					d_2peel_next = d_2peel_current;
			}
		}
	}
} // namespace nucleus34_direct

int main(int argc, char **argv)
{

	if (argc < 2)
	{
		std::cerr << "Usage: "
				  << strip_path(argv[0])
				  << " <dataset file (.mtx or ."
				  << graph_t<EDGE_T, VERTEX_T>::ext << ")>"
				  << " [parts (default 1; >1 enables the parted multi-GPU pipeline)]"
				  << " [results output file (optional)]" // Output the triangle and support vector
				  << std::endl;
		return EXIT_FAILURE;
	}

	std::string dataset_file(argv[1]);

	// argv[2]: number of parts. num_parts > 1 routes triangle and four-clique
	// counting through the parted (multi-GPU) implementations. Default = 1.
	size_t num_parts = 1;
	if (argc > 2)
	{
		num_parts = static_cast<size_t>(std::stoul(argv[2]));
		if (num_parts < 1)
			num_parts = 1;
	}

	std::string output_file;
	if (argc > 3)
	{
		output_file = argv[3];
	}
	
	std::string step("");

	SET_STEP("0.1 Environment info collection");
	auto cpu_info{cpuinfo()};
	auto gpu_info{gpuinfo()};
	auto peer_devices = cu::get_peer_devices();
	int nProcs = omp_get_num_procs();

	// Number of CUDA devices actually visible. The parted (multi-GPU) pipeline
	// only helps when more than one device is available; on a single device it
	// degrades to an even, non-memory-aware split, so we route single-device
	// runs through the single-GPU pipeline whose triangle/four-clique phases
	// carry the path-count-aware memory guardrails.
	int cuda_device_count = 0;
	cudaGetDeviceCount(&cuda_device_count);
	const bool use_parted = (num_parts > 1 && cuda_device_count > 1);

	std::ostringstream jsonSS;
	g_jsonSS = &jsonSS;

	beginObject(true, "environment");
	field(true, "app", APP_NAME);
	field(false, "version", APP_VER);
	field(false, "g++", CXX_VER);
	field(false, "cuda", CUDART_VERSION);
	field(false, "parts", num_parts);
	field(false, "slurm_requested_gpus", env("SLURM_GPUS"));
	field(false, "slurm_available_gpus", env("SLURM_VISIBLE_DEVICES"));
	field(false, "slurm_job_gpus", env("SLURM_JOB_GPUS"));
	field(false, "peer_gpu_count", peer_devices.size());
	field(false, "thrust", thrust_version_string());
	field(false, "policy", std::string(EXE_POL_STR));
	field(false, "procs", nProcs);
	field(false, "cpu_core_count", cpu_info.size());
	field(false, "cpu_affinity", cpu_affinity());
	field(false, "cpu_model", cpu_info.size() ? cpu_info[0]["model name"] : std::string("Unknown"));
	field(false, "cpu_physical_cores", cpu_info.size() ? cpu_info[0]["cpu cores"] : std::string("Unknown"));

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

		auto oriented_graph_vertex_count = oriented_graph.size_vertices();
		auto oriented_graph_edge_count = oriented_graph.size_edges();

		field(false, "vertex_count", oriented_graph_vertex_count);
		field(false, "edge_count", oriented_graph_edge_count * 2);
		{
			std::ostringstream os;
			os << std::fixed << std::setprecision(2) << t_data.count();
			raw_field(false, "data_loading_time_sec", os.str());
		}
		field(false, "data_memory_bytes", oriented_graph.bytes());

		{
			std::ostringstream os;
			os << "vertices=" << oriented_graph_vertex_count
			   << " oriented_edges=" << oriented_graph_edge_count
			   << " host_graph=" << (oriented_graph.bytes() >> 20) << " MiB"
			   << " load=" << std::fixed << std::setprecision(1) << t_data.count() << "s";
			mem_note("graph", os.str());
		}

		std::unordered_map<std::string, seconds> tms;

		SET_STEP("0.3 GPU Kernel Initialization");
		initialize_kernel<<<1, 1>>>();

		CU_ERR(cudaPeekAtLastError());
		CU_ERR(cudaDeviceSynchronize());

		SET_STEP("1.0 Allocating memory for triangle phase");
		thrust::device_vector<VERTEX_T> d_D(oriented_graph.D);
		thrust::device_vector<EDGE_T> d_O(oriented_graph.O);
		auto e_ptr = reinterpret_cast<const VERTEX_T *>(oriented_graph.E.data());
		thrust::device_vector<VERTEX_T> d_E(e_ptr, e_ptr + oriented_graph_edge_count);

		// path_count = sum of D[E[i]] = total length-two paths in the oriented graph.
		// Each edge (u,v) contributes degree(v) paths; this is the exact size needed
		// for the M (path second-edge indexes) vector and an upper bound on triangles
		// since each path can produce at most one triangle.
		// Accumulate in size_t: the per-element degrees are VERTEX_T (32-bit) but
		// their sum can exceed 2^32 on large graphs, which would overflow the
		// default VERTEX_T accumulator and under-size d_M / d_tris.
		auto path_count = thrust::reduce(
			thrust::make_permutation_iterator(d_D.begin(), d_E.begin()),
			thrust::make_permutation_iterator(d_D.begin(), d_E.end()),
			std::size_t{0}, thrust::plus<std::size_t>());

		thrust::device_vector<VERTEX_T> d_DE(oriented_graph_edge_count, 0);
		thrust::device_vector<EDGE_T> d_OE(oriented_graph_edge_count, 0);

		thrust::device_vector<EDGE_T> d_N(oriented_graph_edge_count + 1, 0);

		// Decide whether the single-shot triangle scratch (d_M + d_tris, both
		// sized to the path-count upper bound) fits in device memory. Graphs that
		// fit (the common case) allocate exactly as before, outside the timed
		// region, so their triangle runtime is unchanged. Very large graphs whose
		// scratch would exceed the budget defer to the memory-bounded segmented
		// pass instead of crashing with an out-of-memory error. The check itself
		// is a single cheap cudaMemGetInfo.
		const size_t tri_single_shot_bytes =
			path_count * sizeof(MTYPE) + path_count * sizeof(tri_t);
		size_t tri_free_bytes = 0, tri_total_bytes = 0;
		cudaMemGetInfo(&tri_free_bytes, &tri_total_bytes);
		const size_t tri_budget =
			static_cast<size_t>(static_cast<double>(tri_free_bytes) * cu::GPU_SCRATCH_MEM_SAFETY);
		const bool tri_segmented = (tri_budget > 0 && tri_single_shot_bytes > tri_budget);

		{
			std::ostringstream os;
			os << "path_count=" << path_count
			   << " single_shot=" << (tri_single_shot_bytes >> 20) << " MiB"
			   << " budget=" << (tri_budget >> 20) << " MiB"
			   << " mode=" << (tri_segmented ? "segmented" : "single-shot");
			mem_note("tri-plan", os.str());
		}

		// d_M / d_tris are sized to the path-count upper bound only on the
		// single-shot path; the segmented path sizes its own per-segment scratch.
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
			// Exact size: one entry per length-two path; pre-filled with 1 so
			// extend_edges_to_paths' internal resize() is a no-op during timing.
			d_M.assign(path_count, 1);

			// Upper bound: each length-two path yields at most one triangle.
			d_tris.resize(path_count);
		}

		// Degree and offset vector for triangle graph with size
		// equal to edge size in the base graph
		thrust::device_vector<VERTEX_T> d_trisD(oriented_graph_edge_count, 0);
		thrust::device_vector<VERTEX_T> d_trisO(oriented_graph_edge_count, 0);

		CU_ERR(cudaPeekAtLastError());
		CU_ERR(cudaDeviceSynchronize());

		SET_STEP("1.1 Triangle counting");
		size_t triangle_count = 0;
		seconds triangle_gpu_time = seconds::zero();
		if (use_parted)
		{
			// Override the nProcs split with the user-requested parts count
			// and run the multi-GPU parted triangle pipeline. The result is
			// uploaded back to d_tris so the rest of the GPU pipeline (build
			// triangles graph, peeling) continues to work unchanged.
			oriented_graph.make_split_parts(num_parts);
			auto host_tris = cu::compute_triangles_parted(oriented_graph, &triangle_gpu_time);
			triangle_count = host_tris.size();
			d_tris.resize(triangle_count);
			thrust::copy(host_tris.begin(), host_tris.end(), d_tris.begin());
		}
		else
		{
			t = hrc::now();
			if (tri_segmented)
			{
				cu::compute_triangles_segmented(d_D, d_O, d_E, num_parts, d_tris);
			}
			else
			{
				cu::compute_triangles(d_D, d_O, d_E, d_DE, d_OE, d_N, d_M, d_tris);
			}
			triangle_gpu_time = hrc::now() - t;
			cudaMemcpyFromSymbol(&triangle_count, d_global_num_triangles,
								 sizeof(d_global_num_triangles), 0,
								 cudaMemcpyDeviceToHost);
		}
		tms["Triangles"] = triangle_gpu_time;
		field(false, "triangle_time_sec", tms["Triangles"].count());

		CU_ERR(cudaPeekAtLastError());
		CU_ERR(cudaDeviceSynchronize());

		field(false, "triangle_count", triangle_count);

		{
			std::ostringstream os;
			os << "triangles=" << triangle_count
			   << " device_tris=" << ((triangle_count * sizeof(tri_t)) >> 20) << " MiB"
			   << " host_copy=" << ((triangle_count * sizeof(tri_t)) >> 20) << " MiB"
			   << " time=" << std::fixed << std::setprecision(1) << triangle_gpu_time.count() << "s";
			mem_note("tri-done", os.str());
		}

		std::vector<tri_t> triangles(triangle_count);
		thrust::copy(d_tris.cbegin(), d_tris.cbegin() + triangle_count, triangles.begin());

		// Output triangle graph to a temporary file if output file is specified.
		// The K augmentation pass later reads this temp file and writes the final result.
		if (!output_file.empty())
		{
			std::ofstream ofs(output_file + ".tmp");
			if (!ofs)
			{
				throw std::runtime_error("Failed to open temp output file for triangle dump: " + output_file + ".tmp");
			}
			for (const auto &tri : triangles)
			{
				ofs << tri << '\n';
			}
		}

		// output first PRINT_CAP elements of triangles array
		{
			auto n_dbg = std::min<std::size_t>(PRINT_CAP, triangle_count);
			array_field(false, "triangles", triangles.cbegin(), triangles.cbegin() + n_dbg,
						[](std::ostream &os, const auto &v)
						{ os << "\"" << v << "\""; });
		}

		// Early finish if there is no triangle
		if (!triangle_count)
		{
			endObject();
			lastObject();
			g_jsonSS = nullptr;
			std::cout << jsonSS.str() << std::endl;
			return EXIT_SUCCESS;
		}

		// Free base-graph vectors no longer needed after triangle counting
		thrust::device_vector<VERTEX_T>().swap(d_D);
		thrust::device_vector<EDGE_T>().swap(d_O);
		thrust::device_vector<VERTEX_T>().swap(d_E);
		thrust::device_vector<VERTEX_T>().swap(d_DE);
		thrust::device_vector<EDGE_T>().swap(d_OE);
		thrust::device_vector<EDGE_T>().swap(d_N);
		thrust::device_vector<MTYPE>().swap(d_M);
		// Shrink d_tris from path_count upper-bound to actual triangle_count
		d_tris.resize(triangle_count);
		d_tris.shrink_to_fit();

		SET_STEP("2.0 Building triangles graph");
		t = hrc::now();
		cu::build_triangles_graph(d_tris, d_trisD, d_trisO, triangle_count);
		tms["Triangles-graph"] = hrc::now() - t;
		field(false, "triangle_graph_time_sec", tms["Triangles-graph"].count());

		SET_STEP("2.1 Allocating memory for four-cliques phase");
		// 2. Four-cliques Phase
		thrust::device_vector<VERTEX_T> d_DE_b(triangle_count, 0);
		thrust::device_vector<VERTEX_T> d_DE_c(triangle_count, 0);
		thrust::device_vector<EDGE_T> d_OE_b(triangle_count, 0);
		thrust::device_vector<EDGE_T> d_OE_c(triangle_count, 0);

		thrust::device_vector<EDGE_T> d_N0(triangle_count + 1, 0);
		thrust::device_vector<EDGE_T> d_N1(triangle_count + 1, 0);

		// Sized internally by cu::compute_fourcliques based on per-component path counts.
		thrust::device_vector<MTYPE> d_M0;
		thrust::device_vector<MTYPE> d_M1;
		thrust::device_vector<qud_t> d_quds;

		// d_trisD is only read by cu_take_quds; copy to host now so the device
		// vector can be freed after the four-cliques call.
		std::vector<VERTEX_T> h_trisD(d_trisD.size());
		thrust::copy(d_trisD.begin(), d_trisD.end(), h_trisD.begin());

		SET_STEP("2.2 Four-cliques counting");
		seconds fourclique_gpu_time = seconds::zero();
		std::vector<qud_t> host_quds_parted;
		if (use_parted)
		{
			// Build a host triangle graph that matches the test/cpu code path,
			// then run the multi-GPU parted four-cliques pipeline. The
			// graph_t<U,V,2> constructor sorts the input in place, so we keep
			// the sorted ordering when handing the triangle list back to
			// compute_fourcliques_parted.
			std::vector<std::array<VERTEX_T, 3>> tris_arr(triangle_count);
			for (size_t i = 0; i < triangle_count; ++i)
			{
				tris_arr[i] = {triangles[i].a, triangles[i].b, triangles[i].c};
			}
			graph_t<EDGE_T, VERTEX_T, 2> otris(tris_arr, oriented_graph_edge_count, true);
			otris.make_split_parts(num_parts);
			std::vector<tri_t> tris_sorted(triangle_count);
			for (size_t i = 0; i < triangle_count; ++i)
			{
				tris_sorted[i] = {tris_arr[i][0], tris_arr[i][1], tris_arr[i][2]};
			}
			host_quds_parted = cu::compute_fourcliques_parted(otris, tris_sorted, &fourclique_gpu_time);
		}
		else
		{
			t = hrc::now();
			cu::compute_fourcliques(d_trisD, d_trisO, d_tris, triangle_count,
									d_DE_b, d_DE_c, d_OE_b, d_OE_c,
									d_N0, d_N1, d_M0, d_M1, d_quds);
			fourclique_gpu_time = hrc::now() - t;
		}
		tms["Four-cliques"] = fourclique_gpu_time;

		if (use_parted)
		{
			// Upload parted results so the downstream peeling pipeline can
			// consume d_quds exactly as in the single-GPU path.
			d_quds.resize(host_quds_parted.size());
			thrust::copy(host_quds_parted.begin(), host_quds_parted.end(), d_quds.begin());
		}

		// Transfer and free device vectors no longer needed after four-clique counting
		thrust::device_vector<VERTEX_T>().swap(d_trisD);
		std::vector<tri_t> h_tris(triangle_count);
		thrust::copy(d_tris.begin(), d_tris.begin() + triangle_count, h_tris.begin());
		thrust::device_vector<tri_t>().swap(d_tris);

		std::vector<EDGE_T> h_trisO(d_trisO.size());
		thrust::copy(d_trisO.begin(), d_trisO.end(), h_trisO.begin());
		thrust::device_vector<VERTEX_T>().swap(d_trisO);

		thrust::device_vector<VERTEX_T>().swap(d_DE_b);
		thrust::device_vector<VERTEX_T>().swap(d_DE_c);
		thrust::device_vector<EDGE_T>().swap(d_OE_b);
		thrust::device_vector<EDGE_T>().swap(d_OE_c);
		thrust::device_vector<EDGE_T>().swap(d_N0);
		thrust::device_vector<EDGE_T>().swap(d_N1);
		thrust::device_vector<MTYPE>().swap(d_M0);
		thrust::device_vector<MTYPE>().swap(d_M1);

		field(false, "four_clique_time_sec", tms["Four-cliques"].count());

		size_t fourclique_count = 0;
		if (use_parted)
		{
			fourclique_count = host_quds_parted.size();
		}
		else
		{
			cudaMemcpyFromSymbol(&fourclique_count, d_global_num_4cliques,
								 sizeof(d_global_num_4cliques), 0,
								 cudaMemcpyDeviceToHost);
		}

		field(false, "four_clique_count", fourclique_count);

		{
			// The device-resident peel needs, per four-clique: the qud_t itself,
			// 4 incidence ids (plus the 4 triangle keys that the sort consumes
			// before they are freed) and one claim flag. Printing the projection
			// here says up front whether step 3.0 can fit, instead of finding out
			// via a bad_alloc part-way through building the incidence lists.
			const size_t peel_bytes =
				fourclique_count * sizeof(qud_t) +
				4 * fourclique_count * sizeof(EDGE_T) +
				fourclique_count * sizeof(int) +
				// The CSR offsets, one per triangle. Previously left out of this
				// projection; now worth counting, since INCOFF_T is 8 bytes and
				// on a dense prefix the triangle count is in the hundreds of
				// millions -- a couple of GB the estimate should not hide.
				(triangle_count + 1) * sizeof(INCOFF_T);
			const size_t inc_build_peak = peel_bytes + 4 * fourclique_count * sizeof(VERTEX_T);
			std::ostringstream os;
			os << "four_cliques=" << fourclique_count
			   << " peel_resident=" << (peel_bytes >> 20) << " MiB"
			   << " inc_build_peak=" << (inc_build_peak >> 20) << " MiB"
			   << " time=" << std::fixed << std::setprecision(1) << fourclique_gpu_time.count() << "s";
			mem_note("fc-done", os.str());
		}

		// output first PRINT_CAP elements of four-clique array
		{
			auto n_dbg = std::min<std::size_t>(PRINT_CAP, fourclique_count);
			std::vector<qud_t> h_quds(n_dbg);
			thrust::copy(d_quds.cbegin(), d_quds.cbegin() + n_dbg, h_quds.begin());
			array_field(false, "four-cliques", h_quds.cbegin(), h_quds.cend(),
						[](std::ostream &os, const auto &v)
						{ os << "\"" << v << "\""; });
		}

		// Early finish if there is no four-clique
		if (!fourclique_count)
		{
			endObject();
			lastObject();
			g_jsonSS = nullptr;
			std::cout << jsonSS.str() << std::endl;
			return EXIT_SUCCESS;
		}

		// Build a DIM=1 undirected four-cliques graph from the generated
		// four-cliques, then run a single peeling pass on the full graph.
		SET_STEP("3.0 Per-triangle four-clique incidence");
		t = hrc::now();
		const size_t nFC = fourclique_count;
		// The four-clique ID stored in the incidence array is an EDGE_T, so the
		// CLIQUE COUNT (not the incidence size) is what must fit in 32 bits.
		// Fail loudly rather than wrap, as cuNucleus34_ondemand does for its
		// own index limit -- a silent wrap here returns a plausible, wrong
		// K_max with no error at all, which is how this went unnoticed.
		if (fourclique_count > static_cast<size_t>(std::numeric_limits<EDGE_T>::max()))
		{
			std::ostringstream os;
			os << "four-clique count " << fourclique_count
			   << " exceeds the 32-bit EDGE_T id range;"
			   << " rebuild with -DEDGE_UL";
			throw std::runtime_error(os.str());
		}


		// (triangle, four-clique) incidences sorted by triangle -> CSR. This
		// replaces the host-side graph_t<...,3> build entirely: no round trip
		// to the host, and 4 ids per clique instead of the 12 the flattened
		// edge list needed.
		thrust::device_vector<VERTEX_T> d_inc_tri(4 * nFC);
		thrust::device_vector<EDGE_T>   d_inc_fc(4 * nFC);
		{
			auto Q  = thrust::raw_pointer_cast(d_quds.data());
			auto IT = thrust::raw_pointer_cast(d_inc_tri.data());
			auto IF = thrust::raw_pointer_cast(d_inc_fc.data());
			thrust::for_each_n(thrust::counting_iterator<size_t>(0), nFC,
				[Q, IT, IF] __device__(size_t i) -> void {
					const qud_t x = Q[i];
					IT[4 * i + 0] = x.a; IF[4 * i + 0] = static_cast<EDGE_T>(i);
					IT[4 * i + 1] = x.b; IF[4 * i + 1] = static_cast<EDGE_T>(i);
					IT[4 * i + 2] = x.c; IF[4 * i + 2] = static_cast<EDGE_T>(i);
					IT[4 * i + 3] = x.d; IF[4 * i + 3] = static_cast<EDGE_T>(i);
				});
		}
		thrust::sort_by_key(d_inc_tri.begin(), d_inc_tri.end(), d_inc_fc.begin());

		const size_t vertex_count = triangle_count;
		thrust::device_vector<INCOFF_T> d_incO(vertex_count + 1);
		thrust::lower_bound(d_inc_tri.cbegin(), d_inc_tri.cend(),
							thrust::counting_iterator<VERTEX_T>(0),
							thrust::counting_iterator<VERTEX_T>(static_cast<VERTEX_T>(vertex_count + 1)),
							d_incO.begin());
		thrust::device_vector<VERTEX_T>().swap(d_inc_tri);

		// support = number of incident four-cliques; no 3x slot scaling
		thrust::device_vector<VERTEX_T> d_deg(vertex_count);
		thrust::device_vector<MTYPE>    d_DD(vertex_count);
		{
			auto Op = thrust::raw_pointer_cast(d_incO.data());
			auto Gp = thrust::raw_pointer_cast(d_deg.data());
			auto Dp = thrust::raw_pointer_cast(d_DD.data());
			thrust::for_each_n(thrust::counting_iterator<size_t>(0), vertex_count,
				[Op, Gp, Dp] __device__(size_t v) -> void {
					const auto d = Op[v + 1] - Op[v];
					Gp[v] = static_cast<VERTEX_T>(d);
					Dp[v] = static_cast<MTYPE>(d);
				});
		}
		CU_ERR(cudaDeviceSynchronize());
		mem_note("inc-done", "triangle->four-clique CSR built");
		tms["Four-cliques-graph"] = hrc::now() - t;
		field(false, "four_clique_graph_time_sec", tms["Four-cliques-graph"].count());

		SET_STEP("4.0 Peeling (device resident, claim based)");
		thrust::device_vector<MTYPE> d_kv(vertex_count, -1);
		thrust::device_vector<int>   d_claimed(nFC, 0);
		dim3 nPeelBlocks(blocks(vertex_count), 1, 1);

		t = hrc::now();
		cu::initialize<<<1, 1>>>(1U);
		cu::peel_zeros<<<nPeelBlocks, MAX_THRD_BLK>>>(
			thrust::raw_pointer_cast(d_kv.data()),
			thrust::raw_pointer_cast(d_deg.data()), vertex_count);
		CU_ERR(cudaPeekAtLastError());
		CU_ERR(cudaDeviceSynchronize());
		tms["Peeling-Preprocess"] = hrc::now() - t;

		unsigned peeled = 0U;
		auto nLoopCounter = 0U;
		t = hrc::now();
		while (true)
		{
			cu::update_value_to_peel<<<1, 1>>>();
			CU_ERR(cudaPeekAtLastError());

			nucleus34_direct::peel_fourcliques<<<nPeelBlocks, MAX_THRD_BLK>>>(
				thrust::raw_pointer_cast(d_incO.data()),
				thrust::raw_pointer_cast(d_inc_fc.data()),
				thrust::raw_pointer_cast(d_quds.data()),
				thrust::raw_pointer_cast(d_kv.data()),
				thrust::raw_pointer_cast(d_DD.data()),
				thrust::raw_pointer_cast(d_claimed.data()), vertex_count);
			CU_ERR(cudaPeekAtLastError());
			CU_ERR(cudaDeviceSynchronize());
			++nLoopCounter;

			cudaMemcpyFromSymbol(&peeled, d_peeled, sizeof(d_peeled), 0,
								 cudaMemcpyDeviceToHost);
			if (peeled >= vertex_count)
				break;
		}
		CU_ERR(cudaDeviceSynchronize());
		tms["Peeling"] = hrc::now() - t;

		unsigned k_max_gpu = 0U;
		cudaMemcpyFromSymbol(&k_max_gpu, d_2peel_current, sizeof(d_2peel_current), 0,
							 cudaMemcpyDeviceToHost);

		std::vector<MTYPE> h_kv(vertex_count);
		thrust::copy(d_kv.cbegin(), d_kv.cend(), h_kv.begin());

		field(false, "peeling_mode", std::string("resident-direct"));
		field(false, "peeling_preprocess_time_sec", tms["Peeling-Preprocess"].count());
		field(false, "peeling_time_sec", tms["Peeling"].count());
		field(false, "peeling_iterations", nLoopCounter);
		field(false, "K_max", k_max_gpu);

		tms["all"] = seconds::zero();
		for (const auto &tm : tms)
		{
			if (tm.first != "all")
			{
				tms["all"] += tm.second;
			}
		}
		field(false, "total_time_sec", tms["all"].count());

		SET_STEP("4.3 Finalizing results on host");

		// 3.3 Output K support Vector
		field(false, "K_count", h_kv.size());
		// output first PRINT_CAP elements of K array
		{
			auto n_dbg = std::min<std::size_t>(PRINT_CAP, h_kv.size());
			array_field(false, "K", h_kv.cbegin(), h_kv.cbegin() + n_dbg,
						[](std::ostream &os, const auto &v)
						{ os << v; });
		}

		if (!output_file.empty())
		{
			// Read the triangle rows from the temp file and write the final output with K appended.
			std::ifstream in(output_file + ".tmp");
			std::ofstream out(output_file);
			if (!in || !out)
			{
				throw std::runtime_error("Failed to open files for K augmentation: " + output_file);
			}

			std::string triangle_row;
			size_t row_index = 0;
			while (std::getline(in, triangle_row))
			{
				out << triangle_row;
				if (row_index < h_kv.size())
				{
					out << '\t' << h_kv[row_index];
				}
				out << '\n';
				++row_index;
			}

			if (row_index != h_kv.size())
			{
				throw std::runtime_error("Triangle row count does not match K size while updating output file: " + output_file);
			}

			in.close();
			out.close();
			if (std::remove((output_file + ".tmp").c_str()) != 0)
			{
				throw std::runtime_error("Failed to remove temp output file after appending K values: " + output_file + ".tmp");
			}
		}

		// Accumulate in long long: without an explicit init, std::reduce deduces
		// the accumulator from the iterator (MTYPE = int), which overflows once
		// the triangle count times the mean peel value passes 2^31.
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
			{
				std::rethrow_exception(expPtr);
			}
		}
		catch (const std::exception &e)
		{
			std::cerr
				<< "Dataset: " << dataset_file << std::endl
				<< "Step: " << step << std::endl
				<< "Error: " << e.what() << std::endl
				<< "NVIDIA SMI: " << std::endl
				<< cu::nvidiaSmi() << std::endl;

			// Output collected information with error message if possible
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
