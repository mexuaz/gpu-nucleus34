#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <sstream>

#include <omp.h>

#include <thrust/iterator/permutation_iterator.h>

#include "utility/def_system.hpp"
#include "utility/cpuinfo.hpp"
#include "graph/graph.hpp"
#include "utility/ioutils.hpp"
#include "utility/json.h"
#include "utils.cuh"

using namespace json;

#define APP_NAME "cuda-nucleus34"
#define APP_VER "0.9.2"

int main(int argc, char** argv) {

	if(argc < 2) {
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
	std::string step ("");

	// argv[2]: number of parts. parts > 1 routes triangle and four-clique
	// counting through the parted (multi-GPU) implementations. Default = 1.
	size_t parts = 1;
	if (argc > 2) {
		try { parts = std::stoul(argv[2]); } catch (...) { parts = 1; }
		if (parts < 1) parts = 1;
	}

	std::string output_file;
	if (argc > 3)
	{
		output_file = argv[3];
	}
	
	step = "0.1 Environment info collection";
	auto cpu_info {cpuinfo()};
	auto gpu_info {gpuinfo()};
	auto peer_devices = cu::get_peer_devices();
	int nProcs = omp_get_num_procs();

	std::ostringstream jsonSS;
	g_jsonSS = &jsonSS;

	beginObject(true, "environment");
	field(true,  "app", APP_NAME);
	field(false, "version", APP_VER);
	field(false, "g++", CXX_VER);
	field(false, "cuda", CUDART_VERSION);
	field(false, "parts", parts);
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

	for (const auto& gi : gpu_info) {
		for (const auto& p : gi) {
			std::string gpu_key = "gpu_" + p.first;
			std::replace(gpu_key.begin(), gpu_key.end(), ' ', '_');
			field(false, gpu_key, p.second);
		}
	}

	field(false, "host", hostname());
	endObject();

	beginObject(false, "dataset");
	field(true,  "file", strip_path(dataset_file));

	try{
		step = "0.2 Data loading and graph construction";
		graph_t<EDGE_T, VERTEX_T> oriented_graph;
		auto ext = get_ext(dataset_file);
		auto t = hrc::now();
		if( ext == ".mtx" ) {
			oriented_graph.from_edges(dataset_file, true);
			oriented_graph.make_oriented();
		} else if ( ext == ".edges" || ext == ".txt" ) {
			oriented_graph.from_edges(dataset_file, false);
			oriented_graph.make_oriented();
		} else if( ext == graph_t<EDGE_T, VERTEX_T>::ext ) {
			oriented_graph.deserialize(dataset_file);
			oriented_graph.build_offset();
		} else {
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
			   
		std::unordered_map<std::string, seconds> tms;
		
		step = "0.3 GPU Kernel Initialization";
		initialize_kernel<<<1, 1>>>();

		CU_ERR( cudaPeekAtLastError() );
		CU_ERR( cudaDeviceSynchronize() );

		step = "1.0 Allocating memory for triangle phase";
		thrust::device_vector<VERTEX_T> d_D(oriented_graph.D);
		thrust::device_vector<EDGE_T> d_O(oriented_graph.O);
		auto e_ptr = reinterpret_cast<const VERTEX_T*>(oriented_graph.E.data());
		thrust::device_vector<VERTEX_T> d_E(e_ptr, e_ptr + oriented_graph_edge_count);

		// path_count = sum of D[E[i]] = total length-two paths in the oriented graph.
		// Each edge (u,v) contributes degree(v) paths; this is the exact size needed
		// for the M (path second-edge indexes) vector and an upper bound on triangles
		// since each path can produce at most one triangle.
		auto path_count = static_cast<size_t>(thrust::reduce(
			thrust::make_permutation_iterator(d_D.begin(), d_E.begin()),
			thrust::make_permutation_iterator(d_D.begin(), d_E.end())));

		thrust::device_vector<VERTEX_T> d_DE(oriented_graph_edge_count, 0);
		thrust::device_vector<EDGE_T> d_OE(oriented_graph_edge_count, 0);

		thrust::device_vector<EDGE_T> d_N(oriented_graph_edge_count+1, 0);

		// Exact size: one entry per length-two path; pre-filled with 1 so
		// extend_edges_to_paths' internal resize() is a no-op during timing.
		thrust::device_vector<MTYPE> d_M(path_count, 1);

		// Upper bound: each length-two path yields at most one triangle.
		thrust::device_vector<tri_t> d_tris(path_count);

		// Degree and offset vector for triangle graph with size
		// equal to edge size in the base graph
		thrust::device_vector<VERTEX_T> d_trisD(oriented_graph_edge_count, 0);
		thrust::device_vector<VERTEX_T> d_trisO(oriented_graph_edge_count, 0);

		CU_ERR( cudaPeekAtLastError() );
		CU_ERR( cudaDeviceSynchronize() );
		
		step = "1.1 Triangle counting";
		size_t triangle_count = 0;
		t = hrc::now();
		if (parts > 1) {
			// Override the nProcs split with the user-requested parts count
			// and run the multi-GPU parted triangle pipeline. The result is
			// uploaded back to d_tris so the rest of the GPU pipeline (build
			// triangles graph, peeling) continues to work unchanged.
			oriented_graph.make_split_parts(parts);
			auto host_tris = cu::compute_triangles_parted( oriented_graph );
			triangle_count = host_tris.size();
			d_tris.resize(triangle_count);
			thrust::copy(host_tris.begin(), host_tris.end(), d_tris.begin());
		} else {
			cu::compute_triangles( d_D, d_O, d_E, d_DE, d_OE, d_N, d_M, d_tris );
			cudaMemcpyFromSymbol(&triangle_count, d_global_num_triangles,
				sizeof(d_global_num_triangles), 0,
				cudaMemcpyDeviceToHost);
		}
		tms["Triangles"] = hrc::now() - t;
		field(false, "triangle_time_sec", tms["Triangles"].count());

		CU_ERR( cudaPeekAtLastError() );
		CU_ERR( cudaDeviceSynchronize() );

		field(false, "triangle_count", triangle_count);

		std::vector<tri_t> triangles(triangle_count);
		thrust::copy(d_tris.cbegin(), d_tris.cbegin() + triangle_count, triangles.begin());
		
		// Output triangle graph to a temporary file if output file is specified.
		// The K augmentation pass later reads this temp file and writes the final result.
		if (!output_file.empty()) {
			std::ofstream ofs(output_file + ".tmp");
			if (!ofs) {
				throw std::runtime_error("Failed to open temp output file for triangle dump: " + output_file + ".tmp");
			}
			for (const auto& tri : triangles) {
				ofs << tri << '\n';
			}
		}

		// output first PRINT_CAP elements of triangles array
		{
			auto n_dbg = std::min<std::size_t>(PRINT_CAP, triangle_count);
			array_field(false, "triangles", triangles.cbegin(), triangles.cbegin() + n_dbg,
				[](std::ostream& os, const auto& v) { os << "\"" << v << "\""; });
		}

		// Early finish if there is no triangle
		if (!triangle_count) {
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
	
		step = "2.0 Building triangles graph";
		t = hrc::now();
        cu::build_triangles_graph( d_tris, d_trisD, d_trisO, triangle_count );
		tms["Triangles-graph"] = hrc::now() - t;
		field(false, "triangle_graph_time_sec", tms["Triangles-graph"].count());

		step = "2.1 Allocating memory for four-cliques phase";
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

		step = "2.2 Four-cliques counting";
		t = hrc::now();
		std::vector<qud_t> host_quds_parted;
		if (parts > 1) {
			// Build a host triangle graph that matches the test/cpu code path,
			// then run the multi-GPU parted four-cliques pipeline. The
			// graph_t<U,V,2> constructor sorts the input in place, so we keep
			// the sorted ordering when handing the triangle list back to
			// compute_fourcliques_parted.
			std::vector<std::array<VERTEX_T, 3>> tris_arr(triangle_count);
			for (size_t i = 0; i < triangle_count; ++i) {
				tris_arr[i] = { triangles[i].a, triangles[i].b, triangles[i].c };
			}
			graph_t<EDGE_T, VERTEX_T, 2> otris(tris_arr, oriented_graph_edge_count, true);
			otris.make_split_parts(parts);
			std::vector<tri_t> tris_sorted(triangle_count);
			for (size_t i = 0; i < triangle_count; ++i) {
				tris_sorted[i] = { tris_arr[i][0], tris_arr[i][1], tris_arr[i][2] };
			}
			host_quds_parted = cu::compute_fourcliques_parted( otris, tris_sorted );
		} else {
			cu::compute_fourcliques( d_trisD, d_trisO, d_tris, triangle_count,
									 d_DE_b, d_DE_c, d_OE_b, d_OE_c,
									 d_N0, d_N1, d_M0, d_M1, d_quds );
		}
		tms["Four-cliques"] = hrc::now() - t;

		if (parts > 1) {
			// Upload parted results so the downstream peeling pipeline can
			// consume d_quds exactly as in the single-GPU path.
			d_quds.resize(host_quds_parted.size());
			thrust::copy(host_quds_parted.begin(), host_quds_parted.end(), d_quds.begin());
		}

		// Transfer and free device vectors no longer needed after four-clique counting
		thrust::device_vector<VERTEX_T>().swap(d_trisD);
		std::vector<tri_t> h_tris(triangle_count);
		thrust::copy(d_tris.begin(), d_tris.begin()+triangle_count, h_tris.begin());
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
		if (parts > 1) {
			fourclique_count = host_quds_parted.size();
		} else {
			cudaMemcpyFromSymbol(&fourclique_count, d_global_num_4cliques,
								 sizeof(d_global_num_4cliques), 0,
								 cudaMemcpyDeviceToHost);
		}

		field(false, "four_clique_count", fourclique_count);

		// output first PRINT_CAP elements of four-clique array
		{
			auto n_dbg = std::min<std::size_t>(PRINT_CAP, fourclique_count);
			std::vector<qud_t> h_quds(n_dbg);
			thrust::copy(d_quds.cbegin(), d_quds.cbegin() + n_dbg, h_quds.begin());
			array_field(false, "four-cliques", h_quds.cbegin(), h_quds.cend(),
				[](std::ostream& os, const auto& v) { os << "\"" << v << "\""; });
		}
		
		// Early finish if there is no four-clique
		if (!fourclique_count) {
			endObject();
			lastObject();
			g_jsonSS = nullptr;
			std::cout << jsonSS.str() << std::endl;
			return EXIT_SUCCESS;
		}
		
		// Degree and Offset vectors for four-cliques undirected-graph with vertex size equal to triangle_count and edge size equal to fourclique_count * 4 (each FC can produce up to 4 undirected edges in the FC graph) or fourclique_count * 12 (each FC can produce up to 12 undirected edges in the FC graph if we consider all possible edges between the 4 triangles in a four-clique).

		step = "3.0 Allocating memory for four-cliques graph construction";
		thrust::device_vector<VERTEX_T> d_qudsD(triangle_count, 0);
		thrust::device_vector<VERTEX_T> d_qudsO(triangle_count, 0);
			
#if(0)		
		// Number of threads per block
		dim3 nBlocksFroucliqueEdgeCount (blocks(fourclique_count*4), 1, 1);
		t = hrc::now();
		cu_quds_to_ugraph<<<nBlocks, MAX_THRD_BLK>>>(
				thrust::raw_pointer_cast(d_quds.data()),
				fourclique_count);

		thrust::sort(d_quds.begin(), d_quds.begin()+fourclique_count * 4);

		cu_build_qudsD<<<nBlocks, MAX_THRD_BLK>>>(
				thrust::raw_pointer_cast(d_quds.data()),
					fourclique_count*4,
					thrust::raw_pointer_cast(d_qudsD.data()));

		thrust::exclusive_scan(d_qudsD.begin(),
							d_qudsD.end(),
							d_qudsO.begin(),
								0);
		tms["Four-cliques-graph"] = hrc::now() - t;
#else
		decltype(fourclique_count) uquds_id = fourclique_count*12;
		// Exact size: each four-clique produces 12 undirected dbl_t edges.
		thrust::device_vector<dbl_t> d_uquds(uquds_id);
		// Number of threads per block
		dim3 nBlocksFroucliqueEdgeCount (blocks(uquds_id), 1, 1);
		step = "3.1 Building four-cliques undirected graph";
		t = hrc::now();

		cu_quds_to_ugraph2<<<nBlocksFroucliqueEdgeCount, MAX_THRD_BLK>>>(
				thrust::raw_pointer_cast(d_quds.data()),
				thrust::raw_pointer_cast(d_uquds.data()),
				fourclique_count);

		thrust::sort(d_uquds.begin(), d_uquds.begin()+uquds_id);

		cu_build_dblsD<<<nBlocksFroucliqueEdgeCount, MAX_THRD_BLK>>>(
				thrust::raw_pointer_cast(d_uquds.data()),
					uquds_id,
					thrust::raw_pointer_cast(d_qudsD.data()));

		thrust::exclusive_scan(d_qudsD.begin(),
							d_qudsD.end(),
							d_qudsO.begin(),
								0);
		tms["Four-cliques-graph"] = hrc::now() - t;
#endif
		CU_ERR( cudaPeekAtLastError() );
		CU_ERR( cudaDeviceSynchronize() );

		field(false, "four_clique_graph_time_sec", tms["Four-cliques-graph"].count());

		// d_quds last used in cu_quds_to_ugraph2; transfer to host and free
		std::vector<qud_t> h_quds(fourclique_count);
		thrust::copy(d_quds.begin(), d_quds.begin()+fourclique_count, h_quds.begin());
		thrust::device_vector<qud_t>().swap(d_quds);

		step = "4.0 Allocating memory for peeling phase";
		// Allocating memory for peeling phase and transferring data to device
		thrust::device_vector<MTYPE> d_qudsDD(triangle_count, 0);
		thrust::copy(d_qudsD.cbegin(), d_qudsD.cend(), d_qudsDD.begin());
		//using namespace thrust::placeholders;
		//thrust::transform(d_qudsDD.begin(), d_qudsDD.end(), d_qudsDD.begin(), NUCLEUS34_FACTOR * _1);
		thrust::device_vector<MTYPE> d_kv(triangle_count, -1);
	
		
		field(false, "peeling_blocks_x", nBlocksFroucliqueEdgeCount.x);

		step = "4.1 Peeling preprocess"; // Similar to bucket construction in CPU
		t = hrc::now();
		cu::initialize<<<1, 1>>>();
		cu::peel_zeros<<<nBlocksFroucliqueEdgeCount, MAX_THRD_BLK>>>(
			thrust::raw_pointer_cast(d_kv.data()),
			thrust::raw_pointer_cast(d_qudsD.data()),
			triangle_count);
		CU_ERR( cudaPeekAtLastError() );
		CU_ERR( cudaDeviceSynchronize() );
		tms["Peeling-Preprocess"] = hrc::now() - t;
		field(false, "peeling_preprocess_time_sec", tms["Peeling-Preprocess"].count());
		
	
		step = "4.2 Peeling iterations";
		auto nLoopCounter = 0U;
		unsigned peeled = 0U;

		t = hrc::now();
		while(uquds_id) { // Do the peeling if there are FCs

			nLoopCounter++;

			cu::update_value_to_peel<<<1, 1>>>();
			CU_ERR( cudaPeekAtLastError() );
			CU_ERR( cudaDeviceSynchronize() );
			
			cu::nucleus34::peeling2<<<nBlocksFroucliqueEdgeCount, MAX_THRD_BLK>>>(
					//thrust::raw_pointer_cast(d_quds.data()),
					thrust::raw_pointer_cast(d_uquds.data()),
					thrust::raw_pointer_cast(d_kv.data()),
					thrust::raw_pointer_cast(d_qudsD.data()),
					thrust::raw_pointer_cast(d_qudsDD.data()),
					thrust::raw_pointer_cast(d_qudsO.data()),
					triangle_count);
			CU_ERR( cudaPeekAtLastError() );
			CU_ERR( cudaDeviceSynchronize() );
		
			cudaMemcpyFromSymbol(&peeled,
							d_peeled,
							sizeof(d_peeled), 0,
							cudaMemcpyDeviceToHost);


			if(peeled >= triangle_count){
				break;
			}
		} // End of while loop
		
		CU_ERR( cudaPeekAtLastError() );
		CU_ERR( cudaDeviceSynchronize() );

		tms["Peeling"] = hrc::now() - t;
		field(false, "peeling_iterations", nLoopCounter);
		field(false, "peeling_time_sec", tms["Peeling"].count());


		// d_2peel_current is defined in header and updated during kernel execution of peeling2
		unsigned k_max_gpu;
		cudaMemcpyFromSymbol(&k_max_gpu, d_2peel_current, sizeof(d_2peel_current),
			0, cudaMemcpyDeviceToHost);
		k_max_gpu /= 3;

		field(false, "K_max", k_max_gpu);

		tms["all"] = seconds::zero();
		for (const auto& tm: tms) {
			if(tm.first!="all") {
				tms["all"] += tm.second;
			}
		}
		field(false, "total_time_sec", tms["all"].count());

		step = "4.3 Transferring remaining results back to host";
		std::vector<VERTEX_T> h_qudsD(d_qudsD.size());
		thrust::copy(d_qudsD.begin(), d_qudsD.end(), h_qudsD.begin());

		std::vector<EDGE_T> h_qudsO(d_qudsO.size());
		thrust::copy(d_qudsO.begin(), d_qudsO.end(), h_qudsO.begin());

		std::vector<MTYPE> h_kv(d_kv.size());
		thrust::copy(d_kv.cbegin(), d_kv.cend(), h_kv.begin());

		// Free remaining device vectors after peeling
		thrust::device_vector<dbl_t>().swap(d_uquds);
		thrust::device_vector<VERTEX_T>().swap(d_qudsD);
		thrust::device_vector<VERTEX_T>().swap(d_qudsO);
		thrust::device_vector<MTYPE>().swap(d_qudsDD);
		thrust::device_vector<MTYPE>().swap(d_kv);
		
		using namespace thrust::placeholders;
		thrust::transform(h_kv.begin(), h_kv.end(), h_kv.begin(), _1 / 3);
	
		// 3.3 Output K support Vector
		field(false, "K_count", h_kv.size());
		// output first PRINT_CAP elements of K array
		{
			auto n_dbg = std::min<std::size_t>(PRINT_CAP, h_kv.size());
			array_field(false, "K", h_kv.cbegin(), h_kv.cbegin() + n_dbg,
				[](std::ostream& os, const auto& v) { os << v; });
		}

		if(!output_file.empty()) {
			// Read the triangle rows from the temp file and write the final output with K appended.
			std::ifstream in(output_file + ".tmp");
			std::ofstream out(output_file);
			if (!in || !out) {
				throw std::runtime_error("Failed to open files for K augmentation: " + output_file);
			}

			std::string triangle_row;
			size_t row_index = 0;
			while (std::getline(in, triangle_row)) {
				out << triangle_row;
				if (row_index < h_kv.size()) {
					out << '\t' << h_kv[row_index];
				}
				out << '\n';
				++row_index;
			}

			if (row_index != h_kv.size()) {
				throw std::runtime_error("Triangle row count does not match K size while updating output file: " + output_file);
			}

			in.close();
			out.close();
			if (std::remove((output_file + ".tmp").c_str()) != 0) {
				throw std::runtime_error("Failed to remove temp output file after appending K values: " + output_file + ".tmp");
			}
		}

		auto k_avg_gpu = static_cast<double>(
				std::reduce(EXE_POL, h_kv.cbegin(), h_kv.cend()))
				/static_cast<double>(h_kv.size());
		
		field(false, "K_avg", k_avg_gpu);

		endObject();
		lastObject();
		g_jsonSS = nullptr;
		std::cout << jsonSS.str() << std::endl;
	}
	catch(thrust::system_error &e) {
    	std::cerr << "Allocation failed: " << e.what() << std::endl;
	}
	catch(...) {
		auto expPtr = std::current_exception();
		try {
			if(expPtr) {
				std::rethrow_exception(expPtr);
			}
		} catch(const std::exception& e) {
			std::cerr 
			<< "Dataset: " << dataset_file << std::endl
<< "Step: " << step << std::endl
			<< "Error: " <<
			e.what() << std::endl
			<< "NVIDIA SMI: " << std::endl
			<< cu::nvidiaSmi() << std::endl;

			// Output collected information with error message if possible
			if (!jsonSS.str().empty()) {
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
