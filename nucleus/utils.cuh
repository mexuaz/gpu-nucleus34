#include <map>

#include "utility/defs.hpp"
#include "utility/def_ver.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h> // max_element
#include <thrust/remove.h> // remove remove_if
#include <thrust/sort.h> // sort
#include <thrust/copy.h> // copy
#include <thrust/count.h> // count_if
#include <mutex>
#include <thread>

#define NUCLEUS12_FACTOR 1
#define NUCLEUS23_FACTOR 2
#define NUCLEUS34_FACTOR 1

// Maximum thread per block in cuda
#define MAX_THRD_BLK 1024

#define CU_ERR(err_no) { cu_error((err_no), __FILE__, __LINE__); }

inline void cu_error(cudaError_t code,
	const char *file,
	int line,
	bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr, "CUDA Error: %s %s %d\n",
	  	cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


template <typename T>
inline int blocks(const T& nThreads) {
	if(!nThreads) {
		return 1; // This function shouldn't return zero
	}
	return static_cast<int>(nThreads/MAX_THRD_BLK) +
		   static_cast<int>
		   (static_cast<int>(nThreads)%MAX_THRD_BLK > 0);
}

struct dbl_t {
	VERTEX_T a = 0;
	VERTEX_T b = 0;
	__host__ __device__ bool operator<(const dbl_t& t) const {
		return a < t.a || (a == t.a && b < t.b);
	}
};

struct tri_t {
	VERTEX_T a = 0;
	VERTEX_T b = 0;
	VERTEX_T c = 0;
	__host__ __device__ bool operator<(const tri_t& t) const {
		return a < t.a || (a == t.a && b < t.b);
	}
};

struct qud_t {
	VERTEX_T a = 0;
	VERTEX_T b = 0;
	VERTEX_T c = 0;
	VERTEX_T d = 0;
	__host__ __device__ bool operator<(const qud_t& q) const {
		return a < q.a || (a == q.a && b < q.b);
	}
};

__device__ unsigned d_global_num_triangles;
__device__ unsigned d_global_num_4cliques;
__device__ unsigned d_2peel_current;
__device__ unsigned d_2peel_next;
__device__ unsigned d_peeled;

template <typename T>
std::ostream& operator<< (std::ostream& out,
			  const thrust::device_vector<T>& v) {
    out << "[ ";
    thrust::copy (v.cbegin(), v.cend(), std::ostream_iterator<T>(out, " "));
    out << ']';
  return out;
}

std::ostream& operator<< (std::ostream& out, const dim3& d) {
    out << "("	<< d.x << "," << d.y << "," << d.z <<")";
  return out;
}

std::ostream& operator<< (std::ostream& out, const tri_t& t) {
    out << "("	<< t.a << "," << t.b << "," << t.c <<")";
  return out;
}

std::ostream& operator<< (std::ostream& out, const qud_t& q) {
    out << "("	<< q.a << "," << q.b << ","
		<< q.c << "," << q.d <<")";
  return out;
}

__global__
void initialize_kernel() {
	d_global_num_triangles = 0;
	d_global_num_4cliques = 0;
}

/**
 * Materialises the CSR graph data for the endpoint of each edge.
 * 
 * Consider an edge e=(u,v). The thread handling edge e will
 * materialise the degree and offset for v at a unique index corresponding to the thread/edge id.
 * For example, in a graph 0<-1<-2, vertex_degrees would be [0,1,1],
 * neighbour_offsets would be [0,0,1] and edge_destinations would be [0,1].
 * The result of this would also be of length E and would produce for [(1,0), (2,1)]
 * degree vector [0,1] and offset vector [0,0].
 * 
 * @TODO Store these results in shared memory instead?
 */ 
__global__
void cuTake( VERTEX_T const* vertex_degrees
           , EDGE_T   const* neighbour_offsets
           , VERTEX_T const* edge_destinations
           , VERTEX_T      * edge_endpoint_degrees
           , EDGE_T        * edge_endpoint_offsets
           , std::size_t num_edges )
{
	auto const edge = blockIdx.x * blockDim.x + threadIdx.x;

	if( edge < num_edges )
    {
        edge_endpoint_degrees[ edge ] =    vertex_degrees[ edge_destinations[ edge ] ];
		edge_endpoint_offsets[ edge ] = neighbour_offsets[ edge_destinations[ edge ] ];
	}
}

/**
 * @brief Builds count vector D out of values of vector v
 * @param v Input vector
 * @param v_sz size of input vector
 * @param D must have a length equal to maximum value of vector v
 */
 
__global__
void cu_build_D(
		VERTEX_T* v,
		size_t v_sz,
		VERTEX_T* D) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < v_sz) {
		auto& d = D[v[i]];
		atomicAdd(&d, 1);
	}
}

__global__
void cu_build_dblsD(
		dbl_t* dbls,
		size_t dbls_sz,
		VERTEX_T* dblsD) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < dbls_sz) {
		auto& v = dblsD[dbls[i].a];
		atomicAdd(&v, 1);
	}
}

/**
 * Calculates the degree of every source vertex in the triangles graph
 */
__global__
void cu_build_trisD( tri_t const * tris
                   , std::size_t   num_triangles
                   , VERTEX_T    * trisD )
{
	auto triangle_id = static_cast< std::size_t >( blockIdx.x * blockDim.x + threadIdx.x );
	if( triangle_id < num_triangles )
    {
        VERTEX_T const first_vertex_of_triangle = tris[ triangle_id ].a; 
		auto ptr_num_triangles_starting_with_this_vertex = trisD + first_vertex_of_triangle;
		atomicAdd( ptr_num_triangles_starting_with_this_vertex, 1 );
	}
}

__global__
void cu_build_qudsD(
		qud_t* quds,
		size_t quds_sz,
		VERTEX_T* qudsD) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < quds_sz) {
		auto& v = qudsD[quds[i].a];
		atomicAdd(&v, 1);
	}
}

__device__
auto get_range_from_offsets( VERTEX_T vertex_id, EDGE_T * offset_array, std::size_t num_offsets, std::size_t max_offset )
	-> std::pair< EDGE_T, EDGE_T >
{
	EDGE_T const start = offset_array[ vertex_id ];
	EDGE_T const end   = vertex_id < num_offsets - 1 ? offset_array[ vertex_id + 1 ] : max_offset;
	return std::make_pair( start, end );
}

/**
 * Determines all common elements in [list1_start, list2_end) and
 * [list2_start, list2_end) after applying one level of indirection
 * to elements in the second list. Appends results to a
 * globally shared list using a global, atomic tail counter.
 * 
 * @note Result produced by this thread are not guaranteed to be
 * contiguous in the output array
 * @precondition: both lists must be sorted
 * @complexity: linear in the length of the largest list
 */
__device__
void intersect_lists( 
	  VERTEX_T * list1_start
	, VERTEX_T * list1_end
	, MTYPE    * list2_start
	, MTYPE    * list2_end
	, VERTEX_T * list2_indirection_array // to retrieve destination given edge index !!! 
	, tri_t   && new_triangle
	, tri_t    * output_array
	, std::size_t max_output_capacity )
{
	// uses standard "zipper" algorithm for intersecting two lists
	// Loop through, advancing the pointer that points to the smallest element
    // Lists must be sorted in advance!
	while( list1_start < list1_end && list2_start < list2_end )
	{
		const auto p = *list1_start;
		const auto o = static_cast< VERTEX_T >( *list2_start );
		const auto q = list2_indirection_array[ o ];               // NON-LOCAL READ, BUT TO IMMUTABLE !!!

		if( p < q )
		{
			++list1_start;
		}
		else
		{
			if( p == q ) // Found match!
			{
				// a = i1: the actual matching edge index in list1 (mirrors CPU {i1, d, o})
				new_triangle.a = static_cast< VERTEX_T >( list1_start - list2_indirection_array );
				new_triangle.c = o; // complete triangle with the common neighbour
				auto const output_array_tail = atomicAdd( &d_global_num_triangles, 1ul );
				if( output_array_tail < max_output_capacity )
				{
					output_array[ output_array_tail ] = new_triangle;
				}
                // else we didn't allocate enough space and the answer is wrong!
				++list1_start; // always advance on match, mirrors CPU version
			}
			++list2_start;
		}
	} // end of intersection loop
}

/**
 * Calculates all triangles in the graph from CSR representations
 * of the graph and of length-two paths in the graph and stores the
 * result in `tris`
 * 
 * [neighbour_offsets, edge_destinations] is the CSR representation of the original graph;
 * [path_offsets, path_destinations] is CSR of edges (u,v) represented by v and all
 * nodes w, u=/=w, connected to v.
 */
__global__
void cu_cliques_tris( VERTEX_T  * edge_destinations // formerly "E"
					, EDGE_T    * neighbour_offsets // formerly "O"
					, MTYPE     * path_destinations // formerly "M"  <---- no, index of edge!!!
					, EDGE_T    * path_offsets      // formerly "N"
					, std::size_t num_edges         // formerly "ESize"
					, std::size_t num_vertices      // formerly "OSize"
					, tri_t     * tris              // vector to store triangles results
					, std::size_t tris_capacity     // max capacity of triangles vector
					)
{
	auto const vertex_id = static_cast< VERTEX_T >( blockIdx.x * blockDim.x + threadIdx.x );

	if( vertex_id < num_vertices )
	{
		// If output is over-capacity already, sol'n is likely incorrect anyway! So, this condition
		// has been eliminated, even if it could save some unnecessary work.
		// Note that it was a race condition previously and that this check has
		// been moved to after snapshotting a thread-local write index in tris to fix the race condition.
		// if( d_global_num_triangles < tris_capacity );

		auto const [ first_neighbour, last_neighbour ] = get_range_from_offsets( vertex_id
																			   , neighbour_offsets
                                                                               , num_vertices
                                                                               , num_edges );

        // iterate edges originating from vertex_id!!!
		for( auto neighbour = first_neighbour; neighbour < last_neighbour; ++neighbour )
		{
			intersect_lists( edge_destinations + first_neighbour
						   , edge_destinations + last_neighbour
						   , path_destinations + path_offsets[ neighbour ]
						   , path_destinations + path_offsets[ neighbour + 1 ] // no boundary case here like before??
						   , edge_destinations
						   , tri_t{ first_neighbour, neighbour, 0 /* overwritten with common neighbours */ }
						   , tris
						   , tris_capacity );
		} // end of for-loop
	}
}

/**
 * Segmented version of cu_cliques_tris.
 *
 * Processes only vertices in [vertex_bg, vertex_ed). The path arrays
 * (path_destinations / path_offsets) are sized for the local edge slice
 * [dst_bg, O[vertex_ed]) and are indexed by (global_edge - dst_bg).
 *
 * This is the GPU analogue of the per-thread work performed inside
 * cliques_tri_par in cliques.hpp: the graph is split by split_parts and
 * each segment runs on a different GPU; results are gathered on the host.
 */
__global__
void cu_cliques_tris_seg( VERTEX_T  * edge_destinations
                        , EDGE_T    * neighbour_offsets
                        , MTYPE     * path_destinations
                        , EDGE_T    * path_offsets       // size: (dst_ed - dst_bg + 1)
                        , std::size_t num_edges
                        , std::size_t num_vertices
                        , std::size_t vertex_bg
                        , std::size_t vertex_ed
                        , std::size_t dst_bg             // = neighbour_offsets[vertex_bg]
                        , tri_t     * tris
                        , std::size_t tris_capacity )
{
	auto const tid = static_cast< std::size_t >( blockIdx.x * blockDim.x + threadIdx.x );
	auto const vertex_id = static_cast< VERTEX_T >( vertex_bg + tid );

	if( vertex_id < vertex_ed )
	{
		auto const [ first_neighbour, last_neighbour ] = get_range_from_offsets( vertex_id
		                                                                       , neighbour_offsets
		                                                                       , num_vertices
		                                                                       , num_edges );

		for( auto neighbour = first_neighbour; neighbour < last_neighbour; ++neighbour )
		{
			auto const local_n = neighbour - dst_bg;
			intersect_lists( edge_destinations + first_neighbour
			               , edge_destinations + last_neighbour
			               , path_destinations + path_offsets[ local_n ]
			               , path_destinations + path_offsets[ local_n + 1 ]
			               , edge_destinations
			               , tri_t{ first_neighbour, neighbour, 0 }
			               , tris
			               , tris_capacity );
		}
	}
}

__global__
void cu_take_quds(
		VERTEX_T* D,
		EDGE_T* O,
		tri_t* tris,
		unsigned tris_sz,
		VERTEX_T* DE_b,
		VERTEX_T* DE_c,
		EDGE_T* OE_b,
		EDGE_T* OE_c) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < tris_sz) {
		DE_b[i] = D[tris[i].b];
		OE_b[i] = O[tris[i].b];
		DE_c[i] = D[tris[i].c];
		OE_c[i] = O[tris[i].c];
	}
}

namespace create_paths {

/**
 * Populates an array of offsets given a list of sizes
 * 
 * @pre The first element of path_endpoint_offsets must be zero
 * @pre path_endpoint_offsets must be allocated more spaced than none-zero elements in edge_endpoint_degrees
 * path_endpoint_offsets must have a length of at least |non-zero elments of edge_endpoint_degrees.size()| + 1
 */
__host__
void populate_path_offsets( thrust::device_vector<VERTEX_T> const& edge_endpoint_degrees
                          , thrust::device_vector<EDGE_T>        & path_endpoint_offsets )
{
	// using thrust materizlize a vector that only has none-zero elements of  edge_endpoint_degrees and then run an inclusive scan to populate path_endpoint_offsets
    thrust::inclusive_scan( edge_endpoint_degrees.cbegin()
                          , edge_endpoint_degrees.cend()
                          , path_endpoint_offsets.begin() + 1u
                          , thrust::plus<EDGE_T>() );
}


__global__
void set_negated_first_instances( VERTEX_T const* edge_endpoint_degrees
								, EDGE_T   const* path_offsets
                                , EDGE_T   const* edge_endpoint_offsets
                                , INDEX_T       * path_final_edges
                                , std::size_t     num_edges )
{
    auto const edge = blockIdx.x * blockDim.x + threadIdx.x;

	if ( edge >= num_edges )
	{
		return;
	}

	if (edge == 0)
	{
		path_final_edges[ path_offsets[ edge ] ] = edge_endpoint_offsets[ edge ]; // + 1 - 0, but that plus one is not needed because there is no previous edge
		return;
	}

	INDEX_T result_from_previous_edge = edge_endpoint_offsets[ edge - 1 ] + edge_endpoint_degrees[ edge - 1 ];
	path_final_edges[ path_offsets[ edge ] ] = edge_endpoint_offsets[ edge ] + 1 - result_from_previous_edge;
	// WHY THE PLUS ONE???
}

/**
 * Completes every path by writing the index of the second edge in the path into path_final_edges
 */
__host__
void extend_edges_to_paths( thrust::device_vector<VERTEX_T> const& edge_endpoint_degrees
                          , thrust::device_vector<EDGE_T>   const& edge_endpoint_offsets
                          , thrust::device_vector<EDGE_T>   const& path_offsets
                          , thrust::device_vector<MTYPE>         & path_second_edges
                          )
{
    // Algorithm goes in three steps:
    // a) set all values to 1 (or throw error if over-capacity)
    // b) set the first instance of each subsequence to its correct initial value less
    //    the previous subsequences correct initial value
    // c) run an inclusive scan
    // That last step will create an increasing sequence for each subsequence
    // because the inclusive scan is run over an array initialised with 1's.
    // Because the edges emanating from each original vertex are contiguous,
    // the increasing sequence corresponds exactly to the sorted list of all
    // edge id's starting from this seed edge's endpoint. 

    auto const num_edges = edge_endpoint_degrees.size(); // must be nnz, not path_offsets.size()
    auto const num_blocks = dim3( blocks( num_edges ), 1, 1 );

    if( num_edges > path_second_edges.capacity() )
    {
        throw std::runtime_error("Can't fit data to path_second_edges.");
    }

    path_second_edges.resize( thrust::reduce( edge_endpoint_degrees.cbegin()
											, edge_endpoint_degrees.cend()
											, 0 )
							, 1);


    set_negated_first_instances<<<num_blocks, MAX_THRD_BLK>>>( thrust::raw_pointer_cast( edge_endpoint_degrees.data() )
															 , thrust::raw_pointer_cast( path_offsets.data() )
                                                             , thrust::raw_pointer_cast( edge_endpoint_offsets.data() )
                                                             , thrust::raw_pointer_cast( path_second_edges.data() )
                                                             , num_edges );

    thrust::inclusive_scan( path_second_edges.cbegin()
                          , path_second_edges.cend()
                          , path_second_edges.begin()
                          , thrust::plus<MTYPE>() );
}

}

/**
 * Materialises all length-two paths as a CSR graph by indexing edges to lists of paths.
 * The input is two maps from edge->endpoint degree and edge->endpoint offset. This is
 * expanded to an output of edge->paths offset and path->index of second edge.
 * 
 * The operation itself takes an input of counts like [2, 0, 3, 1] and starting indexes
 * like [2, 4, 1, 0] and produces an expanded sequence in which each start index is
 * expanded to an incrementing sequence of length indicated by the corresponding index
 * of the count array. In this example, the result would be: [2, 3, 1, 2, 3, 0]
 * with offsets set to the prefix sum of the counts, i.e., [0, 2, 2, 5, 6] that indicate
 * the index at which paths starting from the i'th edge begin. 
 * 
 * @param edge_endpoint_degrees Degree of vertex v for each edge (u,v). Formerly DE.
 * @param edge_endpoint_offsets Offset to start of neighbours of v for each edge (u,v). Formerly OE.
 * @param path_offsets Offset to start of neighbours w for paths (u,v,w). Formerly N.
 * @param path_second_edges Index second edge in length-two paths. Formerly M.
 */ 
__host__
void cu_multi_arrange( thrust::device_vector<VERTEX_T> const& edge_endpoint_degrees
		             , thrust::device_vector<EDGE_T>   const& edge_endpoint_offsets
                     , thrust::device_vector<EDGE_T>        & path_offsets
		             , thrust::device_vector<MTYPE>         & path_second_edges
		             )
{
    using namespace create_paths;

	// 1. Count non-zero elements directly
	auto nnz = thrust::count_if(edge_endpoint_degrees.cbegin(),
								edge_endpoint_degrees.cend(),
								[] __device__ (VERTEX_T v) { return v != 0; });

	// 2. Copy non-zero elements using edge_endpoint_degrees as stencil
	thrust::device_vector<VERTEX_T> compacted_degrees(nnz);
	thrust::copy_if(edge_endpoint_degrees.cbegin(),
					edge_endpoint_degrees.cend(),
					edge_endpoint_degrees.cbegin(),  // stencil
					compacted_degrees.begin(),
					[] __device__ (VERTEX_T v) { return v != 0; });

	// 3. Copy corresponding offsets using the same stencil
	thrust::device_vector<EDGE_T> compacted_offsets(nnz);
	thrust::copy_if(edge_endpoint_offsets.cbegin(),
					edge_endpoint_offsets.cend(),
					edge_endpoint_degrees.cbegin(),  // stencil
					compacted_offsets.begin(),
					[] __device__ (VERTEX_T v) { return v != 0; });


	path_offsets.resize(edge_endpoint_degrees.size() + 1); //  +1 for first element to be zero
    populate_path_offsets( edge_endpoint_degrees, path_offsets );
	thrust::device_vector<EDGE_T> compacted_path_offsets(nnz); // no need to +1 here as last element will be ignored
    populate_path_offsets( compacted_degrees, compacted_path_offsets );
    extend_edges_to_paths( compacted_degrees, compacted_offsets, compacted_path_offsets, path_second_edges );
}



__global__
void cu_cliques_quds(tri_t* tris, // we use tris struct instead of E vector
					 EDGE_T* O,
					 MTYPE* M0,
					 EDGE_T* N0,
					 MTYPE* M1,
					 EDGE_T* N1,
					 size_t trisSize, // size of E vector
					 size_t OSize, // Size of O vector
					 qud_t* quds, // vector to store four clique results
					 size_t qudsSize // four cliques vector size
) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < OSize && d_global_num_4cliques < qudsSize) {
		EDGE_T d1 = O[i];
		EDGE_T d2 = trisSize;
		if (i < OSize-1) {
			d2 = O[i+1];
		}
		// loop E[d1:d2]
		for(auto d = d1; d < d2; d++) {
			auto i1 = d1;
			auto j1 = N0[d];
			const auto& j2 = N0[d+1];

			// loop EM[j1:j2]
			// intersection(E[d1..d..d2], EM[N[d]..j1..N[d+1])
			// find indices of intersection
			while(i1 < d2 && j1 < j2) {
				const auto& p = tris[i1].b; //E[i1][0];
				const auto& o1 = static_cast<EDGE_T>(M0[j1]);
				const auto& q = tris[o1].b; //E[o1][0];
				if(p < q) {
					i1++;
				} else {
					if(!(q < p)) {
						auto& k1 = N1[i1];
						auto& k2 = N1[i1+1];
						auto k = k1;
						for (; k < k2; k++) {
							const auto& o2 = static_cast<EDGE_T>(M1[k]);
							//if(E[d][1] == E[o2][1]) {
							if(tris[d].c == tris[o2].c) {
									auto id = atomicAdd(&d_global_num_4cliques, 1ul);
									quds[id].a = i1;
									quds[id].b = o1;
									quds[id].c = d;
									quds[id].d = o2;
									i1++;
									break;
							}
						}
					}
					j1++;
				}
			} // end of intersection loop
		} // end of for-loop
	}
}

/**
 * Segmented version of cu_take_quds.
 *
 * Materialises (DE_b, OE_b, DE_c, OE_c) for triangles in the local edge slice
 * [seg_bg, seg_bg + seg_size). Output arrays are sized seg_size and indexed by
 * (global_triangle - seg_bg).
 */
__global__
void cu_take_quds_seg(
		VERTEX_T* trisD,           // triangle-graph degrees (global, size num_triangles)
		EDGE_T* trisO,             // triangle-graph offsets (global, size num_triangles)
		tri_t* tris,               // triangle list (global, size num_triangles)
		std::size_t seg_bg,        // first triangle index in this segment
		std::size_t seg_size,      // number of triangles in this segment
		VERTEX_T* DE_b,
		VERTEX_T* DE_c,
		EDGE_T* OE_b,
		EDGE_T* OE_c) {
	auto const tid = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
	if(tid < seg_size) {
		auto const i = seg_bg + tid;
		DE_b[tid] = trisD[tris[i].b];
		OE_b[tid] = trisO[tris[i].b];
		DE_c[tid] = trisD[tris[i].c];
		OE_c[tid] = trisO[tris[i].c];
	}
}

/**
 * Segmented version of cu_cliques_quds.
 *
 * Processes only triangle-vertices in [vertex_bg, vertex_ed) of the triangle
 * graph. The path arrays (M0/N0, M1/N1) and the per-triangle scratch
 * (OE_b/OE_c implicit through N0/N1) are sized for the local edge slice
 * [dst_bg, trisO[vertex_ed]) and indexed by (global_edge - dst_bg).
 *
 * tris, trisO are read with global indexing because the intersection step
 * dereferences arbitrary triangle ids (o1 = M0[j1], o2 = M1[k]) that can
 * point anywhere in the global triangle list.
 */
__global__
void cu_cliques_quds_seg(tri_t* tris,
                         EDGE_T* trisO,
                         MTYPE* M0,
                         EDGE_T* N0,              // size: (dst_ed - dst_bg + 1)
                         MTYPE* M1,
                         EDGE_T* N1,              // size: (dst_ed - dst_bg + 1)
                         std::size_t num_triangles,   // size of tris[] (== tgraph.size_edges())
                         std::size_t num_tverts,      // size of trisO[] (== tgraph.size_vertices())
                         std::size_t vertex_bg,
                         std::size_t vertex_ed,
                         std::size_t dst_bg,      // = trisO[vertex_bg]
                         qud_t* quds,
                         std::size_t qudsSize) {
	auto const tid = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
	auto const i = vertex_bg + tid;

	if(i < vertex_ed && d_global_num_4cliques < qudsSize) {
		EDGE_T d1 = trisO[i];
		EDGE_T d2 = (i + 1 < num_tverts) ? trisO[i + 1] : static_cast<EDGE_T>(num_triangles);
		// loop tris[d1:d2]
		for(auto d = d1; d < d2; d++) {
			auto i1 = d1;
			auto const local_d = static_cast<std::size_t>(d) - dst_bg;
			auto j1 = N0[local_d];
			const auto& j2 = N0[local_d + 1];

			while(i1 < d2 && j1 < j2) {
				const auto& p = tris[i1].b;
				const auto& o1 = static_cast<EDGE_T>(M0[j1]);
				const auto& q = tris[o1].b;
				if(p < q) {
					i1++;
				} else {
					if(!(q < p)) {
						auto const local_i1 = static_cast<std::size_t>(i1) - dst_bg;
						auto& k1 = N1[local_i1];
						auto& k2 = N1[local_i1 + 1];
						auto k = k1;
						for (; k < k2; k++) {
							const auto& o2 = static_cast<EDGE_T>(M1[k]);
							if(tris[d].c == tris[o2].c) {
									auto id = atomicAdd(&d_global_num_4cliques, 1ul);
									quds[id].a = i1;
									quds[id].b = o1;
									quds[id].c = d;
									quds[id].d = o2;
									i1++;
									break;
							}
						}
					}
					j1++;
				}
			} // end of intersection loop
		} // end of for-loop
	}
}


/**
 * Make quds undirected by adding quds three 
 * other permutation to the quds vector then
 * we do the sorting
*/
__global__
void cu_quds_to_ugraph(qud_t* quds,
					   size_t qudsSize) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < qudsSize) {
		const auto& qud = quds[i];

		quds[i+qudsSize].a = qud.b;
		quds[i+qudsSize].b = qud.c;
		quds[i+qudsSize].c = qud.d;
		quds[i+qudsSize].d = qud.a;

		quds[i+(2*qudsSize)].a = qud.c;
		quds[i+(2*qudsSize)].b = qud.d;
		quds[i+(2*qudsSize)].c = qud.a;
		quds[i+(2*qudsSize)].d = qud.b;
	
		quds[i+(3*qudsSize)].a = qud.d;
		quds[i+(3*qudsSize)].b = qud.a;
		quds[i+(3*qudsSize)].c = qud.b;
		quds[i+(3*qudsSize)].d = qud.c;
	}
}

/**
 * Make quds undirected by adding quds three 
 * other permutation to the quds vector then
 * we do the sorting
*/
__global__
void cu_quds_to_ugraph2(qud_t* quds,
						dbl_t* dbls,
					   size_t qudsSize) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < qudsSize) {
		const auto& qud = quds[i];

		//a, 
		dbls[i].a = qud.a;
		dbls[i].b = qud.b;

		dbls[i+qudsSize].a = qud.a;
		dbls[i+qudsSize].b = qud.c;

		dbls[i+(2*qudsSize)].a = qud.a;
		dbls[i+(2*qudsSize)].b = qud.d;

		//b, 
		dbls[i+(3*qudsSize)].a = qud.b;
		dbls[i+(3*qudsSize)].b = qud.a;
		
		dbls[i+(4*qudsSize)].a = qud.b;
		dbls[i+(4*qudsSize)].b = qud.c;
		
		dbls[i+(5*qudsSize)].a = qud.b;
		dbls[i+(5*qudsSize)].b = qud.d;

		//c,
		dbls[i+(6*qudsSize)].a = qud.c;
		dbls[i+(6*qudsSize)].b = qud.a;
		
		dbls[i+(7*qudsSize)].a = qud.c;
		dbls[i+(7*qudsSize)].b = qud.b;
		
		dbls[i+(8*qudsSize)].a = qud.c;
		dbls[i+(8*qudsSize)].b = qud.d;

		//d,
		dbls[i+(9*qudsSize)].a = qud.d;
		dbls[i+(9*qudsSize)].b = qud.a;
		
		dbls[i+(10*qudsSize)].a = qud.d;
		dbls[i+(10*qudsSize)].b = qud.b;
		
		dbls[i+(11*qudsSize)].a = qud.d;
		dbls[i+(11*qudsSize)].b = qud.c;
	}
}

namespace wipeout {
	
	__global__
	void cu_initialize() {
		d_2peel_current = 1U;
		d_2peel_next = -1U;
	}

	__global__
	void cu_next_value_to_peel(
				const MTYPE* V,
				const VERTEX_T* D,
				size_t sz) {
		
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if(i < sz && V[i] == -1) {
			atomicMin(&d_2peel_next, D[i]);
		}
	}

	__global__
	void cu_update_value_to_peel() {
		atomicMax(&d_2peel_current, d_2peel_next);
		d_2peel_next = -1U;
	}

	__global__
	void cu_peeling_kcore(const VERTEX_T* E,
					MTYPE* V,
					const VERTEX_T* D,
					VERTEX_T* DD,
					const VERTEX_T* O,
					size_t sz) {

		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if(i < sz && V[i]==-1 && DD[i]<=d_2peel_current) {
				atomicExch_system(&V[i], d_2peel_current);
				const auto& deg = D[i];
				const auto& off = O[i];

				for(size_t d = 0; d < deg; d++) {
					auto& j = E[off+d];
					atomicSub_system(&DD[j], 1U);
				} // neighbors loop
		}
	}


	__host__
	unsigned int run(const std::vector<VERTEX_T>& D,
		const std::vector<EDGE_T>& O,
		const std::vector<VERTEX_T>& E,
		std::vector<MTYPE>& kv,
		std::map<std::string, milliseconds>& tms) {
		
		std::cout << "=========================================" << std::endl;
		std::cout << "CUDA allocating memory ... " << std::endl;

		auto nV = D.size();
		thrust::device_vector<VERTEX_T> d_D(D);
		auto d_DD = d_D; // D vector to decrease
		thrust::device_vector<EDGE_T> d_O(O);
		thrust::device_vector<VERTEX_T> d_E(E);
		thrust::device_vector<MTYPE> d_kv(nV, -1);
		dim3 nBlocks (blocks(nV), 1, 1);


		std::cout << "=========================================" << std::endl;
		std::cout << "CUDA constructing Bucket ..." << std::endl;

		auto t = hrc::now();

		cu_initialize<<<1, 1>>>();

		// only copies zeros the rest will remain -1
		thrust::replace_copy_if(
			d_D.begin(),
			d_D.end(),
			d_kv.begin(), 
			[] __device__(const auto& t){return t!=0;},
			-1);


		CU_ERR( cudaPeekAtLastError() );
		CU_ERR( cudaDeviceSynchronize() );
		tms["constructing"] = hrc::now() - t;
		std::cout << "CUDA constructing Bucket finished: "
			<< tms["constructing"].count() << " ms" << std::endl;
		
		std::cout << "=========================================" << std::endl;
		std::cout << "CUDA peeling Bucket ..." << std::endl;
		auto nLoopCounter = 0U;
		unsigned next_peel = 0U;
		t = hrc::now();
		while(true) {
			cu_next_value_to_peel<<<nBlocks, MAX_THRD_BLK>>>(
				thrust::raw_pointer_cast(d_kv.data()),
				thrust::raw_pointer_cast(d_DD.data()),
				d_kv.size());
			cudaMemcpyFromSymbol(&next_peel,
								d_2peel_next,
								sizeof(d_2peel_next),
								0,
								cudaMemcpyDeviceToHost);
			if(next_peel==-1U){
				break;
			}

			cu_update_value_to_peel<<<1, 1>>>();
				
			cu_peeling_kcore<<<nBlocks, MAX_THRD_BLK>>>(
					thrust::raw_pointer_cast(d_E.data()),
					thrust::raw_pointer_cast(d_kv.data()),
					thrust::raw_pointer_cast(d_D.data()),
					thrust::raw_pointer_cast(d_DD.data()),
					thrust::raw_pointer_cast(d_O.data()),
					d_D.size());
			
			nLoopCounter++;

		} // End of while loop
		CU_ERR( cudaPeekAtLastError() );
		CU_ERR( cudaDeviceSynchronize() ); // This is required
		tms["peeling"] = hrc::now() - t;
		std::cout << "CUDA peeling Bucket finished: "
			<< tms["peeling"].count() << " ms" << std::endl;
		std::cout << "Total Kernel launches: "
			<< nLoopCounter	<< std::endl;
		std::cout << "=========================================" << std::endl;
		std::cout << "Transferring data back to CPU ..." << std::endl;

		unsigned k_max;
		cudaMemcpyFromSymbol(&k_max,
							d_2peel_current,
							sizeof(d_2peel_current),
							0,
							cudaMemcpyDeviceToHost);

		kv.resize(d_kv.size());
		thrust::copy(d_kv.cbegin(), d_kv.cend(), kv.begin());
		tms["all"] = milliseconds::zero();
		for(const auto& tm: tms) {
			if(tm.first!="all") {
				tms["all"] += tm.second;
			}
		}
		return k_max;
	} // End of run()
} // End of wipeout approach



namespace cu {
	__global__
	void initialize(unsigned int startPeel=1U) {
		d_2peel_current = 0U;
		d_2peel_next = startPeel;
		d_peeled = 0U;
	}

	__global__
	void peel_zeros(
				MTYPE* V,
				const VERTEX_T* D,
				size_t sz) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if(i < sz && D[i] == 0) {
			V[i] = 0;
			atomicAdd(&d_peeled, 1);
		}
	}

	__global__
	void update_value_to_peel() {
		d_2peel_current = d_2peel_next;
		//printf("cu peel %d\n", d_2peel_current);
		d_2peel_next++;
	}
	
	namespace kcore {
		__global__
		void peeling(const VERTEX_T* E,
						MTYPE* V,
						const VERTEX_T* D,
						VERTEX_T* DD,
						const VERTEX_T* O,
						size_t sz) {

			int i = blockIdx.x * blockDim.x + threadIdx.x;
			if(i < sz && V[i]==-1 && DD[i] <= d_2peel_current) {
				atomicExch_system(&V[i], d_2peel_current);
				atomicAdd(&d_peeled, 1);

				const auto& deg = D[i];
				const auto& off = O[i];

				for(size_t d = 0; d < deg; d++) {
					auto& j = E[off+d];
					auto old = atomicSub_system(&DD[j], 1U);
					if(old == d_2peel_current+1) {
						d_2peel_next = d_2peel_current;
					}
				} // neighbors loop
			} // kernel block
		} // cu_peeling_kcore(...)

		__host__
		unsigned int run(const std::vector<VERTEX_T>& D,
			const std::vector<EDGE_T>& O,
			const std::vector<VERTEX_T>& E,
			std::vector<MTYPE>& kv,
			std::map<std::string, milliseconds>& tms) {
			
			std::cout << "=========================================" << std::endl;
			std::cout << "CUDA allocating memory ... " << std::endl;

			auto nV = D.size();
			thrust::device_vector<VERTEX_T> d_D(D);
			auto d_DD = d_D; // D vector to decrease
			thrust::device_vector<EDGE_T> d_O(O);
			thrust::device_vector<VERTEX_T> d_E(E);
			thrust::device_vector<MTYPE> d_kv(nV, -1);
			dim3 nBlocks (blocks(nV), 1, 1);


			std::cout << "=========================================" << std::endl;
			std::cout << "CUDA constructing Bucket ..." << std::endl;

			auto t = hrc::now();

			cu::initialize<<<1, 1>>>(4U);

			cu::peel_zeros<<<nBlocks, MAX_THRD_BLK>>>(
				thrust::raw_pointer_cast(d_kv.data()),
				thrust::raw_pointer_cast(d_D.data()),
				d_D.size());


			CU_ERR( cudaPeekAtLastError() );
			CU_ERR( cudaDeviceSynchronize() );
			tms["constructing"] = hrc::now() - t;
			std::cout << "CUDA constructing Bucket finished: "
				<< tms["constructing"].count() << " ms" << std::endl;
			
			std::cout << "=========================================" << std::endl;
			std::cout << "CUDA peeling Bucket ..." << std::endl;
			auto nLoopCounter = 0U;
			unsigned peeled = 0U;
			t = hrc::now();
			while(true) {
				nLoopCounter++;

				cu::update_value_to_peel<<<1, 1>>>();
					
				peeling<<<nBlocks, MAX_THRD_BLK>>>(
						thrust::raw_pointer_cast(d_E.data()),
						thrust::raw_pointer_cast(d_kv.data()),
						thrust::raw_pointer_cast(d_D.data()),
						thrust::raw_pointer_cast(d_DD.data()),
						thrust::raw_pointer_cast(d_O.data()),
						d_D.size());

				cudaMemcpyFromSymbol(&peeled,
									d_peeled,
									sizeof(d_peeled),
									0,
									cudaMemcpyDeviceToHost);
				if(peeled>=nV){
					break;
				}
			} // End of while loop
			
			CU_ERR( cudaPeekAtLastError() );
			CU_ERR( cudaDeviceSynchronize() ); // This is required
		
			tms["peeling"] = hrc::now() - t;
			std::cout << "CUDA peeling Bucket finished: "
				<< tms["peeling"].count() << " ms" << std::endl;
			std::cout << "Total Kernel launches: "
				<< nLoopCounter	<< std::endl;
			std::cout << "=========================================" << std::endl;
			std::cout << "Transferring data back to CPU ..." << std::endl;

			unsigned k_max;
			cudaMemcpyFromSymbol(&k_max,
								d_2peel_current,
								sizeof(d_2peel_current),
								0,
								cudaMemcpyDeviceToHost);

			kv.resize(d_kv.size());
			thrust::copy(d_kv.cbegin(), d_kv.cend(), kv.begin());
			tms["all"] = milliseconds::zero();
			for(const auto& tm: tms) {
				if(tm.first!="all") {
					tms["all"] += tm.second;
				}
			}
			return k_max;
		} // End of run()
	} // End of namespace kcore

	namespace ktruss {
		/**
		* @brief Peeling for ktruss
		* @param tris is the Triangles vector
		* @param V is the values vector
		* @param D is the degree vector
		* @param DD is the mutable degree vector
		* @param O is the offset vector
		* @param sz Size of vector V, O, D and DD
		*/
		__global__
		void peeling(const tri_t* tris,
						MTYPE* V,
						const VERTEX_T* D,
						MTYPE* DD,
						const VERTEX_T* O,
						size_t sz) {

			int i = blockIdx.x * blockDim.x + threadIdx.x;

			if(i < sz && V[i] == -1 && DD[i] <= static_cast<MTYPE>(d_2peel_current*2)) {
				atomicExch_system(&V[i], d_2peel_current);
				atomicAdd(&d_peeled, 1);
				
				const auto& deg = D[i];
				const auto& off = O[i];

				for(size_t g = 0; g < deg; g++) {
					const auto& t = tris[off+g];
					auto old_b = atomicSub_system(&DD[t.b], 1U);
					auto old_c = atomicSub_system(&DD[t.c], 1U);
					if(old_b == d_2peel_current+1
						|| old_c == d_2peel_current+1
					) {
						d_2peel_next = d_2peel_current;
					}
				} // neighbors loop
			} // kernel block
		} // peeling(...)
	} // End of namespace ktruss

	namespace nucleus34 {
		/**
		* @brief Peeling for nucleus34
		* @param quds is the FC vector
		* @param V is the values vector
		* @param D is the degree vector
		* @param DD is the mutable degree vector
		* @param O is the offset vector
		* @param sz Size of vector V, O, D and DD
		*/
		__global__
		void peeling(const qud_t* quds,
						MTYPE* V,
						const VERTEX_T* D,
						MTYPE* DD,
						const VERTEX_T* O,
						size_t sz) {

			int i = blockIdx.x * blockDim.x + threadIdx.x;
			const auto& actual_min = static_cast<MTYPE>(d_2peel_current * NUCLEUS34_FACTOR );
			//const int v_debug = 3215; 
			const int v_debug = 1669; 

			if(i < sz && V[i] == -1 && DD[i] <= actual_min) {
				if(i==v_debug) {
					MSG__("Peeling %d when min: %d, D: %u and DD: %d\n",
						i, d_2peel_current, D[i], DD[i]);
				}
				atomicExch_system(&V[i], d_2peel_current);
				atomicAdd(&d_peeled, 1);
				
				const auto& deg = D[i];
				const auto& off = O[i];

				for(size_t g = 0; g < deg; g++) {
					const auto& q = quds[off+g];
					auto old_b = atomicSub_system(&DD[q.b], 1U);
					auto old_c = atomicSub_system(&DD[q.c], 1U);
					auto old_d = atomicSub_system(&DD[q.d], 1U);
					if(q.b==v_debug) {
						MSG__("Reducing %i when min is %u, by i: %d, D[i]: %u, DD[i]: %d, when DD is %d\n",
							v_debug, d_2peel_current, i, D[i], DD[i], DD[q.b]);
					}
					if(q.c==v_debug) {
						MSG__("Reducing %i when min is %u, by i: %d, D[i]: %u, DD[i]: %d, when DD is %d\n",
							v_debug, d_2peel_current, i, D[i], DD[i], DD[q.c]);
					}
					if(q.d==v_debug) {
						MSG__("Reducing %i when min is %u, by i: %d, D[i]: %u, DD[i]: %d, when DD is %d\n",
							v_debug, d_2peel_current, i, D[i], DD[i], DD[q.d]);
					}
					//const auto& nextMin = static_cast<MTYPE>(d_2peel_current+1);
					//const auto& nextMin = static_cast<MTYPE>((d_2peel_current+1)*NUCLEUS34_FACTOR);
					const auto& nextMin = static_cast<MTYPE>(actual_min+1);

					if(old_b <= nextMin || old_c <= nextMin || old_d <= nextMin) {
						d_2peel_next = d_2peel_current;
					}
				} // neighbors loop
			} // kernel block
		} // peeling(...)
		
		__global__
		void peeling2(const dbl_t* dbls,
						MTYPE* V,
						const VERTEX_T* D,
						MTYPE* DD,
						const VERTEX_T* O,
						size_t sz) {

			int i = blockIdx.x * blockDim.x + threadIdx.x;
			const auto& actual_min = static_cast<MTYPE>(d_2peel_current /*NUCLEUS34_FACTOR*/ );

			if(i < sz && V[i] == -1 && DD[i] <= actual_min) {
				atomicExch(&V[i], d_2peel_current);
				atomicAdd(&d_peeled, 1);
				
				const auto& deg = D[i];
				const auto& off = O[i];

				for(size_t g = 0; g < deg; g++) {
					const auto& q = dbls[off+g];
					auto old_b = atomicSub(&DD[q.b], 1);
					//const auto& nextMin = static_cast<MTYPE>(d_2peel_current+1);
					//const auto& nextMin = static_cast<MTYPE>((d_2peel_current+1)*NUCLEUS34_FACTOR);
					const auto& nextMin = static_cast<MTYPE>(actual_min+1);

					if(old_b == nextMin) {
						d_2peel_next = d_2peel_current;
					}
				} // neighbors loop
			} // kernel block
		} // peeling(...)
	} // End of namespace nucleus34

	/**
	 * Materialises a list of all triangles in a graph.
	 * 
	 * The input is a graph represented in CSR format by one vector of length |E| that gives
	 * the destination vertex of each edge sorted by the source vertex and
	 * two vectors of length |V|, one that gives the degree of each i'th vertex and one
	 * that gives the index in the length-|E| first vector at which the neighbours of the i'th
	 * vertex begin.
	 * 
	 * The output is the population of one vector of triangles. Three other pre-allocated
	 * vectors of length |E| and one unallocated vector that will be of length |P2| (i.e., the
	 * number of unique length-two paths in the graph) are also passed for storing itermediate
	 * data structures.
	 */
	void compute_triangles( thrust::device_vector<VERTEX_T>		& d_vertex_degrees // const
						, thrust::device_vector<EDGE_T>		& d_edge_offsets // const 
						, thrust::device_vector<VERTEX_T>		& d_edge_destinations // const
						, thrust::device_vector<VERTEX_T>      & d_edge_destination_degrees
						, thrust::device_vector<EDGE_T>        & d_edge_destination_offsets
						, thrust::device_vector<EDGE_T>        & d_path_offsets
						, thrust::device_vector<INDEX_T>       & d_path_second_edge_indexes
						, thrust::device_vector<tri_t>         & d_output_triangles )
	{
		auto const num_vertices = d_edge_offsets.size();
		auto const num_edges = d_edge_destinations.size();
		auto const max_capacity_output_triangles = d_output_triangles.size();
		const dim3 nBlocks (blocks(num_edges), 1, 1);
		
		// Broadly, the algorithm to materialise all triangles first materialises all
		// length-two paths (u,v,w) and then finds triangles by intersecting that sorted
		// vector with the sorted vector of edges to find matches (u,w). Then {u,v,w} is a triangle.
		// Because edges are oriented from low- to high-degree and u<v<w, this guarantees
		// that all triangles that are discovered are unique and that the number of
		// neighbours of each vertex is upper-bounded by the degeneracy of the graph.
		//
		// The algorithm to materialise all triangles consists of three steps:
		//   a) Materialise a map from each sorted edge (u,v) onto degree(v) and offset(v) for locality
		//   b) Build an expanded sequence by replacing each edge (u,v) with all i where edge[i] = (v, ?)
		//        Note that these subsequences are sorted. Because they are indexes into another sorted list,
		//        they are still sorted after dereferencing the destination of the edge to determine '?'
		//   c) Intersect the expanded sequence with the original edge list, outputing a triangle for each match.
		//        The expanded sequence are indexes, not vertices, so to determine if there is a match, the index
		//        is looked up in the original edge list, too.
		//
		// In terms of parallelism, the lightweight first step exposes very fine-grained tasks per edge.
		// The main work (the second step and the third step) exposes coarse-grained tasks per vertex.

		// Populate temp structures d_edge_destination_degrees and d_edge_destination_offsets.
		// Legacy notation: OE = O[E], DE = D[E]
		cuTake<<<nBlocks, MAX_THRD_BLK>>>( thrust::raw_pointer_cast( d_vertex_degrees.data() )
										, thrust::raw_pointer_cast( d_edge_offsets.data() )
										, thrust::raw_pointer_cast( d_edge_destinations.data() )
										, thrust::raw_pointer_cast( d_edge_destination_degrees.data() )
										, thrust::raw_pointer_cast( d_edge_destination_offsets.data() )
										, num_edges );

		// Materialise length-two paths in d_path_offsets and d_path_second_edge_indexes
		// Legacy notaion: (OE, DE) =>  M, N
		cu_multi_arrange( d_edge_destination_degrees
						, d_edge_destination_offsets
						, d_path_offsets
						, d_path_second_edge_indexes );

		// Do intersections to produce triangles
		cu_cliques_tris<<<nBlocks, MAX_THRD_BLK>>>( thrust::raw_pointer_cast( d_edge_destinations.data() )
												, thrust::raw_pointer_cast( d_edge_offsets.data() )
												, thrust::raw_pointer_cast( d_path_second_edge_indexes.data() )
												, thrust::raw_pointer_cast( d_path_offsets.data() )
												, num_edges
												, num_vertices
												, thrust::raw_pointer_cast( d_output_triangles.data() )
												, max_capacity_output_triangles );

		

		CU_ERR( cudaPeekAtLastError() );
		CU_ERR( cudaDeviceSynchronize() );
	}

	/**
	 * Multi-GPU version of compute_triangles. GPU analogue of cliques_tri_par:
	 * graph.split_parts defines vertex ranges that are processed independently
	 * on different GPUs (round-robin across visible CUDA devices), then the
	 * per-segment triangle lists are concatenated into a single host vector.
	 *
	 * The graph (D, O, E) is replicated on each device; each device runs
	 * cuTake / cu_multi_arrange / cu_cliques_tris_seg on its segment only.
	 */
	template <typename U, typename V, size_t DIM>
	std::vector<tri_t> compute_triangles_multi_gpu( const graph_t<U, V, DIM>& graph )
	{
		if( graph.split_parts.empty() ) {
			throw std::runtime_error( "split_parts vector not initialized for multi-GPU call." );
		}

		int num_devices = 0;
		CU_ERR( cudaGetDeviceCount( &num_devices ) );
		if( num_devices < 1 ) {
			throw std::runtime_error( "No CUDA devices available." );
		}

		const size_t num_segments = graph.split_parts.size();
		const size_t num_vertices = graph.O.size();
		const size_t num_edges    = graph.size_edges();

		std::vector<std::vector<tri_t>> per_seg( num_segments );

		int prev_device = -1;
		cudaGetDevice( &prev_device );

		for( size_t s = 0; s < num_segments; ++s )
		{
			const int dev = static_cast<int>( s % num_devices );
			CU_ERR( cudaSetDevice( dev ) );

			const size_t vb = graph.split_parts[s];
			const size_t ve = ( s + 1 < num_segments ) ? graph.split_parts[s+1] : num_vertices;
			if( ve <= vb ) continue;

			const size_t db = static_cast<size_t>( graph.O[vb] );
			const size_t de = ( ve < num_vertices ) ? static_cast<size_t>( graph.O[ve] ) : num_edges;
			const size_t seg_edges = de - db;
			const size_t seg_verts = ve - vb;
			if( seg_edges == 0 ) continue;

			thrust::device_vector<VERTEX_T> d_D( graph.D );
			thrust::device_vector<EDGE_T>   d_O( graph.O );
			auto e_ptr = reinterpret_cast<const VERTEX_T*>( graph.E.data() );
			thrust::device_vector<VERTEX_T> d_E( e_ptr, e_ptr + num_edges );

			// path_count = sum of D[E[i]] for i in [db, de): exact size of local M
			const size_t path_count = static_cast<size_t>( thrust::reduce(
				thrust::make_permutation_iterator( d_D.begin(), d_E.begin() + db ),
				thrust::make_permutation_iterator( d_D.begin(), d_E.begin() + de ) ) );

			thrust::device_vector<VERTEX_T> d_DE( seg_edges, 0 );
			thrust::device_vector<EDGE_T>   d_OE( seg_edges, 0 );
			thrust::device_vector<EDGE_T>   d_N ( seg_edges + 1, 0 );
			thrust::device_vector<INDEX_T>  d_M ( path_count, 1 );
			thrust::device_vector<tri_t>    d_tris( path_count );

			// Reset per-device global triangle counter
			initialize_kernel<<<1, 1>>>();
			CU_ERR( cudaDeviceSynchronize() );

			// cuTake on the local edge slice [db, de)
			const dim3 nB_e( blocks( seg_edges ), 1, 1 );
			cuTake<<<nB_e, MAX_THRD_BLK>>>( thrust::raw_pointer_cast( d_D.data() )
			                              , thrust::raw_pointer_cast( d_O.data() )
			                              , thrust::raw_pointer_cast( d_E.data() ) + db
			                              , thrust::raw_pointer_cast( d_DE.data() )
			                              , thrust::raw_pointer_cast( d_OE.data() )
			                              , seg_edges );

			cu_multi_arrange( d_DE, d_OE, d_N, d_M );

			const dim3 nB_v( blocks( seg_verts ), 1, 1 );
			cu_cliques_tris_seg<<<nB_v, MAX_THRD_BLK>>>( thrust::raw_pointer_cast( d_E.data() )
			                                           , thrust::raw_pointer_cast( d_O.data() )
			                                           , thrust::raw_pointer_cast( d_M.data() )
			                                           , thrust::raw_pointer_cast( d_N.data() )
			                                           , num_edges
			                                           , num_vertices
			                                           , vb
			                                           , ve
			                                           , db
			                                           , thrust::raw_pointer_cast( d_tris.data() )
			                                           , d_tris.size() );

			CU_ERR( cudaPeekAtLastError() );
			CU_ERR( cudaDeviceSynchronize() );

			size_t seg_count = 0;
			CU_ERR( cudaMemcpyFromSymbol( &seg_count, d_global_num_triangles
			                            , sizeof( d_global_num_triangles ), 0
			                            , cudaMemcpyDeviceToHost ) );
			if( seg_count > d_tris.size() ) {
				throw std::runtime_error( "compute_triangles_multi_gpu: per-segment output capacity exceeded." );
			}

			per_seg[s].resize( seg_count );
			thrust::copy( d_tris.cbegin(), d_tris.cbegin() + seg_count, per_seg[s].begin() );
		}

		if( prev_device >= 0 ) cudaSetDevice( prev_device );

		size_t total = 0;
		for( const auto& v : per_seg ) total += v.size();
		std::vector<tri_t> all;
		all.reserve( total );
		for( auto& v : per_seg ) {
			all.insert( all.end(), v.begin(), v.end() );
		}
		return all;
	}

	/**
	 * Parted (pipelined / multi-GPU) version of compute_triangles. GPU analogue
	 * of cliques_tri_par: graph.split_parts defines vertex ranges that are
	 * processed independently. Segments are distributed round-robin across the
	 * visible CUDA devices and each device is driven by its own host thread, so
	 * with N GPUs up to N segments run concurrently. On each device the graph
	 * (D, O, E) is uploaded once and reused by all segments assigned to that
	 * device, then per-segment scratch (DE, OE, N, M, tris) is allocated, the
	 * per-device triangle counter (d_global_num_triangles) is reset, the same
	 * cuTake / cu_multi_arrange / cu_cliques_tris_seg pipeline used by
	 * compute_triangles_multi_gpu is run, and the resulting tris slice is
	 * copied back to host. After all threads join the per-segment host buffers
	 * are concatenated into a single triangle list.
	 *
	 * Falls back to single-GPU execution when only one device is visible.
	 */
	template <typename U, typename V, size_t DIM>
	std::vector<tri_t> compute_triangles_parted( const graph_t<U, V, DIM>& graph )
	{
		if( graph.split_parts.empty() ) {
			throw std::runtime_error( "split_parts vector not initialized for parted call." );
		}

		int num_devices = 0;
		CU_ERR( cudaGetDeviceCount( &num_devices ) );
		if( num_devices < 1 ) {
			throw std::runtime_error( "No CUDA devices available." );
		}

		const size_t num_segments = graph.split_parts.size();
		const size_t num_vertices = graph.O.size();
		const size_t num_edges    = graph.size_edges();

		// Cap worker threads to min(num_devices, num_segments): launching more
		// threads than segments would leave threads idle and risks setting a
		// device that has no work assigned.
		const size_t num_workers = std::min<size_t>( num_segments,
		                                             static_cast<size_t>( num_devices ) );

		std::vector<std::vector<tri_t>> per_seg( num_segments );
		std::mutex err_mtx;
		std::exception_ptr first_err = nullptr;

		int prev_device = -1;
		cudaGetDevice( &prev_device );

		auto worker = [&]( int dev )
		{
			try {
				CU_ERR( cudaSetDevice( dev ) );

				// Upload the full graph once per device; reused by every
				// segment assigned to this device.
				thrust::device_vector<VERTEX_T> d_D( graph.D );
				thrust::device_vector<EDGE_T>   d_O( graph.O );
				auto e_ptr = reinterpret_cast<const VERTEX_T*>( graph.E.data() );
				thrust::device_vector<VERTEX_T> d_E( e_ptr, e_ptr + num_edges );

				for( size_t s = static_cast<size_t>( dev ); s < num_segments; s += num_workers )
				{
					const size_t vb = graph.split_parts[s];
					const size_t ve = ( s + 1 < num_segments ) ? graph.split_parts[s+1] : num_vertices;
					if( ve <= vb ) continue;

					const size_t db = static_cast<size_t>( graph.O[vb] );
					const size_t de = ( ve < num_vertices ) ? static_cast<size_t>( graph.O[ve] ) : num_edges;
					const size_t seg_edges = de - db;
					const size_t seg_verts = ve - vb;
					if( seg_edges == 0 ) continue;

					// path_count = sum of D[E[i]] for i in [db, de): exact size of local M
					const size_t path_count = static_cast<size_t>( thrust::reduce(
						thrust::make_permutation_iterator( d_D.begin(), d_E.begin() + db ),
						thrust::make_permutation_iterator( d_D.begin(), d_E.begin() + de ) ) );

					thrust::device_vector<VERTEX_T> d_DE( seg_edges, 0 );
					thrust::device_vector<EDGE_T>   d_OE( seg_edges, 0 );
					thrust::device_vector<EDGE_T>   d_N ( seg_edges + 1, 0 );
					thrust::device_vector<INDEX_T>  d_M ( path_count, 1 );
					thrust::device_vector<tri_t>    d_tris( path_count );

					// Reset this device's global triangle counter. Segments
					// scheduled to the same device run serially within this
					// thread, so the shared counter is safe.
					initialize_kernel<<<1, 1>>>();
					CU_ERR( cudaDeviceSynchronize() );

					const dim3 nB_e( blocks( seg_edges ), 1, 1 );
					cuTake<<<nB_e, MAX_THRD_BLK>>>( thrust::raw_pointer_cast( d_D.data() )
					                              , thrust::raw_pointer_cast( d_O.data() )
					                              , thrust::raw_pointer_cast( d_E.data() ) + db
					                              , thrust::raw_pointer_cast( d_DE.data() )
					                              , thrust::raw_pointer_cast( d_OE.data() )
					                              , seg_edges );

					cu_multi_arrange( d_DE, d_OE, d_N, d_M );

					const dim3 nB_v( blocks( seg_verts ), 1, 1 );
					cu_cliques_tris_seg<<<nB_v, MAX_THRD_BLK>>>( thrust::raw_pointer_cast( d_E.data() )
					                                           , thrust::raw_pointer_cast( d_O.data() )
					                                           , thrust::raw_pointer_cast( d_M.data() )
					                                           , thrust::raw_pointer_cast( d_N.data() )
					                                           , num_edges
					                                           , num_vertices
					                                           , vb
					                                           , ve
					                                           , db
					                                           , thrust::raw_pointer_cast( d_tris.data() )
					                                           , d_tris.size() );

					CU_ERR( cudaPeekAtLastError() );
					CU_ERR( cudaDeviceSynchronize() );

					size_t seg_count = 0;
					CU_ERR( cudaMemcpyFromSymbol( &seg_count, d_global_num_triangles
					                            , sizeof( d_global_num_triangles ), 0
					                            , cudaMemcpyDeviceToHost ) );
					if( seg_count > d_tris.size() ) {
						throw std::runtime_error( "compute_triangles_parted: per-segment output capacity exceeded." );
					}

					per_seg[s].resize( seg_count );
					thrust::copy( d_tris.cbegin(), d_tris.cbegin() + seg_count, per_seg[s].begin() );
				}
			} catch (...) {
				std::lock_guard<std::mutex> lk( err_mtx );
				if( !first_err ) first_err = std::current_exception();
			}
		};

		std::vector<std::thread> threads;
		threads.reserve( num_workers );
		for( size_t i = 0; i < num_workers; ++i ) {
			threads.emplace_back( worker, static_cast<int>( i ) );
		}
		for( auto& t : threads ) t.join();

		if( prev_device >= 0 ) cudaSetDevice( prev_device );

		if( first_err ) std::rethrow_exception( first_err );

		size_t total = 0;
		for( const auto& v : per_seg ) total += v.size();
		std::vector<tri_t> all;
		all.reserve( total );
		for( auto& v : per_seg ) {
			all.insert( all.end(), v.begin(), v.end() );
		}
		return all;
	}

	void build_triangles_graph( thrust::device_vector<tri_t>    & d_triangles
							, thrust::device_vector<VERTEX_T> & d_triangles_degrees
							, thrust::device_vector<VERTEX_T> & d_triangles_offsets
							, std::size_t                       num_triangles )
	{
		if(num_triangles)
		{
			thrust::sort( d_triangles.begin(), d_triangles.begin() + num_triangles );

			dim3 nBlocks (blocks(num_triangles), 1, 1);
			cu_build_trisD<<<nBlocks, MAX_THRD_BLK>>>( thrust::raw_pointer_cast( d_triangles.data() )
													, num_triangles
													, thrust::raw_pointer_cast( d_triangles_degrees.data() ) );

			thrust::exclusive_scan( d_triangles_degrees.begin()
								, d_triangles_degrees.end()
								, d_triangles_offsets.begin()
								, 0 );
			
			CU_ERR( cudaPeekAtLastError() );
			CU_ERR( cudaDeviceSynchronize() );
		}
		// else there are no triangles, so don't do anything
	}

	/**
	 * Materialises a list of all four-cliques in the triangle graph.
	 *
	 * Given the triangle-graph CSR (d_triangles_degrees, d_triangles_offsets) and
	 * the sorted triangle list d_triangles, this expands each triangle's b- and
	 * c-component into length-two paths and intersects them to discover
	 * four-cliques. The kernel writes the number of discovered four-cliques into
	 * the device symbol d_global_num_4cliques.
	 *
	 * d_DE_b, d_DE_c, d_OE_b, d_OE_c, d_N0 and d_N1 are scratch vectors that the
	 * caller must pre-allocate (sized as in the original code). d_M0, d_M1 and
	 * d_quds are sized internally based on the per-component path counts, which
	 * are not known until cu_take_quds has run.
	 */
	void compute_fourcliques( thrust::device_vector<VERTEX_T>      & d_triangles_degrees
							, thrust::device_vector<EDGE_T>        & d_triangles_offsets
							, thrust::device_vector<tri_t>         & d_triangles
							, std::size_t                            num_triangles
							, thrust::device_vector<VERTEX_T>      & d_DE_b
							, thrust::device_vector<VERTEX_T>      & d_DE_c
							, thrust::device_vector<EDGE_T>        & d_OE_b
							, thrust::device_vector<EDGE_T>        & d_OE_c
							, thrust::device_vector<EDGE_T>        & d_N0
							, thrust::device_vector<EDGE_T>        & d_N1
							, thrust::device_vector<MTYPE>         & d_M0
							, thrust::device_vector<MTYPE>         & d_M1
							, thrust::device_vector<qud_t>         & d_quds )
	{
		const dim3 nBlocks (blocks(num_triangles * 4), 1, 1);

		// Materialise triangle-graph endpoint degrees/offsets along the b- and
		// c-components for each triangle. Legacy notation: DE = D[E], OE = O[E].
		cu_take_quds<<<nBlocks, MAX_THRD_BLK>>>(
				thrust::raw_pointer_cast(d_triangles_degrees.data()),
				thrust::raw_pointer_cast(d_triangles_offsets.data()),
				thrust::raw_pointer_cast(d_triangles.data()),
				num_triangles,
				thrust::raw_pointer_cast(d_DE_b.data()),
				thrust::raw_pointer_cast(d_DE_c.data()),
				thrust::raw_pointer_cast(d_OE_b.data()),
				thrust::raw_pointer_cast(d_OE_c.data()));

		CU_ERR( cudaPeekAtLastError() );
		CU_ERR( cudaDeviceSynchronize() );

		// path_count_m0/m1 = sum of triangle-graph endpoint degrees along the
		// b- and c-components respectively. These are the exact sizes for M0/M1
		// (FC path second-edge indexes) and upper bounds on four-cliques.
		auto path_count_m0 = static_cast<size_t>(thrust::reduce(d_DE_b.begin(), d_DE_b.end()));
		auto path_count_m1 = static_cast<size_t>(thrust::reduce(d_DE_c.begin(), d_DE_c.end()));

		// Exact size per M vector; pre-filled with 1 to avoid resize during timing.
		d_M0.assign(path_count_m0, 1);
		d_M1.assign(path_count_m1, 1);

		// Upper bound: each M0 path yields at most one four-clique.
		d_quds.resize(path_count_m0 ? path_count_m0 : 1);

		// Materialise length-two paths for each component.
		// (OE_b, DE_b) =>  M_0, N_0
		cu_multi_arrange(d_DE_b, d_OE_b, d_N0, d_M0);
		// (OE_c, DE_c) =>  M_1, N_1
		cu_multi_arrange(d_DE_c, d_OE_c, d_N1, d_M1);

		// Do intersections to produce four-cliques
		cu_cliques_quds<<<nBlocks, MAX_THRD_BLK>>>(
				thrust::raw_pointer_cast(d_triangles.data()),
				thrust::raw_pointer_cast(d_triangles_offsets.data()),
				thrust::raw_pointer_cast(d_M0.data()),
				thrust::raw_pointer_cast(d_N0.data()),
				thrust::raw_pointer_cast(d_M1.data()),
				thrust::raw_pointer_cast(d_N1.data()),
				num_triangles,
				d_triangles_offsets.size(),
				thrust::raw_pointer_cast(d_quds.data()),
				d_quds.size());

		CU_ERR( cudaPeekAtLastError() );
		CU_ERR( cudaDeviceSynchronize() );
	}

	/**
	 * Parted (pipelined / multi-GPU) version of compute_fourcliques. GPU
	 * analogue of cliques_qud_par: tgraph.split_parts defines triangle-vertex
	 * ranges that are processed independently. Segments are distributed
	 * round-robin across the visible CUDA devices and each device is driven by
	 * its own host thread, so with N GPUs up to N segments run concurrently.
	 *
	 * On each device the triangle-graph CSR (trisD, trisO) and the triangle
	 * list (tris) are uploaded once and reused by every segment assigned to
	 * that device. Per-segment scratch (DE_b, DE_c, OE_b, OE_c, N0, N1, M0,
	 * M1, quds) is allocated, the per-device counter d_global_num_4cliques is
	 * reset, then the cu_take_quds_seg / cu_multi_arrange / cu_cliques_quds_seg
	 * pipeline runs and the resulting four-cliques are copied back to host.
	 * After all threads join the per-segment buffers are concatenated.
	 */
	template <typename U, typename V>
	std::vector<qud_t> compute_fourcliques_parted( const graph_t<U, V, 2>& tgraph
	                                             , const std::vector<tri_t>& tris )
	{
		if( tgraph.split_parts.empty() ) {
			throw std::runtime_error( "split_parts vector not initialized for parted call." );
		}

		int num_devices = 0;
		CU_ERR( cudaGetDeviceCount( &num_devices ) );
		if( num_devices < 1 ) {
			throw std::runtime_error( "No CUDA devices available." );
		}

		const size_t num_segments  = tgraph.split_parts.size();
		const size_t num_tverts    = tgraph.size_vertices();   // = tgraph.O.size() = tgraph.D.size()
		const size_t num_triangles = tgraph.size_edges();      // = tgraph.E.size() = tris.size()
		if( tris.size() != num_triangles ) {
			throw std::runtime_error( "compute_fourcliques_parted: tris.size() must equal tgraph.size_edges()." );
		}

		const size_t num_workers = std::min<size_t>( num_segments,
		                                             static_cast<size_t>( num_devices ) );

		std::vector<std::vector<qud_t>> per_seg( num_segments );
		std::mutex err_mtx;
		std::exception_ptr first_err = nullptr;

		int prev_device = -1;
		cudaGetDevice( &prev_device );

		auto worker = [&]( int dev )
		{
			try {
				CU_ERR( cudaSetDevice( dev ) );

				// Upload triangle graph + triangle list once per device.
				thrust::device_vector<VERTEX_T> d_trisD( tgraph.D );
				thrust::device_vector<EDGE_T>   d_trisO( tgraph.O );
				thrust::device_vector<tri_t>    d_tris ( tris.begin(), tris.end() );

				for( size_t s = static_cast<size_t>( dev ); s < num_segments; s += num_workers )
				{
					const size_t vb = tgraph.split_parts[s];
					const size_t ve = ( s + 1 < num_segments ) ? tgraph.split_parts[s+1] : num_tverts;
					if( ve <= vb ) continue;

					const size_t db = static_cast<size_t>( tgraph.O[vb] );
					const size_t de = ( ve < num_tverts ) ? static_cast<size_t>( tgraph.O[ve] ) : num_triangles;
					const size_t seg_size  = de - db; // number of triangles indexed by inner d/i1
					const size_t seg_verts = ve - vb;
					if( seg_size == 0 ) continue;

					thrust::device_vector<VERTEX_T> d_DE_b( seg_size, 0 );
					thrust::device_vector<VERTEX_T> d_DE_c( seg_size, 0 );
					thrust::device_vector<EDGE_T>   d_OE_b( seg_size, 0 );
					thrust::device_vector<EDGE_T>   d_OE_c( seg_size, 0 );
					thrust::device_vector<EDGE_T>   d_N0  ( seg_size + 1, 0 );
					thrust::device_vector<EDGE_T>   d_N1  ( seg_size + 1, 0 );

					// Reset this device's global four-clique counter. Segments
					// scheduled to the same device run serially within this
					// thread, so the shared counter is safe.
					initialize_kernel<<<1, 1>>>();
					CU_ERR( cudaDeviceSynchronize() );

					const dim3 nB_t( blocks( seg_size ), 1, 1 );
					cu_take_quds_seg<<<nB_t, MAX_THRD_BLK>>>(
							thrust::raw_pointer_cast( d_trisD.data() ),
							thrust::raw_pointer_cast( d_trisO.data() ),
							thrust::raw_pointer_cast( d_tris.data() ),
							db,
							seg_size,
							thrust::raw_pointer_cast( d_DE_b.data() ),
							thrust::raw_pointer_cast( d_DE_c.data() ),
							thrust::raw_pointer_cast( d_OE_b.data() ),
							thrust::raw_pointer_cast( d_OE_c.data() ) );

					CU_ERR( cudaPeekAtLastError() );
					CU_ERR( cudaDeviceSynchronize() );

					// Path counts (exact M0/M1 sizes, upper bound on quds).
					auto const path_count_m0 = static_cast<size_t>( thrust::reduce( d_DE_b.begin(), d_DE_b.end() ) );
					auto const path_count_m1 = static_cast<size_t>( thrust::reduce( d_DE_c.begin(), d_DE_c.end() ) );

					thrust::device_vector<MTYPE> d_M0( path_count_m0, 1 );
					thrust::device_vector<MTYPE> d_M1( path_count_m1, 1 );
					thrust::device_vector<qud_t> d_quds( path_count_m0 ? path_count_m0 : 1 );

					cu_multi_arrange( d_DE_b, d_OE_b, d_N0, d_M0 );
					cu_multi_arrange( d_DE_c, d_OE_c, d_N1, d_M1 );

					const dim3 nB_v( blocks( seg_verts ), 1, 1 );
					cu_cliques_quds_seg<<<nB_v, MAX_THRD_BLK>>>(
							thrust::raw_pointer_cast( d_tris.data() ),
							thrust::raw_pointer_cast( d_trisO.data() ),
							thrust::raw_pointer_cast( d_M0.data() ),
							thrust::raw_pointer_cast( d_N0.data() ),
							thrust::raw_pointer_cast( d_M1.data() ),
							thrust::raw_pointer_cast( d_N1.data() ),
							num_triangles,
							num_tverts,
							vb,
							ve,
							db,
							thrust::raw_pointer_cast( d_quds.data() ),
							d_quds.size() );

					CU_ERR( cudaPeekAtLastError() );
					CU_ERR( cudaDeviceSynchronize() );

					size_t seg_count = 0;
					CU_ERR( cudaMemcpyFromSymbol( &seg_count, d_global_num_4cliques
					                            , sizeof( d_global_num_4cliques ), 0
					                            , cudaMemcpyDeviceToHost ) );
					if( seg_count > d_quds.size() ) {
						throw std::runtime_error( "compute_fourcliques_parted: per-segment output capacity exceeded." );
					}

					per_seg[s].resize( seg_count );
					thrust::copy( d_quds.cbegin(), d_quds.cbegin() + seg_count, per_seg[s].begin() );
				}
			} catch (...) {
				std::lock_guard<std::mutex> lk( err_mtx );
				if( !first_err ) first_err = std::current_exception();
			}
		};

		std::vector<std::thread> threads;
		threads.reserve( num_workers );
		for( size_t i = 0; i < num_workers; ++i ) {
			threads.emplace_back( worker, static_cast<int>( i ) );
		}
		for( auto& t : threads ) t.join();

		if( prev_device >= 0 ) cudaSetDevice( prev_device );

		if( first_err ) std::rethrow_exception( first_err );

		size_t total = 0;
		for( const auto& v : per_seg ) total += v.size();
		std::vector<qud_t> all;
		all.reserve( total );
		for( auto& v : per_seg ) {
			all.insert( all.end(), v.begin(), v.end() );
		}
		return all;
	}

	// Returns free memory (bytes) on the specified device without changing the active device.
	static size_t gpu_free_memory(int device_id) {
		int prev = -1;
		cudaGetDevice(&prev);
		if (device_id != prev) cudaSetDevice(device_id);
		size_t free_bytes = 0, total_bytes = 0;
		cudaMemGetInfo(&free_bytes, &total_bytes);
		if (device_id != prev) cudaSetDevice(prev);
		return free_bytes;
	}

	// Enables unidirectional peer access so that kernels on `from_dev` can
	// directly read/write memory allocated on `to_dev` (e.g. via NVLink).
	// Returns true if peer access is now active.
	static bool try_peer_access(int from_dev, int to_dev) {
		int capable = 0;
		cudaDeviceCanAccessPeer(&capable, from_dev, to_dev);
		if (!capable) return false;
		int prev = -1;
		cudaGetDevice(&prev);
		cudaSetDevice(from_dev);
		cudaError_t err = cudaDeviceEnablePeerAccess(to_dev, 0);
		if (prev != from_dev) cudaSetDevice(prev);
		return err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled;
	}

	// Holds a raw CUDA device allocation that may live on a non-primary GPU.
	// Kernels on GPU 0 can access it via NVLink peer access.
	// When device == 0 (or no offload needed), ptr is a normal GPU 0 pointer.
	struct PeerAlloc {
		void*  ptr    = nullptr;
		int    device = 0;   // device that owns ptr
		size_t bytes  = 0;

		// Allocate `n_bytes` on the best available device:
		// prefer GPU 0 if it has >= MEM_SAFETY fraction free, otherwise spill
		// to the first peer_devices entry that has enough room.
		static PeerAlloc make(size_t n_bytes,
							const std::vector<int>& peer_devices,
							float mem_safety = 0.85f,
							const char* label = "")
		{
			PeerAlloc a;
			a.bytes = n_bytes;

			// Try primary device first
			if (n_bytes <= static_cast<size_t>(gpu_free_memory(0) * mem_safety)) {
				CU_ERR(cudaMalloc(&a.ptr, n_bytes));
				a.device = 0;
				return a;
			}

			// Spill to first peer with enough room
			for (int peer : peer_devices) {
				if (n_bytes <= static_cast<size_t>(gpu_free_memory(peer) * mem_safety)) {
					int prev = -1; cudaGetDevice(&prev);
					cudaSetDevice(peer);
					CU_ERR(cudaMalloc(&a.ptr, n_bytes));
					cudaSetDevice(prev);
					a.device = peer;
					std::cerr << "[multi-GPU] " << label << " ("
							<< n_bytes / (1ul << 20) << " MiB) on GPU " << peer << "\n";
					return a;
				}
			}

			// Last resort: allocate on GPU 0 anyway (will throw on OOM)
			std::cerr << "[multi-GPU] WARNING: no GPU has enough free memory for "
					<< label << " (" << n_bytes / (1ul << 20)
					<< " MiB). Trying GPU 0.\n";
			CU_ERR(cudaMalloc(&a.ptr, n_bytes));
			a.device = 0;
			return a;
		}

		void free_mem() {
			if (!ptr) return;
			int prev = -1; cudaGetDevice(&prev);
			if (device != prev) cudaSetDevice(device);
			cudaFree(ptr);
			if (device != prev) cudaSetDevice(prev);
			ptr = nullptr;
		}

		// Run thrust::sort on the owning device, then return to GPU 0.
		template <typename T>
		void sort_on_owner(size_t count) {
			int prev = -1; cudaGetDevice(&prev);
			if (device != prev) cudaSetDevice(device);
			thrust::sort(thrust::device_pointer_cast(static_cast<T*>(ptr)),
						thrust::device_pointer_cast(static_cast<T*>(ptr)) + count);
			if (device != prev) {
				CU_ERR(cudaDeviceSynchronize());
				cudaSetDevice(prev);
			}
		}

		template <typename T> T* as() { return static_cast<T*>(ptr); }
	};

	// Function to run nvidia-smi and send to cerr
	inline std::string nvidiaSmi() {
		// Execute command, capturing stdout
		std::unique_ptr<FILE, decltype(&pclose)> pipe(popen("nvidia-smi", "r"), pclose);
		if (!pipe) {
			return "";
		}
		std::string result;
		std::array<char, 128> buffer;
		while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
			result += buffer.data();   // Append to result string
		}
		return result;
	}

	std::vector<int> get_peer_devices() {
		int num_devices = 0;
		CU_ERR(cudaGetDeviceCount(&num_devices));
		std::vector<int> peers;
		for (int i = 1; i < num_devices; i++) {
			bool ok = try_peer_access(0, i) && try_peer_access(i, 0);
			if (ok) {
				peers.push_back(i);
			}
		}
		return peers;
	}

} // End of namespace cu
