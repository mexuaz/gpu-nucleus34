#pragma once

#include <array>
#include <string>
#include <vector>

/**
 * A class representing directed and undirected graphs based on Destination vector
 * and Degree vectors, single dangling nodes can be represented by
 * self-loop edge with degree zero
 * @tparam U is type to address neighbors list and based on max edges size
 * @tparam V is type to address large vertices ids and degree
 * Degree can't be greater than total vertices number so it should be the same
 * type as V
 * @tparam DIM Dimension of the graph greater or equal to one
 */
template <class U, class V, size_t DIM = 1>
struct graph_t
{

	/**
	 * @brief Offset of destination vector: starting position of
	 *  neighbors in destination vector for an specific vertex
	 * This is optional vector and can be make with exclusive_scan(R)
	 */
	std::vector<U> O;

	/**
	 * @brief Degree vector: length of neighbors for an specific vertex
	 *  in destination vector for an specific vertex
	 */
	std::vector<V> D;

	/**
	 * @brief sorted Source and destination vectors together build
	 *  the edges list pairs.
	 * S is an optional vector.
	 * E vector could be extended to multiple destination vectors in case
	 *  of higher cliques like triangles (3-cliques) graph has two T vectors
	 */
	std::vector<V> S;

	/**
	 * @brief E Destination Vector
	 */
	std::vector<std::array<V, DIM>> E;

	/**
	 * @brief split_parts Vector that contains values in which Offset vector
	 * will be split.
	 */
	std::vector<size_t> split_parts;

	/**
	 * @brief copy_to copy the graph to another graph
	 * @param g new graph
	 */
	void copy_to(graph_t &g);

	/**
	 * @brief reduce_to reduce the dimension of the graph
	 * @tparam NEW_DIM new dimension
	 * @param g new graph
	 */
	template <size_t NEW_DIM>
	void reduce_to(graph_t<U, V, NEW_DIM> &g);

	/**
	 * @brief mem_bytes return the approximate amount of the money that
	 * member contents occupies
	 * @return bytes of memory occupied
	 */
	[[nodiscard]] size_t bytes() const;

	/**
	 * @brief release free up all the memory occupied by graph
	 */
	void release();

	/**
	 * @brief resize all the graph data by resetting vectors,
	 * and resize the essential graphs
	 * @param nV size of vertices
	 * @param nE size of edges
	 * @param resize_offset resize the offset vector which is optional
	 * @param resize_source resize the source vector which is optional
	 */
	void resize(size_t nV, size_t nE,
				bool resize_offset = false,
				bool resize_source = false);

	/**
	 * @brief graph_t
	 * @param nV vertices size
	 * @param nE edges size
	 */
	graph_t(size_t nV, size_t nE);

	graph_t() = default;
	~graph_t() = default;

	/**
	 * @brief graph_t construct undirected  or oriented graph_t from given vector of tuples
	 * oriented graphs are used to extract cliques and undirected graphs are used in peeling process
	 * @param tp tuples (pairs, triples, quadruples or ...) vector, the function
	 * will sort the given vector in place and use them to construct graph_t
	 * @param nV maximum value of vertices, this will be used as the size of
	 * degree (D) and offset (O) vector in the graph_t
	 */
	graph_t(std::vector<std::array<V, DIM + 1>> &tp,
			size_t nV,
			bool oriented = true);

	graph_t(const graph_t &) = delete;			  // copy constructor
	graph_t &operator=(const graph_t &) = delete; // copy assignment

	graph_t(graph_t &&) = delete;			 // move constructor
	graph_t &operator=(graph_t &&) = delete; // move assignment

	[[nodiscard]] inline size_t size_vertices() const { return D.size(); }
	[[nodiscard]] inline size_t size_edges() const { return E.size(); }
	[[nodiscard]] inline size_t size_destination() const { return E.size() * DIM; }

	inline bool vertex_less(const V &s, const V &d) const
	{
		return D[s] < D[d] || (D[s] == D[d] && s < d);
	}

	/**
	 * @brief make_oriented convert undirected graph of DIM=1 to
	 * oriented graph of DIM=1. In undirected graph (if there is a edge
	 * from u to v, there is another edge from v to u) but in oriented
	 * graph if u and v vertices are connected, there is only one
	 * edge and it is from less vertex to other one.
	 * less vertex is defined in the also defined in the graph_t
	 */
	void make_oriented();

	/**
	 * @brief make_undirected convert oriented graph of DIM=1 to
	 * undirected graph of DIM=1. In undirected graph (if there is a edge
	 * from u to v, there is another edge from v to u) but in oriented
	 * graph if u and v vertices are connected, there is only one
	 * edge and it is from less vertex to other one.
	 * less vertex is defined in the also defined in the graph_t
	 *
	 * This is relatively expensive operation and it will create
	 *  Source and Offset vectors even if they don't exist
	 */
	void make_undirected();

	/**
	 * @brief from_edges Reads a dataset file in txt/mtx/edges format and
	 *  initialize a graph of DIM 1, This functions filters all
	 *  duplicate edges.
	 * @note This method ignores all self loops
	 * @cite https://math.nist.gov/MatrixMarket/formats.html
	 * @param filename dataset filename
	 * @param mtx_header weather the file include mtx header or not
	 * @param build_offset initialize offset vector when building the graph
	 * @param build_source initialize source vector when building the graph
	 * @param undirected Whether to read in undirected or not
	 * @param comment_chars characters to be considers comment specifiers at the beginning of dataset
	 * Undirected graph means for each edge from u to v there is also an
	 *  inverse edge form v ot u.
	 * @return Returns true if successes
	 */
	void from_edges(const std::string &filename,
				  bool mtx_header = true,
				  bool build_offset = true,
				  bool build_source = false,
				  bool undirected = true,
				  const std::string &comment_chars = "#%");

	/**
	 * @brief to_mtx Saves the graph to mtx format.
	 * @note The method filters out all self loops
	 * @attention Calling this method will initialize the source vector if
	 *  does not exist
	 * @cite https://math.nist.gov/MatrixMarket/formats.html
	 * @param filename dataset file
	 * @param mtx_header Wether the file should include mtx header or not
	 */
	void to_edges(const std::string &filename, bool mtx_header = true);

	/**
	 * @brief strip_self_loops strip all self-loops vertices
	 * @attention Calling this method will initialize the source vector if
	 *  does not exist
	 */
	void strip_self_loops();

	/**
	 * \brief serialize We only serialize E[0] and D, their sizes and
	 * their types which is V
	 * Offset vector with type U could be constructed using D
	 * Source vector could also be made using deg but we don't need that
	 * in our computations
	 * \param filename
	 */
	void serialize(const std::string &filename);
	void deserialize(const std::string &filename);

	void build_offset();

	/**
	 * @brief update_offset Adds const u to all the elements of
	 * offset vector starting form index id
	 * @param id The start index to apply the addition
	 * @param u The value to add to all elements
	 */
	void update_offset(size_t id, const U &u);

	/**
	 * @brief build_source Will build the Source vector and Offset vector
	 * if it is not built
	 */
	void build_source();

	/**
	 * @brief at_source Returns the source vertex value for the given edge id.
	 * This is an alternative to src vector to lookup for src value when source
	 * vector is not build for memory deficiency re-scans
	 * @param id edge id
	 * @return the source vertex value for the given edge id
	 */
	V at_source(V id) const;

	/**
	 * @brief make_split_parts Fill split_parts vector with indices in which
	 * Offset vector will be split for parallel processing of clique
	 * counting. Calling this function is required before calling _par
	 * function calls to initialize split_parts vector.
	 * @param parts Number of expected splits. n should be greater or equal to
	 * one.
	 * If parts=0 n will be set to omp_get_num_procs()
	 * If parts=1 the behaviour of _par call will be like a sequential one.
	 */
	void make_split_parts(size_t parts = 0);

	/**
	 * @brief push_back_vertex Inserts a new dangling vertex to the graph
	 */
	void push_back_vertex();

	/**
	 * @brief Remove all the edges connected to vertex and vertex itself
	 *
	 * @param v the vertex to be deleted
	 */
	void delete_vertex(const V &v);
	
	/**
	 * @brief query_edge
	 * @param a
	 * @param b
	 * @param id The position of edge if the edge exist otherwise the
	 * position where edge should be located (Since Source and Destination
	 * vectors are sorted there is only specific location where edge could
	 * be found). In case edge does not exist this id could be used to
	 * insert the new edge
	 * Requires Offset Vector
	 * @throws
	 *  1) If a is equal to b (self-loop)
	 *  2) Either vertex a or b does not  exist
	 *  3) Offset vector dose not initialized
	 * @return Returns true if edge exist otherwise false
	 */
	bool query_edge(const V &a, const V &b, size_t &id);

	/**
	 * @brief insert_edge Insert a directed edge from vertex a to b
	 * Requires Offset Vector
	 * @param a Source vertex
	 * @param b Destination vertex
	 * @throws If the edge exist it will throw runtime error
	 */
	void insert_edge(const V &a, const V &b, bool insert_source = true);

	/**
	 * @brief delete_edge
	 * Requires offset vector
	 * @param a
	 * @param b
	 * @throws If the edge doesn't exist
	 */
	void delete_edge(const V &a, const V &b);


	/**
	 * @brief Sort Vertices by degree, the vertex with lower value will have the highest degree, the vertices with zero value will be removed
	 * The method by default will attempt to make the graph undirected
	 * @param undirected hint that the graph that is undirected so it will not attempt to make it undirected first
	 * if the graph is not undirected and undirected graph is true the result might be invalid
	 */
	void sort_by_degree(bool undirected = false);
	
	// default extension used for serialization
	//// @todo For C++20 change this to constexpr
	inline static const std::string ext = ".grh";
};
