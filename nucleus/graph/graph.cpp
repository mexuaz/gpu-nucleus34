#include "graph.hpp"

// Explicit instantiation

template void graph_t<EDGE_T, VERTEX_T, 1>::make_oriented();

template void graph_t<EDGE_T, VERTEX_T, 1>::make_undirected();

template void graph_t<EDGE_T, VERTEX_T, 1>::from_edges(const std::string &filename,
								  bool mtx_header /*= true*/,
								  bool build_offset /*= true*/,
								  bool build_source /*= false*/,
								  bool undirected /*= true*/,
								  const std::string &comment_chars /*= "#%"*/);

template void graph_t<EDGE_T, VERTEX_T, 1>::to_edges(const std::string &filename,
	bool mtx_header /*= true*/);

template void graph_t<EDGE_T, VERTEX_T, 1>::strip_self_loops();

template void graph_t<EDGE_T, VERTEX_T, 1>::serialize(const std::string &filename);

template void graph_t<EDGE_T, VERTEX_T, 1>::deserialize(const std::string &filename);

template void graph_t<EDGE_T, VERTEX_T, 1>::build_offset();

template void graph_t<EDGE_T, VERTEX_T, 1>::update_offset(size_t id, const EDGE_T &u);

template void graph_t<EDGE_T, VERTEX_T, 1>::build_source();

