#pragma once

#include "graph_api.hpp"

#include <algorithm>
#include <cassert>
#include <exception> // throw_with_nested
#include <execution>
#include <fstream>
#include <iterator> // istream_iterator
#include <numeric>
#include <sstream>

#include <omp.h>

#include "utility/def_types.hpp"
#include "utility/helpers.hpp"
#include "utility/def_exepol.hpp"

template <typename U, typename V, size_t DIM>
void graph_t<U, V, DIM>::copy_to(graph_t &g)
{
    // copy essentials information of graph
    g.E = this->E;
    g.D = this->D;
    g.O = this->O;
    g.S = this->S;
    g.split_parts = this->split_parts;
}

template <typename U, typename V, size_t DIM>
template <size_t NEW_DIM>
void graph_t<U, V, DIM>::reduce_to(graph_t<U, V, NEW_DIM> &g)
{
    static_assert(NEW_DIM < DIM,
                    "New dimension can not be smaller than original one.");
    // copy essentials information of graph
    g.E.resize(this->E.size());
    for (size_t i = 0; i < this->size_edges(); i++)
    {
        for (size_t s = 0; s < NEW_DIM; s++)
        {
            g.E[i][s] = this->E[i][s];
        }
    }
    g.D = this->D;
    g.O = this->O;
    g.S = this->S;
    g.split_parts = this->split_parts;
}

template <typename U, typename V, size_t DIM>
size_t graph_t<U, V, DIM>::bytes() const
{
    size_t m = sizeof(U)* O.capacity() +
                sizeof(V)* D.capacity() +
                sizeof(V)* S.capacity() +
                sizeof(std::array<V, DIM>)* E.capacity() +
                sizeof(size_t)* split_parts.capacity();
    return m;
}

template <typename U, typename V, size_t DIM>
void graph_t<U, V, DIM>::release()
{
    std::vector<U>().swap(O);
    std::vector<V>().swap(D);
    std::vector<V>().swap(S);
    std::vector<std::array<V, DIM>>().swap(E);
    std::vector<size_t>().swap(split_parts);
}

template <typename U, typename V, size_t DIM>
void graph_t<U, V, DIM>::resize(size_t nV, size_t nE,
            bool resize_offset /*= false*/,
            bool resize_source /*= false*/)
{

    D.resize(nV, 0);
    E.resize(nE, {0});
    split_parts.clear();
    if (resize_offset)
    {
        O.resize(nV, 0);
    }
    if (resize_source)
    {
        S.resize(nE, 0);
    }
}

template <typename U, typename V, size_t DIM>
graph_t<U, V, DIM>::graph_t(size_t nV, size_t nE)
{
    static_assert(DIM > 0,
                    "Graph dimension should be higher than zero.");
    resize(nV, nE);
}

template <typename U, typename V, size_t DIM>
graph_t<U, V, DIM>::graph_t(std::vector<std::array<V, DIM + 1>> &tp,
							size_t nV,
							bool oriented /*= true*/)
{

	static_assert(DIM > 0, "Graph dimension should be higher than zero.");

	sort(EXE_POL, tp, 2);

	this->resize(nV, oriented ? tp.size() : tp.size() * (DIM + 1), true);

	// Used aliasing to be shared in OpenMP
	auto &dd = this->D;
	auto &ee = this->E;

	// Don't use pragma omp here as it might be run insider a parallel region

	if (oriented)
	{
		//#pragma omp for //default(none) shared(dd, ee, tp)
		for (size_t i = 0; i < tp.size(); i++)
		{
			auto &v = dd[tp[i][0]];
			__sync_fetch_and_add(&v, 1);
			for (size_t s = 0; s < DIM; s++)
			{
				ee[i][s] = tp[i][s + 1];
			}
		} // for all tuples
		this->build_offset();
	}
	else
	{ // undirected graph
// first we build the degree (D) vector
#pragma omp parallel for default(none) shared(dd, tp)
		for (size_t i = 0; i < tp.size(); i++)
		{
			for (size_t s = 0; s < DIM + 1; s++)
			{
				auto &v = dd[tp[i][s]];
				__sync_fetch_and_add(&v, 1);
			}
		}

		// Then we build offset vector from degree
		this->build_offset();
		auto TO = O; // Copy of O vector

#pragma omp parallel for default(none) shared(tp, TO, ee)
		for (size_t i = 0; i < tp.size(); i++)
		{
			for (size_t n = 0; n < DIM + 1; n++)
			{
				const auto &id = tp[i][n];
				// Atomically reserve a single slot in id's adjacency region
				// before writing into it. Reserving first (rather than writing
				// to TO[id] and incrementing afterwards) prevents two threads
				// that process different tuples sharing the same vertex `id`
				// from grabbing the same slot, which previously corrupted the
				// adjacency non-deterministically.
				const auto slot = __sync_fetch_and_add(&TO[id], 1);
				size_t dim = 0;
				for (size_t s = 0; s < DIM + 1; s++)
				{
					const auto &id_neighbor = tp[i][s];
					if (id != id_neighbor)
					{
						ee[slot][dim++] = id_neighbor;
					}
				} // loop thru each item's neighbor
			}								 // loop thru tuple items
		}									 // loop thru tuple's vector
	}										 // condition-undirected graph
}

template <typename U, typename V, size_t DIM>
void graph_t<U, V, DIM>::make_oriented()
{
	static_assert(DIM == 1, "Only graph with dimension 1 is supported.");

	if (!this->O.size())
	{
		this->build_offset();
	}

	// copy essentials information of graph
	graph_t<U, V, DIM> graph;
	this->copy_to(graph);
	auto nV = this->size_vertices();
	auto nE = this->size_edges();
	bool sexist = this->S.size();

	this->release();
	this->resize(nV, nE / 2, true);

	size_t n = 0;
	for (size_t u = 0; u < graph.O.size(); u++)
	{
		auto bg = graph.O[u];
		auto sz = graph.D[u];
		this->O[u] = n;
		size_t osz = 0;
		for (size_t j = 0; j < sz; j++)
		{
			auto v = graph.E[bg + j][0];
			if (graph.vertex_less(static_cast<V>(u), v))
			{
				this->E[n++][0] = v;
				osz++;
			}
		}
		this->D[u] = osz;
	}
	if (sexist)
	{
		this->build_source();
	}
}


template <typename U, typename V, size_t DIM>
void graph_t<U, V, DIM>::make_undirected()
{
	static_assert(DIM == 1, "Only graph with dimension 1 is supported.");

	bool o_exist = this->O.size() != 0;

	if (!this->S.size())
	{
		this->build_source();
	}

	// Copy essentials information of graph

	// Create TS and TE vector twice the size of originals edges
	std::vector<V> TS(this->size_edges() * 2);
	std::vector<V> TE(this->size_edges() * 2);
	
	// Copy S and E Vector to TS
	std::copy(S.cbegin(), S.cend(), TS.begin());
	cpy(E, TS, this->size_edges());

	// Copy E and S Vector to TE
	cpy(E, TE);
	std::copy(S.cbegin(), S.cend(), TE.begin() + this->size_edges());

	auto nV = this->size_vertices();

	// stable sort permutation first on TS and then TE
	auto perm = sort_permutation(EXE_POL, TS.cbegin(), TS.cend(), TE.cbegin());

	// Apply the sort permutation to TS and TE
	auto S_s = take(TS, perm); // S_s = TS[perm]
	auto E_s = take(TE, perm); // E_s = TE[perm]

	// Remove duplicates of pairs of both vectors
	auto id = unique_indices(S_s.cbegin(), S_s.cend(), E_s.cbegin());

	// Reinitialize the graph vectors
	this->release();

	S = take(S_s, id); // S = S_s[id]
	
	// E = E_s[id]
	auto E_final = take(E_s, id); 
	E.resize(E_final.size());
	cpy(E_final, E);

	// Construct Degree vector
	this->D.resize(nV, 0);
	#pragma omp parallel for default(none) shared(D, S)
	for (size_t i = 0; i < S.size(); i++)
	{
		auto &d = D[S[i]];
		__sync_fetch_and_add(&d, 1);
	}

	if (o_exist)
	{
		this->build_offset();
	}
}


template <typename U, typename V, size_t DIM>
void graph_t<U, V, DIM>::from_edges(const std::string &filename,
								  bool mtx_header /*= true*/,
								  bool build_offset /*= true*/,
								  bool build_source /*= false*/,
								  bool undirected /*= true*/,
								  const std::string &comment_chars /*= "#%"*/)
{

	static_assert(DIM == 1, "Only graph with dimension 1 is supported.");

	std::ifstream fs(filename);

	if (!fs.is_open())
	{
		throw std::runtime_error("Dataset " + filename + " is not accessible!");
	}

	std::string line;

	size_t nVtx = 0, nEdge = 0;

	// Skip comments at the start of the file
	while (std::getline(fs, line))
	{
		if (comment_chars.find(line[0]) == std::string::npos)
		{
			break;
		}
	}

	std::vector<std::array<V, 2>> edges;

	if (mtx_header)
	{
		// Read header
		std::istringstream iss(line);
		std::vector<std::string> tokens{
			std::istream_iterator<std::string>{iss},
			std::istream_iterator<std::string>{}};

		if (tokens.size() == 2)
		{
			nVtx = static_cast<size_t>(std::stoul(tokens[0], nullptr, 0));
			nEdge = static_cast<size_t>(std::stoul(tokens[1], nullptr, 0));
		}
		else if (tokens.size() == 3)
		{
			nVtx = static_cast<size_t>(std::max(std::stoul(tokens[0], nullptr, 0),
												std::stoul(tokens[1], nullptr, 0)));
			nEdge = static_cast<size_t>(std::stoul(tokens[2], nullptr, 0));
		}
		else
		{
			fs.close();
			throw std::runtime_error("Wrong header for mtx format!");
		}

		if (std::numeric_limits<U>::max() < nEdge)
		{
			throw std::runtime_error("Edges size are larger than defined type.");
		}

		if (std::numeric_limits<V>::max() < nVtx)
		{
			throw std::runtime_error("Vertices id are larger than defined type.");
		}

		edges.reserve(nEdge * (undirected ? 2 : 1));
	}

	// read each edge and add an inverse edge to the graph
	V max_node = 0;
	size_t ln = 0;
	while (std::getline(fs, line))
	{
		std::stringstream ss(line);
		V s, d;
		ss >> s >> d;
		// Test for self-loop
		if (s != d)
		{
			max_node = std::max(max_node, s);
			max_node = std::max(max_node, d);
			edges.push_back({s, d});
			if (undirected)
			{
				edges.push_back({d, s});
			}
		}
		ln++; // Number of lines
	}		  // Read until end of the file

	fs.close();

	// Shrink the edges list to actual size
	edges.shrink_to_fit();

	// Sort pairs
	sort(EXE_POL, edges);

	// Remove duplicates
	auto it = std::unique(edges.begin(), edges.end());
	edges.erase(it, edges.end());

	// Construct the graph vectors
	// +1 since the indices start from zero
	this->resize(max_node + 1, edges.size(), build_offset, build_source);

	// Split the edges to S and E vector
	// Initialize D vector
	for (size_t i = 0; i < edges.size(); i++)
	{
		if (build_source)
		{
			this->S[i] = edges[i][0];
		}
		this->E[i][0] = edges[i][1];
		this->D[edges[i][0]]++;
	}

	if (build_offset)
	{
		// Build offset
		this->build_offset();
	}
}


template <typename U, typename V, size_t DIM>
void graph_t<U, V, DIM>::to_edges(const std::string &filename,
									bool mtx_header /*= true*/)
{

	static_assert(DIM == 1, "Only graph with dimension 1 is supported.");
	strip_self_loops(); // will also build the source vector

	std::ofstream fs(filename);
	if (!fs.is_open())
	{
		throw std::runtime_error("Can't open the file for writing.");
	}
	if (mtx_header)
	{
		fs << "%%MatrixMarket matrix coordinate pattern general integer" << std::endl;
		fs << size_vertices() << '\t' << size_vertices() << '\t' << size_edges() << std::endl;
	}
	for (size_t i = 0; i < this->size_edges(); i++)
	{
		if (S[i] != E[i][0])
		{
			fs << S[i] << '\t' << E[i][0] << std::endl;
		}
	}
	fs.close();
}

template <typename U, typename V, size_t DIM>
void graph_t<U, V, DIM>::strip_self_loops()
{
	static_assert(DIM == 1, "Only graph with dimension 1 is supported.");
	this->build_source();
	for (size_t i = 0; i < size_edges(); i++)
	{
		if (E[i][0] == S[i])
		{
			auto v = S[i];
			S.erase(S.begin() + i);
			E.erase(E.begin() + i);
			D[v]--;
			i--;
		}
	}
	if (!O.empty())
	{
		this->build_offset();
	}
}


template <typename U, typename V, size_t DIM>
void graph_t<U, V, DIM>::serialize(const std::string &filename)
{
	static_assert(DIM == 1, "Only graph with dimension 1 is supported.");

	std::ofstream ofl(filename, std::ios::out | std::ios::binary);
	if (!ofl.is_open())
	{
		throw std::runtime_error("Can't open file for serializing.");
	}
	// return type of sizeof is size_t
	size_t sv = sizeof(V);
	ofl.write(reinterpret_cast<char *>(&sv), sizeof(size_t));

	size_t nV = this->size_vertices();
	size_t nE = this->size_edges();
	ofl.write(reinterpret_cast<char *>(&nV), sizeof(size_t));
	ofl.write(reinterpret_cast<char *>(&nE), sizeof(size_t));

	ofl.write(reinterpret_cast<char *>(&D[0]), D.size() * sizeof(V));
	ofl.write(reinterpret_cast<char *>(&E[0][0]), E.size() * sizeof(V));

	ofl.close();
}


template <typename U, typename V, size_t DIM>
void graph_t<U, V, DIM>::deserialize(const std::string &filename)
{
	static_assert(DIM == 1, "Only graph with dimension 1 is supported.");

	std::ifstream ifl(filename, std::ios::in | std::ios::binary);

	if (!ifl.is_open())
	{
		throw std::runtime_error("Can't open file for reading.");
	}

	size_t sv; // return type of sizeof is size_t
	ifl.read(reinterpret_cast<char *>(&sv), sizeof(size_t));

	if (sizeof(V) != sv)
	{
		throw std::runtime_error("Incompatible types in dataset.");
	}

	size_t nV, nE;
	ifl.read(reinterpret_cast<char *>(&nV), sizeof(size_t));
	ifl.read(reinterpret_cast<char *>(&nE), sizeof(size_t));

	this->resize(nV, nE);
	ifl.read(reinterpret_cast<char *>(&D[0]), D.size() * sizeof(V));
	ifl.read(reinterpret_cast<char *>(&E[0][0]), E.size() * sizeof(V));

	ifl.close();
}


template <typename U, typename V, size_t DIM>
void graph_t<U, V, DIM>::build_offset()
{
	if (O.size() != size_vertices())
	{
		O.resize(size_vertices());
	}
	std::exclusive_scan(EXE_POL,
						D.cbegin(),
						D.cend(),
						O.begin(),
						0,
						std::plus<>());
}


template <typename U, typename V, size_t DIM>
void graph_t<U, V, DIM>::update_offset(size_t id, const U &u)
{
	std::transform(EXE_POL,
					O.cbegin() + id,
					O.cend(),
					O.begin() + id,
					[&u](const auto &val)
					{ return val + u; });
}


template <typename U, typename V, size_t DIM>
void graph_t<U, V, DIM>::build_source()
{
	if (O.empty())
	{
		this->build_offset();
	}
	if (S.size() != size_edges())
	{
		S.resize(size_edges(), 0);
	}
	#pragma omp parallel for default(none) shared(S, O, D)
	for (size_t v = 0; v < size_vertices(); v++)
	{
		for (V i = 0; i < D[v]; i++)
		{
			S[O[v] + i] = v;
		}
	}
}

template <typename U, typename V, size_t DIM>
V graph_t<U, V, DIM>::at_source(V id) const
{
	return static_cast<V>(
		std::distance(O.cbegin(),
						std::upper_bound(O.cbegin(), O.cend(), id)) -
		1);
}

template <typename U, typename V, size_t DIM>
void graph_t<U, V, DIM>::make_split_parts(size_t parts /*=0*/)
{
	size_t segments = (parts == 0) ? static_cast<size_t>(omp_get_num_procs()) : parts;

	if (O.empty())
	{
		this->build_offset();
	}

	// initialize the vector to maximum possible value (len(O)-1)
	// except the first element that should be zero
	split_parts.resize(segments, this->O.size() - 1);
	split_parts[0] = 0;

	auto scale = 1. / static_cast<double>(segments);
	const auto &deg_sum = static_cast<double>(this->O.back());
	for (size_t i = 0, j = 1; i < O.size(); i++)
	{
		if (this->O[i] / deg_sum > static_cast<double>(j) * scale)
		{
			split_parts[j++] = i;
		}
	}
}

template <typename U, typename V, size_t DIM>
void graph_t<U, V, DIM>::push_back_vertex()
{
	static_assert(DIM == 1,
					"This operation is only available for graph with dim one.");
	if (!O.empty())
	{
		auto val = O.back();
		O.push_back(val + D.back());
	}
	D.push_back(0);
}

template <typename U, typename V, size_t DIM>
void graph_t<U, V, DIM>::delete_vertex(const V &v)
{
	static_assert(DIM == 1, "This operation is only available for graph with dim one.");

	ASERT(D.size() > v, "Vertex not exist.");
	ASERT(O.size(), "The offset vector need to materialized.");

	auto &d = D[v];
	auto &o = O[v];

	// Remove edges originates from vertex v
	E.erase(E.begin() + o, E.begin() + o + d);
	d = 0; // Zero degree of vertex v

	// Remove edges that go to vertex v
	U e = 0;
	for (size_t i = 0; i < D.size(); i++)
	{
		const auto di = D[i]; // copy o D[i]
		for (V j = 0; j < di; j++)
		{
			if (E[e][0] == v)
			{
				E.erase(E.begin() + e);
				D[i]--;
			}
			else
			{
				e++;
			}
		}
	}
	// Rebuilds the offset
	this->build_offset();
}

template <typename U, typename V, size_t DIM>
bool graph_t<U, V, DIM>::query_edge(const V &a, const V &b, size_t &id)
{
	static_assert(DIM == 1, "Only graph with dimension 1 is supported.");
	if (a == b)
	{
		throw std::runtime_error("Query not permitted for self-loops!");
	}

	if (a >= size_vertices())
	{
		throw std::runtime_error("Vertex " +
									std::to_string(a) +
									" not found.");
	}

	if (b >= size_vertices())
	{
		throw std::runtime_error("Vertex " +
									std::to_string(b) +
									" not found.");
	}

	// At this point we are sure that vertices_size() > 0
	if (O.empty())
	{
		throw std::runtime_error("This method requires offset vector.");
	}

	const auto &d = D[a];
	const auto &o = O[a];
	for (id = o; id < d + o; id++)
	{
		const auto &u = E[id][0];
		if (u < b)
		{
			continue;
		}
		return u == b;
	}
	return false;
}

template <typename U, typename V, size_t DIM>
void graph_t<U, V, DIM>::insert_edge(const V &a, const V &b, bool insert_source /*= true*/)
{
	static_assert(DIM == 1, "Only graph with dimension 1 is supported.");

	size_t pos;
	if (query_edge(a, b, pos))
	{
		throw std::runtime_error("The edge already exist!");
	}

	auto &d = D[a];
	d++;
	this->update_offset(a + 1, 1);
	E.insert(E.begin() + pos, {b});
	if (insert_source)
	{
		S.insert(S.begin() + pos, a);
	}
}

template <typename U, typename V, size_t DIM>
void graph_t<U, V, DIM>::delete_edge(const V &a, const V &b)
{
	static_assert(DIM == 1, "Only graph with dimension 1 is supported.");

	size_t pos;
	if (!query_edge(a, b, pos))
	{
		throw std::runtime_error("The edge doesn't exist!");
	}
	E.erase(E.begin() + pos);
	if (!S.empty())
	{
		S.erase(S.begin() + pos);
	}
	D[a] -= 1;
	this->update_offset(a + 1, -1);
}


template <typename U, typename V, size_t DIM>
void graph_t<U, V, DIM>::sort_by_degree(bool undirected /*= false*/)
{
	static_assert(DIM == 1, "Only graph with dimension 1 is supported.");

	if(!undirected) {
		// Only applies to undirected graphs
		// Also requires source vector
		this->make_undirected();
	}

	auto perm = sort_permutation(EXE_POL, D.cbegin(), D.cend(), false);
	auto perm_of_perm = sort_permutation(EXE_POL, perm.cbegin(), perm.cend(), true);

	std::vector<std::array<V, 2>> T(size_edges());
	//#pragma omp parallel for
	for (size_t i = 0; i < size_edges(); i++)
	{
		T[i][0] = perm_of_perm[S[i]];
		T[i][1] = perm_of_perm[E[i][0]];
	}

	sort(EXE_POL, T);

	//#pragma omp parallel for
	for (size_t i = 0; i < size_edges(); i++)
	{
		S[i] = T[i][0];
		E[i][0] = T[i][1];
	}

	// Sort Degree Descending
	sort(EXE_POL, D.begin(), D.end(), [](const auto &i1, const auto &i2)
			{ return i1 > i2; });

	D.erase(std::remove(D.begin(), D.end(), 0), D.end());

	this->build_offset();
}


/**
 * @brief Print graph to standard output
*/
template <typename E, typename V, size_t DIM>
std::ostream &operator<<(std::ostream &out, const graph_t<E, V, DIM> &g)
{
	out << "|V_" << DIM << "|: " << g.size_vertices() << std::endl
		<< "|E_" << DIM << "|: " << g.size_edges() << std::endl;
	if (g.O.size())
		out << "O_" << DIM << ": " << g.O << std::endl;
	out << "D_" << DIM << ": " << g.D << std::endl;
	if (g.S.size())
		out << "S_" << DIM << ": " << g.S << std::endl;
	out << "E_" << DIM << ": " << g.E << std::endl;
	return out;
}