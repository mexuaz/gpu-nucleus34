
#ifndef DEF_TYPES_HPP_
#define DEF_TYPES_HPP_

#include <chrono>

#ifdef M_I
#define MTYPE int
#elif M_L
#define MTYPE long
#elif M_LL
#define MTYPE long long
#endif

using INDEX_T = MTYPE; // TODO: Deprecate non-semantic MTYPE type

#ifdef VERTEX_U
#define VERTEX_T unsigned int
#elif VERTEX_UL
#define VERTEX_T unsigned long int
#endif

#ifdef EDGE_U
#define EDGE_T unsigned int
#elif EDGE_UL
#define EDGE_T unsigned long int
#endif

typedef long buk_t;

using hrc = std::chrono::high_resolution_clock;
using seconds = std::chrono::duration<double, std::ratio<1, 1>>;
using milliseconds = std::chrono::duration<double, std::ratio<1, 1'000>>;
using microseconds = std::chrono::duration<double, std::ratio<1, 1'000'000>>;

#endif // DEF_TYPES_HPP_