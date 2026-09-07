#ifndef DEFS_HPP_
#define DEFS_HPP_

#include <execution>
#include <chrono>
#include <sstream>
#include <iostream>
#include <vector>
#include <array>
#include <algorithm> // copy(...)
#include <cmath>


// Timing constants
#define TM_RDATA "read data"
#define TM_TRIANGLES "triangles"
#define TM_FOURCLIQUES "four cliques"
#define TM_TRI_GRAPH_O "triangles o-graph"
#define TM_TRI_GRAPH_U "triangles u-graph"
#define TM_QUAD_GRAPH_U "four cliques u-graph"
#define TM_PRE_PEELING "pre peeling"
#define TM_PEELING "peeling"
#define TM_ALL "all"

#define PRINT_CAP 5 // Max number of elements to print in a container

#ifndef NDEBUG
#define DEBUG 1
#else
#define DEBUG 0
#endif

// @cite: https://stackoverflow.com/a/1644898
#define TRACE__(fmt, ...) \
            do { if (DEBUG) fprintf(stderr, fmt, __VA_ARGS__); } while (0)

#define TRACE2__(fmt, ...) \
        do { if (DEBUG) fprintf(stderr, "%s:%d:%s(): " fmt, __FILE__, \
                                __LINE__, __func__, __VA_ARGS__); } while (0)

#define MSG__(fmt, ...) \
            do { if (DEBUG) printf(fmt, __VA_ARGS__); } while (0)

#define MSG2__(fmt, ...) \
        do { if (DEBUG) printf("%s:%d:%s(): " fmt, __FILE__, \
                                __LINE__, __func__, __VA_ARGS__); } while (0)

#ifndef NDEBUG
#       define ASERT(Expr, Msg) \
	_ASERT_(#Expr, Expr, __FILE__, __FUNCTION__, __LINE__, Msg)
#else
#       define ASERT(Expr, Msg);
#endif

void _ASERT_(const char* expr_str,
		 bool expr,
		 const char* file,
		 const char* func,
		 size_t line,
		 const std::string& msg);




template <typename T, typename RET=T>
inline RET round(T val, size_t precision=2) {
	auto tens = std::pow(10., precision);
	return static_cast<RET>(std::ceil(val * tens) / tens);
}
	
#define SET_STEP(s) do { step = (s); std::cerr << "[phase] " << step << std::endl; } while(0)



#endif // DEFS_HPP_
