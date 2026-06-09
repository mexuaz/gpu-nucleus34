cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(graph LANGUAGES CXX)

add_library(graph STATIC
	graph.cpp
	graph.hpp
	graph_api.hpp)

target_link_libraries(graph PRIVATE
	build_flags_cxx
	utils
	TBB::tbb)


target_compile_options(graph PRIVATE
	${OpenMP_CXX_FLAGS})

set_target_properties(graph PROPERTIES
	CXX_STANDARD 23
	CXX_STANDARD_REQUIRED ON
	CXX_EXTENSIONS OFF)

