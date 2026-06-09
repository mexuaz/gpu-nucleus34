cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(utility LANGUAGES CXX)

add_library(utils STATIC
	helpers.cpp
	defs.cpp
	cpuinfo.hpp
	ioutils.hpp
	json.h
	helpers.hpp
	helpers_api.hpp
	defs.hpp
	def_exepol.hpp
	def_system.hpp
	def_types.hpp
	def_ver.hpp)


target_link_libraries(utils PRIVATE
	build_flags_cxx
	TBB::tbb
	OpenMP::OpenMP_CXX)


# Benchmarks moved to test/CMakeLists.txt

set_target_properties(
	utils
	PROPERTIES
	CXX_STANDARD 23
	CXX_STANDARD_REQUIRED ON
	CXX_EXTENSIONS OFF)