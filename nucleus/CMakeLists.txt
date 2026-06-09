cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
# CUDA Standard 17 is only supported by cmake 3.18 and above

project(gpuNucleus34 LANGUAGES CXX) # CUDA is optional we enable in CUDA section

#=====================================================
# Generic flags
add_library(build_flags_cxx INTERFACE)
target_compile_options(build_flags_cxx INTERFACE
	-Werror
	-Wall
	-Wextra)

add_library(build_flags_cxx_gcc INTERFACE)

target_compile_options(build_flags_cxx_gcc INTERFACE
	-Wshadow
	-Wnon-virtual-dtor
    -Wold-style-cast
	-Woverloaded-virtual
	-Wzero-as-null-pointer-constant
    -pedantic
	-fPIE
	-fstack-protector-all
	-fno-rtti)


#set(CMAKE_CXX_STANDARD                  23)
#set(CMAKE_CXX_STANDARD_REQUIRED         ON)
#set(CMAKE_CXX_EXTENSIONS                OFF)

set(CMAKE_CXX_FLAGS_DEBUG				"-O0 -g")
set(CMAKE_CXX_FLAGS_MINSIZEREL			"-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE				"-DNDEBUG -pedantic -march=native -msse3 -ftree-vectorize -funroll-loops -O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO		"-O2 -g")


set(CMAKE_VERBOSE_MAKEFILE              TRUE CACHE BOOL "Verbose Output" FORCE)

message(STATUS "C++ Compiler: "	${CMAKE_CXX_COMPILER_ID})
message(STATUS "Compiler Version: " ${CMAKE_CXX_COMPILER_VERSION})

if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
	set(MIN_COMPILER_VERSION 17.0.0)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	set(MIN_COMPILER_VERSION 12.0.0)
endif()
	
if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS MIN_COMPILER_VERSION)
	message(FATAL_ERROR "Require at least ${MIN_COMPILER_VERSION} or higher.")
endif()


# Build-time datasets directory (downloaded large graphs, passed from super-build)
set(DATASETS_DIR "" CACHE PATH "Directory where the build-time datasets are stored")
if(DATASETS_DIR)
	add_definitions(-DDATASETS_DIR="${DATASETS_DIR}")
endif()

if(WIN32)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
	add_definitions(-D_USE_MATH_DEFINES)
	add_definitions(-DNDEBUG) # https://stackoverflow.com/a/32845763
endif(WIN32)

#=====================================================
# Datasets
set(DATASETS "" CACHE STRING "List of the dataset files")
message(STATUS "Datasets: ${DATASETS}")

#=====================================================
# Vertex type
set(VertexType "unsigned" CACHE STRING "Vertex data type")
set_property(CACHE VertexType PROPERTY STRINGS "unsigned" "unsigned_long")
message(STATUS "VertexType: ${VertexType}")

IF(${VertexType} STREQUAL "unsigned")
	add_definitions(-DVERTEX_U)
	remove_definitions(-DVERTEX_UL)
ENDIF()

IF(${VertexType} STREQUAL "unsigned_long")
	remove_definitions(-DVERTEX_U)
	add_definitions(-DVERTEX_UL)
ENDIF()

#=====================================================
# Edge type
set(EdgeType "unsigned" CACHE STRING "Edge data type")
set_property(CACHE EdgeType PROPERTY STRINGS "unsigned" "unsigned_long")
message(STATUS "EdgeType: ${EdgeType}")

IF(${EdgeType} STREQUAL "unsigned")
	add_definitions(-DEDGE_U)
	remove_definitions(-DEDGE_UL)
ENDIF()

IF(${EdgeType} STREQUAL "unsigned_long")
	remove_definitions(-DEDGE_U)
	add_definitions(-DEDGE_UL)
ENDIF()

#=====================================================
# M type
set(MultiArrangeType "mr_int" CACHE STRING "multi arrange data type")
set_property(CACHE MultiArrangeType PROPERTY STRINGS "mr_int" "mr_long" "mr_long_long")
message(STATUS "MultiArrangeType: ${MultiArrangeType}")

IF(${MultiArrangeType} STREQUAL "mr_int")
	add_definitions(-DM_I)
	remove_definitions(-DM_L)
	remove_definitions(-DM_LL)
ENDIF()

IF(${MultiArrangeType} STREQUAL "mr_long")
	add_definitions(-DM_L)
	remove_definitions(-DM_I)
	remove_definitions(-DM_LL)
ENDIF()

IF(${MultiArrangeType} STREQUAL "mr_long_long")
	add_definitions(-DM_LL)
	remove_definitions(-DM_I)
	remove_definitions(-DM_L)
ENDIF()

#=====================================================
# Execution Policy Option
set(ExecutionPolicy "parunseq" CACHE STRING "STD Execution Policy.")
set_property(CACHE ExecutionPolicy PROPERTY STRINGS seq par parunseq unseq)
message(STATUS "ExecutionPolicy: ${ExecutionPolicy}")

IF(${ExecutionPolicy} STREQUAL "seq")
	add_definitions(-DEXE_SEQ)
	remove_definitions(-DEXE_PAR)
	remove_definitions(-DEXE_PARUNSEQ)
	remove_definitions(-DEXE_UNSEQ)
ENDIF()

IF(${ExecutionPolicy} STREQUAL "par")
        remove_definitions(-DEXE_SEQ)
	add_definitions(-DEXE_PAR)
	remove_definitions(-DEXE_PARUNSEQ)
	remove_definitions(-DEXE_UNSEQ)
ENDIF()

IF(${ExecutionPolicy} STREQUAL "parunseq")
        remove_definitions(-DEXE_SEQ)
	remove_definitions(-DEXE_PAR)
	add_definitions(-DEXE_PARUNSEQ)
	remove_definitions(-DEXE_UNSEQ)
ENDIF()

IF(${ExecutionPolicy} STREQUAL "unseq")
        remove_definitions(-DEXE_SEQ)
	remove_definitions(-DEXE_PAR)
	remove_definitions(-DEXE_PARUNSEQ)
	add_definitions(-DEXE_UNSEQ)
ENDIF()

#=====================================================
# Add current folder to current directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#=====================================================
# Threading
find_package(Threads REQUIRED)

#=====================================================
# OpenMP
find_package(OpenMP REQUIRED)
include_directories(${OPENMP_INCLUDE_DIRS})
link_directories(${OPENMP_LIBRARY_DIRS})


#======================================================
# Intel oneTBB
set(ONETBB_PATH "${CMAKE_CURRENT_BINARY_DIR}/inetl_onetbb/install" CACHE PATH "Intel oneTBB install directory")
message(STATUS "Intel oneTBB install directory: ${ONETBB_PATH}")
find_package(TBB REQUIRED PATHS ${ONETBB_PATH} NO_DEFAULT_PATH)
message(STATUS "TBB Version: ${TBB_VERSION}")
message(STATUS "TBB DIR: ${TBB_DIR}")


#======================================================
# Other Components
add_subdirectory(graph)
add_subdirectory(utility)

#======================================================
# CUDA
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11.1.1")
	message(FATAL_ERROR "Require at least cuda 11.1.1 or higher.")
endif()

message(STATUS "CUDA compiler: " "${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}" )

if(NOT (${VertexType} STREQUAL "unsigned" OR ${EdgeType} STREQUAL "unsigned"))
	message(FATAL_ERROR "CUDA only supports unsigned type at the moment")
endif()

# In source code following macros can be used
# __CUDACC__  # will be defined on both host and device
# __CUDA_ARCH__ will be defined only on device only

#includes thrust and cuda runtime libs header file
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
SET(CMAKE_CUDA_PROPAGATE_HOST_FLAGS ON)
SET(CMAKE_CUDA_VERBOSE_BUILD ON)

#set cuda flags
#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  -ftemplate-backtrace-limit=0; --cudart;   static;  -O4 ; --expt-relaxed-constexpr; -fmad=true; --expt-extended-lambda; --relocatable-device-code=false; -Xptxas -dlcm=ca;-Xptxas --opt-level=4 )

# Enable calling constexpr function from device
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler -fopenmp")

message(STATUS "CUDA Standard: "	"${CMAKE_CUDA_STANDARD}" )

# This defines the flag in cache
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90 CACHE STRING "CUDA Architectures")

# This sets the actual architecture
set(CMAKE_CUDA_ARCHITECTURES 90)
message("CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")

# Detect CUDA architecture and get best NVCC flags
#INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindCudaArch.cmake)
#SELECT_NVCC_ARCH_FLAGS(NVCC_FLAGS_EXTRA)
#LIST(APPEND CUDA_NVCC_FLAGS ${NVCC_FLAGS_EXTRA})

# Enable using lambda function
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; --expt-extended-lambda;)

# Check if variable are set to the right type for CUDA
IF(NOT (${MultiArrangeType} STREQUAL "mr_int"))
	message(FATAL_ERROR "Cuda MR type expects mr_int but ${MultiArrangeType} provided.")
ENDIF()

add_executable(cuNucleus34 cuNucleus34.cu)
# -lgomp used for openmp in host code
target_link_libraries(cuNucleus34 PRIVATE
	CUDA::cuda_driver
	utils)

install(
	TARGETS
	cuNucleus34
	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX})

