cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(nucleus)

include(ExternalProject)

set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build.")
set(CMAKE_CUDA_ARCHITECTURES 90 CACHE STRING "CUDA Architectures")

#======================================================
# Intel oneTBB (Threading Building Blocks)
# Intel oneTBB is required for parallel execution policy in Linux and Windows

set(_ONETBB_DIR "${CMAKE_CURRENT_BINARY_DIR}/intel_onetbb" CACHE PATH "Intel OneTBB root directory")
set(_ONETBB_INSTALL_DIR "${_ONETBB_DIR}/install" CACHE PATH "Intel OneTBB install directory")
set(_ONETBB_TAG "v2021.11.0" CACHE STRING "Intel OneTBB tag")

find_package(TBB PATHS ${_ONETBB_INSTALL_DIR} NO_DEFAULT_PATH)

if(TBB_FOUND)
 	message(STATUS "Intel oneTBB ${TBB_VERSION} found: ${_ONETBB_INSTALL_DIR}")
else(TBB_FOUND)
	message(STATUS "Will try building Intel oneTBB ${_ONETBB_TAG} from source.")	
	ExternalProject_Add(intel_onetbb
			GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
			GIT_TAG ${_ONETBB_TAG}
			GIT_SUBMODULES_RECURSE ON
			GIT_REMOTE_UPDATE_STRATEGY CHECKOUT
			PREFIX "${_ONETBB_DIR}"
			INSTALL_DIR "${_ONETBB_INSTALL_DIR}"
			LIST_SEPARATOR |
			CMAKE_CACHE_ARGS
			"-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}"
			"-DCMAKE_INSTALL_PREFIX:STRING=${_ONETBB_INSTALL_DIR}" 
			"-DTBB_TEST:BOOL=OFF"
			UPDATE_DISCONNECTED TRUE
			)
	# Create a stamp file to prevent unnecessary rebuilds
	ExternalProject_Add_Step(intel_onetbb mark_done
			COMMAND ${CMAKE_COMMAND} -E touch "${_ONETBB_DIR}/.onetbb_installed"
			DEPENDEES install
			)
endif(TBB_FOUND)

#======================================================
# Datasets
set(DATASETS_HOST_NETR "https://nrvis.com/download/data/" CACHE STRING "URL of the networkrepository.com dataset host.")
set(DATASETS_EDGE_NETR "" CACHE STRING "Names of the dataset files")
list(APPEND DATASETS_EDGE_NETR
		"soc/soc-Slashdot0811.zip"		# 2.9MB zip,	9.4MB,			77.4K nodes,	828.2K edges,	5.8M triangles
		"soc/soc-BlogCatalog.zip"		# 5.8MB zip,	20.4MB mtx,		88.8K nodes,	2.1M edges,		153.6M triangles
		"soc/soc-FourSquare.zip"		# 9.9MB zip,	38.3MB mtx,		639K nodes,		3.2M edges,		65M triangles
		"misc/amazon-2008.zip"			# 13.8MB zip,	67.4MB mtx,		735.3K nodes,	5.2M edges,		34.5M triangles	
		"soc/soc-digg.zip"				# 19.3MB zip,	71.8MB mtx,		303K nodes,		1.7M edges,		1.1B triangles
		"cit/cit-patent.zip"			# 73.4MB zip,	244.4MB edges,	3.8M nodes,		16.5M edges,	22.5M triangles
		"soc/soc-LiveJournal1.zip"		# 261.5MB zip,	1019.8MB edges,	4.8M nodes,		69M edges,		4.3B triangles
		)
list(LENGTH DATASETS_EDGE_NETR DATASETS_COUNT_NETR)
set(DATASETS_DIR "${CMAKE_CURRENT_BINARY_DIR}/datasets" CACHE PATH "Directory where the datasets are stored")
message(STATUS "Build will verify or dowload ${DATASETS_COUNT_NETR} datasets from ${DATASETS_HOST_NETR} to ${DATASETS_DIR}")

foreach(DATASET ${DATASETS_EDGE_NETR})

	# Remove path from filename (e.g. misc/amazon-2008.zip -> amazon-2008.zip)
	string(REGEX REPLACE "^.*/" "" DATASET_PURE_FILENAME ${DATASET})
	# Remove .zip from filename (e.g. amazon-2008.zip -> amazon-2008)
	string(REGEX REPLACE "\\.zip$" "" DATASET_PURE_FILENAME ${DATASET_PURE_FILENAME})

    ExternalProject_Add(
        ${DATASET_PURE_FILENAME}
        PREFIX ${DATASETS_DIR}
        URL ${DATASETS_HOST_NETR}/${DATASET}
        DOWNLOAD_DIR ${DATASETS_DIR}/downloads
        DOWNLOAD_NO_EXTRACT TRUE
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
		INSTALL_COMMAND ${CMAKE_COMMAND} -E tar xf ${DATASETS_DIR}/downloads/${DATASET_PURE_FILENAME}.zip
		BINARY_DIR ${DATASETS_DIR}
        LOG_DOWNLOAD TRUE
    )

    list(APPEND DATASET_OUTPUTS ${DATASET_PURE_FILENAME})
endforeach()

add_custom_target(datasets DEPENDS ${DATASET_OUTPUTS})

file(GLOB DATASET_EXTRACTED_FILES "${DATASETS_DIR}/*")
foreach(EXTRACTED_FILE ${DATASET_EXTRACTED_FILES})
	foreach(DATASET_OUTPUT ${DATASET_OUTPUTS})
		string(FIND "${EXTRACTED_FILE}" "${DATASET_OUTPUT}" POSITION)
		if(NOT POSITION EQUAL -1)
			message(STATUS "Dataset found: ${EXTRACTED_FILE}")
			list(APPEND DATASET_FILES ${EXTRACTED_FILE})
		endif()
    endforeach()
endforeach()


#======================================================
# Nuclues Project

# Build options of the inner project, declared here so that they can be given
# on the super-build command line (e.g. cmake .. -DExecutionPolicy=par) and
# forwarded below. Defaults must match nucleus/CMakeLists.txt.
set(ExecutionPolicy "parunseq" CACHE STRING "STD Execution Policy.")
set_property(CACHE ExecutionPolicy PROPERTY STRINGS seq par parunseq unseq)
set(VertexType "unsigned" CACHE STRING "Vertex data type")
set(EdgeType "unsigned" CACHE STRING "Edge data type")
set(MultiArrangeType "mr_int" CACHE STRING "multi arrange data type")

set(_NUCLEUS_DIR "${CMAKE_CURRENT_BINARY_DIR}/nucleus" CACHE PATH "Nucleus project root directory")
set(_NUCLEUS_INSTALL_DIR "${_NUCLEUS_DIR}/install" CACHE PATH "Nucleus project install directory")

ExternalProject_Add(gpuNucleus34
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/nucleus"
	PREFIX "${_NUCLEUS_DIR}"
	INSTALL_DIR "${_NUCLEUS_INSTALL_DIR}"
    LIST_SEPARATOR |
    CMAKE_CACHE_ARGS
	"-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}"
	"-DCMAKE_INSTALL_PREFIX:STRING=${_NUCLEUS_INSTALL_DIR}" 
    "-DONETBB_PATH:STRING=${_ONETBB_INSTALL_DIR}"
    "-DCMAKE_CUDA_ARCHITECTURES:STRING=${CMAKE_CUDA_ARCHITECTURES}"
    "-DExecutionPolicy:STRING=${ExecutionPolicy}"
    "-DVertexType:STRING=${VertexType}"
    "-DEdgeType:STRING=${EdgeType}"
    "-DMultiArrangeType:STRING=${MultiArrangeType}"
    "-DDATASETS:STRING=${DATASET_FILES}"
	"-DDATASETS_DIR:PATH=${DATASETS_DIR}"
)

# When oneTBB is built from source it must be installed before the inner
# project configures, otherwise its find_package(TBB REQUIRED) fails.
if(NOT TBB_FOUND)
	ExternalProject_Add_StepDependencies(gpuNucleus34 configure intel_onetbb)
endif()


ExternalProject_Get_Property(gpuNucleus34 BINARY_DIR)

# Proxy targets: forward the CUDA executables from the inner project, e.g.
# `cmake --build build --target cuNucleus34_ondemand`.
foreach(_inner_target cuNucleus34 cuNucleus34_ondemand)
    add_custom_target(${_inner_target}
        COMMAND ${CMAKE_COMMAND} --build ${BINARY_DIR} --target ${_inner_target}
        COMMENT "Building inner-project target: ${_inner_target}"
        USES_TERMINAL)
    add_dependencies(${_inner_target} gpuNucleus34)
endforeach()
