# Nucleus Decomposition with Clique Reduction Technique

A high-performance implementation of nucleus decomposition using a clique reduction technique for GPU execution. Targets include well-known graph decomposition Nucleus 3-4.

## Table of Contents

- [Usage](#usage)
- [Build Requirements](#build-requirements)
- [Build Instructions](#build-instructions)
- [CMake Switches](#cmake-switches)
- [Intel TBB / OneAPI](#intel-tbb-oneapi)
- [GUI](#gui)
- [Testing](#testing)

---

## Usage

### GPU Targets

| Target | Command |
|---|---|
| Nucleus 3-4 | `./cuNucleus34 dataset.mtx [output-file]` |

**Nucleus 3-4 GPU arguments:**

- `output-file` _(optional)_ — write each triangle and its support vector to a file. Triangles are represented as edges from the edge-list source.

---

## Build Requirements

| Tool | Minimum version |
|---|---|
| gcc | 12.3 |
| cmake | 3.31.0 |
| cuda | 12.3 |

---

## Build Instructions

```bash
mkdir build && cd build
cmake ..           # configure
cmake --build .    # build all targets
```

This also builds dependency submodule Intel OneAPI and downloads any large datasets.

To build a single target:

```bash
cmake --build . --target gpuNucleus34 --clean-first
```

---

## CMake Switches

The following options can be passed to `cmake ..` as `-D<OPTION>=<value>`. The first value listed is the default.

| Option | Values |
|---|---|
| `ExecutionPolicy` | `seq` / `par` / `parunseq` / `unseq` |
| `VertexType` | `unsigned` / `unsigned_long` |
| `EdgeType` | `unsigned` / `unsigned_long` |
| `LocalTBB` | `OFF` / `ON` |
| `LocalTBBPath` | `"/usr/local/"` |

Example:

```bash
cmake .. -DExecutionPolicy=par
```

---

## Intel TBB / OneAPI

Intel TBB is used to provide the threading backend required by C++17 parallel execution policies (`std::execution::par`, etc.).
