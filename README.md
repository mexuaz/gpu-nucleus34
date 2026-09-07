# Nucleus Decomposition with Clique Reduction Technique

GPU (3,4)-nucleus decomposition: triangles are peeled by their four-clique
support. Two executables are built from this tree.

| Target | Approach |
|---|---|
| `cuNucleus34` | Materialises the four-clique list and the triangle -> four-clique incidence on the device, then peels against it. Use when the clique list fits in GPU memory. The binary identifies itself as `cuda-nucleus34-direct`. |
| `cuNucleus34_ondemand` | Never materialises the four-cliques. It keeps an edge -> triangle incidence and re-enumerates a triangle's cliques whenever the peel needs them, so memory is O(triangles) instead of O(four-cliques). Use when the clique list does not fit. |

Both read the same input formats and write a JSON result record to stdout.

## Table of Contents

- [Build Requirements](#build-requirements)
- [Build Instructions](#build-instructions)
- [CMake Options](#cmake-options)
- [Usage](#usage)

---

## Build Requirements

| Tool | Minimum version |
|---|---|
| gcc | 12.3 (AppleClang 17) |
| cmake | 3.25 |
| cuda | 11.1.1 |

The version checks are enforced by CMake and the build fails if they are not
met. Device code is compiled for **sm_90 (Hopper)** by default; see
`CMAKE_CUDA_ARCHITECTURES` below for other GPUs.

Configuring also needs `git` and network access: the build clones and builds
Intel oneTBB unless an existing install is pointed at, and the default build
downloads datasets.

---

## Build Instructions

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

`CMAKE_BUILD_TYPE` defaults to `Debug` (`-O0`), so pass `Release` unless you
are debugging.

The top-level project is a super-build. A full build does three things:

1. Finds Intel oneTBB, or clones and builds `v2021.11.0` into
   `build/intel_onetbb`. oneTBB is the threading backend libstdc++ requires for
   the C++17 parallel execution policies (`std::execution::par`) used by the
   host-side code.
2. Downloads the graphs listed in `DATASETS_EDGE_NETR` (`CMakeLists.txt`) from
   networkrepository.com into `build/datasets`.
3. Configures and builds `nucleus/` as an external project, and installs the
   executables into `build/nucleus/install/`:

```
build/nucleus/install/cuNucleus34
build/nucleus/install/cuNucleus34_ondemand
```

To build one executable — and skip the dataset downloads, which are separate
targets — name it directly:

```bash
cmake --build build --target cuNucleus34_ondemand
```

To use an oneTBB that is already installed instead of building one:

```bash
cmake -S . -B build -D_ONETBB_INSTALL_DIR=/path/to/onetbb/install
```

---

## CMake Options

Pass these to the super-build as `cmake -S . -B build -D<OPTION>=<value>`; they
are forwarded to the inner project. Defaults are listed first.

| Option | Values |
|---|---|
| `CMAKE_BUILD_TYPE` | `Debug` / `Release` / `RelWithDebInfo` / `MinSizeRel` |
| `CMAKE_CUDA_ARCHITECTURES` | `90`, or a list for a fat binary, e.g. `"80;90"` |
| `ExecutionPolicy` | `parunseq` / `seq` / `par` / `unseq` |
| `VertexType` | `unsigned` / `unsigned_long` |
| `EdgeType` | `unsigned` / `unsigned_long` |
| `MultiArrangeType` | `mr_int` / `mr_long` / `mr_long_long` |
| `_ONETBB_INSTALL_DIR` | `build/intel_onetbb/install` |
| `DATASETS_DIR` | `build/datasets` |

The CUDA targets accept only `unsigned` vertex and edge types and `mr_int`
multi-arrange type; the other values are rejected at configure time.

Example:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
```

---

## Usage

```bash
./cuNucleus34 <dataset file> [parts] [output file]
./cuNucleus34_ondemand <dataset file> [output file]
```

- `dataset file` — `.mtx` (Matrix Market, header line expected), `.edges` or
  `.txt` (bare edge list, `#` and `%` comment lines), or `.grh` (serialised
  graph).
- `parts` _(`cuNucleus34` only, optional, default 1)_ — a value greater than 1
  routes triangle and four-clique counting through the parted pipeline.
- `output file` _(optional)_ — one line per triangle with its final `k`.
  Triangles are written as edges from the edge-list source.

Progress is reported on stderr; the result record is JSON on stdout.
