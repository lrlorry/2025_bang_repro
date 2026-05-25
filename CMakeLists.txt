cmake_minimum_required(VERSION 3.3)
project(bang_repro)

# 兼容 cmake 3.3（旧版 FindCUDA 模块，不依赖 CUDAToolkit / LANGUAGES CUDA）
find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${CMAKE_SOURCE_DIR} ${CUDA_INCLUDE_DIRS})

# 公共 NVCC 编译选项
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -O3 --expt-relaxed-constexpr -std=c++17)

# ── plain: reference 实现 ────────────────────────────────────────────────────
# Vamana 图构建（CPU 串行）+ GPU PQ 搜索（顺序执行，无流重叠，串行 worklist）
cuda_add_executable(bang_plain
  plain/plain_build.cu
  plain/plain_search.cu
  plain/plain_main.cu
)
target_link_libraries(bang_plain ${CUDA_LIBRARIES})

# ── engineered: 工程优化版 ───────────────────────────────────────────────────
# Build:  OpenMP parallel greedy_search + robust_prune
# Search: 转置 PQ table / 8-thread PQ dist / shared-mem merge / 4 streams / OpenMP
cuda_add_executable(bang_engineered
  plain/plain_search.cu
  engineered/engineered_build.cu
  engineered/engineered_search.cu
  engineered/engineered_main.cu
  OPTIONS -Xcompiler=-fopenmp
)
target_link_libraries(bang_engineered ${CUDA_LIBRARIES} -fopenmp)

# ── bang_bench: SIFT1M benchmark，sweeping kL at compile time ────────────────
# Each binary outputs one CSV line: L,recall,qps,search_ms
# Explicit targets (avoid foreach + cuda_add_executable interaction in cmake 3.3)
# bang_build: build graph once, save to disk (no CUDA search needed)
cuda_add_executable(bang_build
  bench/bang_build.cu
  engineered/engineered_build.cu
  OPTIONS -Xcompiler=-fopenmp
)
target_link_libraries(bang_build ${CUDA_LIBRARIES} -fopenmp)

set(BENCH_SRCS bench/bang_bench.cu engineered/engineered_build.cu engineered/engineered_search.cu)

cuda_add_executable(bang_bench_L16 ${BENCH_SRCS} OPTIONS -DkL_OVERRIDE=16 -Xcompiler=-fopenmp)
target_link_libraries(bang_bench_L16 ${CUDA_LIBRARIES} -fopenmp)

cuda_add_executable(bang_bench_L32 ${BENCH_SRCS} OPTIONS -DkL_OVERRIDE=32 -Xcompiler=-fopenmp)
target_link_libraries(bang_bench_L32 ${CUDA_LIBRARIES} -fopenmp)

cuda_add_executable(bang_bench_L48 ${BENCH_SRCS} OPTIONS -DkL_OVERRIDE=48 -Xcompiler=-fopenmp)
target_link_libraries(bang_bench_L48 ${CUDA_LIBRARIES} -fopenmp)

cuda_add_executable(bang_bench_L64 ${BENCH_SRCS} OPTIONS -DkL_OVERRIDE=64 -Xcompiler=-fopenmp)
target_link_libraries(bang_bench_L64 ${CUDA_LIBRARIES} -fopenmp)

cuda_add_executable(bang_bench_L128 ${BENCH_SRCS} OPTIONS -DkL_OVERRIDE=128 -Xcompiler=-fopenmp)
target_link_libraries(bang_bench_L128 ${CUDA_LIBRARIES} -fopenmp)

cuda_add_executable(bang_bench_L256 ${BENCH_SRCS} OPTIONS -DkL_OVERRIDE=256 -Xcompiler=-fopenmp)
target_link_libraries(bang_bench_L256 ${CUDA_LIBRARIES} -fopenmp)

# ── Utility tools (plain C++, no GPU) ───────────────────────────────────────
add_executable(fvecs_to_bin    tools/fvecs_to_bin.cpp)
add_executable(diskann_to_bang tools/diskann_to_bang.cpp)
target_compile_options(fvecs_to_bin    PRIVATE -std=c++17)
target_compile_options(diskann_to_bang PRIVATE -std=c++17)
