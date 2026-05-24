cmake_minimum_required(VERSION 3.18)
project(bang_repro LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${CMAKE_SOURCE_DIR})

# ── plain: reference 实现 ────────────────────────────────────────────────────
# host graph + GPU PQ，CPU-GPU 顺序执行，无流重叠，串行 worklist
add_executable(bang_plain
  plain/plain_search.cu
  plain/plain_main.cu
)
target_compile_options(bang_plain PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr>
)
target_link_libraries(bang_plain PRIVATE CUDA::cudart)

# ── engineered: 工程优化版 ───────────────────────────────────────────────────
# 1. 转置 PQ table（改善 centroid access coalescing）
# 2. 8-thread PQ distance + warp reduce（kTPNbr=8）
# 3. Shared-memory parallel worklist merge（2*kL 线程）
# 4. 4 CUDA streams（parent D2H / children H2D / FP H2D / kernels 重叠）
# 5. OpenMP CPU graph fetch
add_executable(bang_engineered
  plain/plain_search.cu       # 提供 HostGraph/DevicePQ 定义（plain 的代码没有独立 .o）
  engineered/engineered_search.cu
  engineered/engineered_main.cu
)
target_compile_options(bang_engineered PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr -Xcompiler -fopenmp>
  $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>
)
target_link_libraries(bang_engineered PRIVATE CUDA::cudart OpenMP::OpenMP_CXX)
