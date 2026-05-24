cmake_minimum_required(VERSION 3.18)
project(bang_repro LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_SOURCE_DIR})

# plain: reference BANG reproduction (host graph + GPU PQ, CPU-GPU handoff)
add_executable(bang_plain
  plain/plain_search.cu
  plain/plain_main.cu
)
target_compile_options(bang_plain PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr>
)
target_link_libraries(bang_plain PRIVATE CUDA::cudart)
