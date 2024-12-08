cmake_minimum_required(VERSION 3.8)
project(frame_simulator_gpu CUDA CXX)

# Set C++14 for both CUDA and C++
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

# Set CUDA architecture (adjust for your GPU)
set(CMAKE_CUDA_ARCHITECTURES 75)  # For RTX 20xx series

# Common source files
set(COMMON_SOURCES
    frame_simulator_cpu.cpp
    frame_simulator_gpu.cu
)

# Basic benchmark executable
add_executable(benchmark
    main.cpp
    ${COMMON_SOURCES}
)

# Common properties
set_target_properties(benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
) 