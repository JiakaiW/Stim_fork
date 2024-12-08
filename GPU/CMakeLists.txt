cmake_minimum_required(VERSION 3.8)
project(frame_simulator_gpu CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

add_executable(benchmark
    main.cpp
    frame_simulator_cpu.cpp
    cuda_kernels.cu
    frame_simulator_gpu.cu
)

set_target_properties(benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
) 