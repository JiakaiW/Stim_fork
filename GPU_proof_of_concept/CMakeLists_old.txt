cmake_minimum_required(VERSION 3.8)
project(gpu_qec_benchmark CUDA CXX)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 75)

# Enable CUDA
enable_language(CUDA)

# Create minimal Stim library
set(STIM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/stim/circuit/circuit.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/stim/circuit/circuit_instruction.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/stim/circuit/gate_target.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/stim/circuit/gate_decomposition.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/stim/gen/gen_rep_code.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/stim/gen/circuit_gen_params.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/stim/gates/gates.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/stim/gates/gate_data_collapsing.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/stim/gates/gate_data_period_4.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/stim/gates/gate_data_heralded.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/stim/gates/gate_data_pauli_product.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/stim/gates/gate_data_annotations.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/stim/gates/gate_data_hada.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/stim/gates/gate_data_pp.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/stim/gates/gate_data_pauli.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/stim/gates/gate_data_noisy.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/stim/gates/gate_data_swaps.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/stim/gates/gate_data_blocks.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/stim/gates/gate_data_period_3.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/stim/gates/gate_data_controlled.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/stim/gates/gate_data_pair_measure.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/stim/simulators/vector_simulator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/stim/simulators/error_analyzer.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/stim/simulators/error_matcher.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/stim/simulators/sparse_rev_frame_tracker.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/stim/mem/simd_util.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/stim/mem/simd_word.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/stim/mem/bit_ref.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/stim/mem/sparse_xor_vec.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/stim/stabilizers/flex_pauli_string.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/stim/util_bot/probability_util.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/stim/util_bot/error_decomp.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/stim/util_bot/arg_parse.cc
)

add_library(stim_core STATIC ${STIM_SOURCES})
set_target_properties(stim_core PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)
target_include_directories(stim_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

# Create CUDA library with C++17
add_library(cuda_impl STATIC
    frame_simulator_gpu.cu
    operation_batch.cu
    cuda_kernels.cu
)
set_target_properties(cuda_impl PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

# Create benchmark executable with C++20
add_executable(qec_benchmark
    qec_benchmark.cpp
    frame_simulator_cpu.cpp
)
set_target_properties(qec_benchmark PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Create test executable with C++20
add_executable(qec_test
    testing.cpp
    frame_simulator_cpu.cpp
)
set_target_properties(qec_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Link everything together
target_link_libraries(qec_benchmark PRIVATE cuda_impl stim_core ${CUDA_LIBRARIES})
target_link_libraries(qec_test PRIVATE cuda_impl stim_core ${CUDA_LIBRARIES})

# Include directories
target_include_directories(cuda_impl PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(qec_benchmark PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

target_include_directories(qec_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
) 