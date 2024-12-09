# GPU QEC Benchmark

This is a proof-of-concept implementation of quantum error correction (QEC) simulation on GPU.

## Overview

The project implements frame-based simulation of quantum error correction codes, with both CPU and GPU implementations for performance comparison. The main focus is on repetition codes and surface codes.

## Building

### Prerequisites

- CMake 3.8 or higher
- CUDA Toolkit 11.0 or higher
- C++20 compatible compiler for host code
- C++17 compatible NVCC for device code

### Build Steps

1. Clone the repository with Stim submodule:
```bash
git clone --recursive <repository-url>
```

2. Build Stim first: (takes 20 minutes)
```bash
mkdir build && cd build
cmake ..
make
cd ..
```

3. Build the project: (takes about 1 minute)
```bash
cd GPU_proof_of_concept
rm -rf build
mkdir build
cd build
cmake ..
make -j
```

This will build:
- `qec_benchmark`: The main benchmark executable
- `qec_test`: The test executable

### Build Structure

The project is organized into three main components:

1. `stim_core`: A static library containing the core Stim functionality needed for QEC simulation
2. `cuda_impl`: A static library containing the CUDA implementation of frame simulation
3. Executables:
   - `qec_benchmark`: For performance benchmarking
   - `qec_test`: For running tests

The build system is configured to use:
- C++20 for host code (Stim and main executables)
- C++17 for device code (CUDA implementation)

### Common Build Issues

1. CUDA Architecture: By default, the build targets compute capability 7.5. If your GPU has a different architecture, modify `CMAKE_CUDA_ARCHITECTURES` in CMakeLists.txt.

2. Compiler Compatibility: Ensure your host compiler is compatible with C++20 and NVCC supports C++17.

3. Include Paths: If you encounter include path issues, verify that:
   - Stim source files are present in the expected location
   - The include paths in CMakeLists.txt match your directory structure

## Usage

This is a standalone proof-of-concept implementation of a GPU-accelerated frame simulator. It demonstrates the potential speedup of using GPU parallelization for quantum circuit simulation.

## Running the Benchmarks

### QEC Code Benchmark
```bash
./qec_benchmark #Normal mode (minimal output):
./qec_benchmark --debug #Detailed debug output
```
Testing repetition code with distance 25 and batch size 100000
----------------------------------------
Original Frame Simulator time: 1774.72ms
CPU time: 1359.41ms
GPU time: 2.03919ms
GPU Speedup vs CPU: 666.645x
GPU Speedup vs Original: 870.306x

Testing repetition code with distance 50 and batch size 100000
----------------------------------------
Original Frame Simulator time: 3929.9ms
CPU time: 2788.49ms
GPU time: 1.39765ms
GPU Speedup vs CPU: 1995.13x
GPU Speedup vs Original: 2811.79x

Testing repetition code with distance 75 and batch size 100000
----------------------------------------
Original Frame Simulator time: 5549.1ms
CPU time: 4234.63ms
GPU time: 0.391556ms
GPU Speedup vs CPU: 10814.9x
GPU Speedup vs Original: 14171.9x

## Implementation Details

The implementation uses several parallelization strategies:

1. **Bit-level Parallelism**: Each 64-bit word stores bits from 64 different simulations
2. **Thread-level Parallelism**: Each CUDA thread processes one word
3. **Operation Batching**: Multiple operations are batched together for efficient GPU execution

### Key Files

- `frame_simulator_base.h`: Base class defining the simulator interface
- `frame_simulator_cpu.h/cpp`: CPU implementation using bit-level parallelism
- `frame_simulator_gpu.h/cu`: GPU implementation using CUDA
- `cuda_kernels.cuh/cu`: CUDA kernel implementations
- `operation_batch.cuh/cu`: Helper class for batching operations
- `testing.cpp`: Basic benchmarking program
- `qec_benchmark.cpp`: Quantum error correction benchmarking program

## Performance Notes

- GPU performance advantage increases dramatically with code distance:
  1. 10x speedup at distance 25
  2. 335x speedup at distance 50
  3. 620x speedup at distance 75
- The dramatic scaling suggests the GPU implementation is particularly efficient for larger codes
- Operation batching is crucial for achieving these speedups by minimizing host-device communication
- The CPU implementation's performance scales linearly with code distance, while GPU shows better scaling

## Troubleshooting

1. **CUDA Architecture Issues**
   - If you get compilation errors about unsupported architecture, modify the `-arch` flag:
   ```bash
   nvcc ... -arch=sm_XX # where XX is your GPU's compute capability
   ```

2. **Memory Allocation Failures**
   - For large problem sizes, you may need to reduce the batch size or number of qubits
   - Check your GPU's available memory with `nvidia-smi`

3. **Performance Issues**
   - Ensure you're running in Release mode (with optimization flags)
   - Check if your GPU is being used for display, which can impact performance
   - Monitor GPU utilization with `nvidia-smi -l 1`