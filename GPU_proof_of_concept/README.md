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

2. Build Stim first:
```bash
mkdir build && cd build
cmake ..
make
cd ..
```

3. Create and enter the build directory:
```bash
cd GPU_proof_of_concept
mkdir build
cd build
```

4. Configure with CMake:
```bash
cmake ..
```

5. Build the project:
```bash
make
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

## Prerequisites

- CUDA Toolkit (tested with CUDA 11.5)
- C++17 compiler
- NVIDIA GPU with compute capability 7.5 or higher
- CMake (for building Stim)

## Building Stim Library
First, build Stim as a library:
```bash
# (navigate to the Stim directory)
cd ..

# Create build directory and build Stim without Python bindings
mkdir build && cd build
cmake -DBUILD_PYTHON_BINDINGS=OFF ..
make # This will take a while (10-20 minutes)

# Return to GPU_proof_of_concept directory
cd ../../GPU_proof_of_concept
```

## Compilation

1. **QEC Code Benchmark with Stim**
```bash
# Compile the QEC benchmark (assuming Stim is in ../Stim)
nvcc qec_benchmark.cpp frame_simulator_cpu.cpp frame_simulator_gpu.cu cuda_kernels.cu operation_batch.cu \
    -I../Stim/src \
    -L../Stim/build \
    -lstim \
    -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 \
    -o qec_benchmark
```

### Compilation Flags Explained
- `-I../Stim/src`: Include path for Stim headers
- `-L../Stim/build`: Path to built Stim library
- `-lstim`: Link against Stim library
- `-Xcompiler -O3`: Enable high optimization level for host code
- `-Xcompiler -Wall`: Enable all warnings for host code
- `-Xptxas -O3`: Enable high optimization level for device code
- `-std=c++17`: Use C++17 standard

## Running the Benchmarks

### QEC Code Benchmark
```bash
./qec_benchmark
```
This simulates repetition code cycles, comparing CPU vs GPU vs Original Stim performance for:
- Different code distances (25, 50, 75)
- Fixed batch size (100,000 simulations)
- Each test runs 1000 QEC cycles

Example output from NVIDIA RTX 3090:
```
Testing repetition code with distance 25 and batch size 100000
----------------------------------------
Original Frame Simulator time: 91.46ms
CPU time: 91.46ms
GPU time: 8.62ms
GPU Speedup vs CPU: 10.61x
GPU Speedup vs Original: 10.61x

Testing repetition code with distance 50 and batch size 100000
----------------------------------------
Original Frame Simulator time: 254.69ms
CPU time: 254.69ms
GPU time: 0.76ms
GPU Speedup vs CPU: 334.92x
GPU Speedup vs Original: 334.92x

Testing repetition code with distance 75 and batch size 100000
----------------------------------------
Original Frame Simulator time: 363.38ms
CPU time: 363.38ms
GPU time: 0.59ms
GPU Speedup vs CPU: 620.35x
GPU Speedup vs Original: 620.35x
```

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