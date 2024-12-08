# GPU Frame Simulator Demo

This is a standalone proof-of-concept implementation of a GPU-accelerated frame simulator. It demonstrates the potential speedup of using GPU parallelization for quantum circuit simulation.

## Prerequisites

- CUDA Toolkit (tested with CUDA 11.5)
- C++17 compiler
- NVIDIA GPU with compute capability 7.5 or higher

## Compilation

1. **Basic Operations Benchmark**
```bash
# Compile the basic benchmark
nvcc main.cpp frame_simulator_cpu.cpp frame_simulator_gpu.cu cuda_kernels.cu operation_batch.cu \
    -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o benchmark
```

2. **QEC Code Benchmark**
```bash
# Compile the QEC benchmark
nvcc qec_benchmark.cpp frame_simulator_cpu.cpp frame_simulator_gpu.cu cuda_kernels.cu operation_batch.cu \
    -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o qec_benchmark
```

### Compilation Flags Explained
- `-Xcompiler -O3`: Enable high optimization level for host code
- `-Xcompiler -Wall`: Enable all warnings for host code
- `-Xptxas -O3`: Enable high optimization level for device code
- `-std=c++17`: Use C++17 standard

## Running the Benchmarks

### 1. Basic Operations Benchmark
```bash
./benchmark
```
This runs basic benchmarks comparing CPU vs GPU performance for:
- ZCX (controlled-X) gates
- Hadamard gates
- Different problem sizes:
  - Small: 100 qubits, 1000 shots
  - Medium: 1000 qubits, 10000 shots
  - Large: 10000 qubits, 100000 shots

Example output:
```
Testing with 100 qubits and batch size 1000
----------------------------------------
Benchmarking ZCX operations...
CPU time: 0.31ms
GPU time: 4.48ms
Speedup: 0.07x

Benchmarking H operations...
CPU time: 0.15ms
GPU time: 0.75ms
Speedup: 0.20x

[... more results for larger sizes ...]
```

### 2. QEC Code Benchmark
```bash
./qec_benchmark
```
This simulates repetition code cycles, comparing CPU vs GPU performance for:
- Different code distances (3, 7, 15)
- Different batch sizes (1000-100000 simulations)
- Each test runs 1000 QEC cycles

Example output:
```
Testing repetition code with distance 3 and batch size 1000
----------------------------------------
CPU time: XXXms
GPU time: XXXms
Speedup: XXx

[... more results for larger sizes ...]
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
- `main.cpp`: Basic benchmarking program
- `qec_benchmark.cpp`: Quantum error correction benchmarking program

## Performance Notes

- GPU performance advantage increases with:
  1. Larger number of qubits
  2. Larger batch sizes
  3. More operations between host-device synchronization
- The CPU implementation is still competitive for small problem sizes due to lower overhead

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