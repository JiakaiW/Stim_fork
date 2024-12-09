#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "operation_batch.cuh"

// Declare CUDA kernels
__global__ void zcx_kernel(uint64_t *x_table, uint64_t *z_table, 
                          const size_t *controls, const size_t *targets,
                          size_t num_ops, size_t num_words);

__global__ void hadamard_kernel(uint64_t *x_table, uint64_t *z_table,
                               const size_t *targets, size_t num_ops, 
                               size_t num_words);

__global__ void measure_kernel(uint64_t *x_table, uint64_t *measurements,
                             const size_t *targets, size_t num_ops,
                             size_t num_words);

__global__ void batch_operations_kernel(uint64_t *x_table, uint64_t *z_table, uint64_t *measurements,
                                      const size_t *controls, const size_t *targets,
                                      const OpType *op_types, size_t num_ops, size_t num_words); 