#include "cuda_kernels.cuh"
#include <stdio.h>

__global__ void zcx_kernel(uint64_t *x_table, uint64_t *z_table, 
                          const size_t *controls, const size_t *targets,
                          size_t num_ops, size_t num_words) {
    size_t op_idx = blockIdx.x;
    
    // Grid stride loop over words
    for (size_t word_idx = threadIdx.x; word_idx < num_words; word_idx += blockDim.x) {
        if (op_idx < num_ops) {
            size_t control = controls[op_idx];
            size_t target = targets[op_idx];
            
            z_table[control * num_words + word_idx] ^= z_table[target * num_words + word_idx];
            x_table[target * num_words + word_idx] ^= x_table[control * num_words + word_idx];
        }
    }
}

__global__ void hadamard_kernel(uint64_t *x_table, uint64_t *z_table,
                               const size_t *targets, size_t num_ops, 
                               size_t num_words) {
    size_t op_idx = blockIdx.x;
    
    // Grid stride loop over words
    for (size_t word_idx = threadIdx.x; word_idx < num_words; word_idx += blockDim.x) {
        if (op_idx < num_ops) {
            size_t target = targets[op_idx];
            size_t idx = target * num_words + word_idx;
            
            uint64_t temp = x_table[idx];
            x_table[idx] = z_table[idx];
            z_table[idx] = temp;
        }
    }
}

__global__ void measure_kernel(uint64_t *x_table, uint64_t *measurements,
                             const size_t *targets, size_t num_ops,
                             size_t num_words) {
    size_t op_idx = blockIdx.x;
    
    // Grid stride loop over words
    for (size_t word_idx = threadIdx.x; word_idx < num_words; word_idx += blockDim.x) {
        if (op_idx < num_ops) {
            size_t target = targets[op_idx];
            size_t idx = target * num_words + word_idx;
            
            measurements[idx] = x_table[idx];
            x_table[idx] = 0;
        }
    }
}

__global__ void batch_operations_kernel(uint64_t *x_table, uint64_t *z_table, uint64_t *measurements,
                                      const size_t *controls, const size_t *targets,
                                      const OpType *op_types, size_t num_ops, size_t num_words,
                                      bool debug_output) {
    size_t op_idx = blockIdx.x;
    
    // Grid stride loop over words
    for (size_t word_idx = threadIdx.x; word_idx < num_words; word_idx += blockDim.x) {
        if (op_idx < num_ops) {
            OpType op_type = op_types[op_idx];
            size_t target = targets[op_idx];
            
            if (debug_output && threadIdx.x == 0) {
                printf("Processing operation %lu: type=%d target=%lu\n", 
                       op_idx, (int)op_type, target);
            }
            
            switch(op_type) {
                case OpType::ZCX: {
                    size_t control = controls[op_idx];
                    z_table[control * num_words + word_idx] ^= z_table[target * num_words + word_idx];
                    x_table[target * num_words + word_idx] ^= x_table[control * num_words + word_idx];
                    break;
                }
                case OpType::H: {
                    size_t idx = target * num_words + word_idx;
                    uint64_t temp = x_table[idx];
                    x_table[idx] = z_table[idx];
                    z_table[idx] = temp;
                    break;
                }
                case OpType::M: {
                    size_t idx = target * num_words + word_idx;
                    measurements[idx] = x_table[idx];
                    x_table[idx] = 0;
                    break;
                }
            }
        }
    }
} 