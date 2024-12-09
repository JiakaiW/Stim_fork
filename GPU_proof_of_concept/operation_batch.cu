#include "operation_batch.cuh"
#include "debug_flags.h"
#include <stdexcept>
#include <iostream>

// Helper function to check CUDA errors
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        throw std::runtime_error("CUDA error"); \
    } \
}

// Helper function to check CUDA errors without throwing
#define CHECK_CUDA_DESTRUCTOR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in destructor: " << cudaGetErrorString(err) << std::endl; \
    } \
}

OperationBatch::OperationBatch() : num_ops(0) {
    op_types.reserve(MAX_BATCH);
    targets.reserve(MAX_BATCH);
    controls.reserve(MAX_BATCH);
    
    if (g_debug_output) {
        std::cout << "Allocating operation batch buffers..." << std::endl;
    }
    CHECK_CUDA(cudaMalloc(&d_targets, MAX_BATCH * sizeof(size_t)));
    CHECK_CUDA(cudaMalloc(&d_controls, MAX_BATCH * sizeof(size_t)));
    CHECK_CUDA(cudaMalloc(&d_op_types, MAX_BATCH * sizeof(OpType)));
}

OperationBatch::~OperationBatch() {
    if (d_targets) CHECK_CUDA_DESTRUCTOR(cudaFree(d_targets));
    if (d_controls) CHECK_CUDA_DESTRUCTOR(cudaFree(d_controls));
    if (d_op_types) CHECK_CUDA_DESTRUCTOR(cudaFree(d_op_types));
    d_targets = d_controls = nullptr;
    d_op_types = nullptr;
}

void OperationBatch::add_single_qubit_op(size_t target, OpType type) {
    if (num_ops < MAX_BATCH) {
        op_types.push_back(type);
        targets.push_back(target);
        controls.push_back(0);  // Dummy value for alignment
        num_ops++;
    } else if (g_debug_output) {
        std::cerr << "Warning: Operation batch full, operation dropped" << std::endl;
    }
}

void OperationBatch::add_two_qubit_op(size_t control, size_t target, OpType type) {
    if (num_ops < MAX_BATCH) {
        op_types.push_back(type);
        controls.push_back(control);
        targets.push_back(target);
        num_ops++;
    } else if (g_debug_output) {
        std::cerr << "Warning: Operation batch full, operation dropped" << std::endl;
    }
}

void OperationBatch::upload_to_device() {
    if (num_ops > 0) {
        if (g_debug_output) {
            std::cout << "Uploading " << num_ops << " operations to device..." << std::endl;
        }
        CHECK_CUDA(cudaMemcpy(d_targets, targets.data(), num_ops * sizeof(size_t), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_controls, controls.data(), num_ops * sizeof(size_t), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_op_types, op_types.data(), num_ops * sizeof(OpType), cudaMemcpyHostToDevice));
    }
}

void OperationBatch::clear() {
    num_ops = 0;
    op_types.clear();
    targets.clear();
    controls.clear();
} 