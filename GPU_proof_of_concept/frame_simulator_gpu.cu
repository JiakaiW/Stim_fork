#include "frame_simulator_gpu.h"
#include "cuda_kernels.cuh"
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

struct QueuedOperation {
    OpType type;
    size_t control;
    size_t target;
};

FrameSimulatorGPU::FrameSimulatorGPU(size_t num_qubits, size_t batch_size) 
    : FrameSimulatorBase(num_qubits, batch_size) {
    num_words = words_needed(batch_size);
    if (g_debug_output) {
        std::cout << "Creating GPU simulator with " << num_qubits << " qubits and batch size " << batch_size << std::endl;
    }
    allocateMemory();
}

FrameSimulatorGPU::~FrameSimulatorGPU() {
    if (g_debug_output) {
        std::cout << "Destroying GPU simulator" << std::endl;
    }
    freeMemory();
}

void FrameSimulatorGPU::allocateMemory() {
    if (g_debug_output) {
        std::cout << "Allocating GPU memory..." << std::endl;
    }
    
    size_t table_size = num_qubits * num_words * sizeof(uint64_t);
    CHECK_CUDA(cudaMalloc(&d_x_table, table_size));
    CHECK_CUDA(cudaMalloc(&d_z_table, table_size));
    CHECK_CUDA(cudaMalloc(&d_measurements, table_size));
    
    // Initialize to zero
    CHECK_CUDA(cudaMemset(d_x_table, 0, table_size));
    CHECK_CUDA(cudaMemset(d_z_table, 0, table_size));
    CHECK_CUDA(cudaMemset(d_measurements, 0, table_size));
    
    if (g_debug_output) {
        std::cout << "GPU memory allocated and initialized" << std::endl;
    }
}

void FrameSimulatorGPU::freeMemory() {
    if (d_x_table) CHECK_CUDA_DESTRUCTOR(cudaFree(d_x_table));
    if (d_z_table) CHECK_CUDA_DESTRUCTOR(cudaFree(d_z_table));
    if (d_measurements) CHECK_CUDA_DESTRUCTOR(cudaFree(d_measurements));
    d_x_table = d_z_table = d_measurements = nullptr;
}

void FrameSimulatorGPU::do_ZCX(size_t control, size_t target) {
    queue_ZCX(control, target);
    execute_queued_operations();
}

void FrameSimulatorGPU::do_H(size_t target) {
    queue_H(target);
    execute_queued_operations();
}

void FrameSimulatorGPU::do_M(size_t target) {
    queue_M(target);
    execute_queued_operations();
}

void FrameSimulatorGPU::queue_ZCX(size_t control, size_t target) {
    QueuedOperation op = {OpType::ZCX, control, target};
    operation_queue.push(op);
}

void FrameSimulatorGPU::queue_H(size_t target) {
    QueuedOperation op = {OpType::H, 0, target};  // control unused for H
    operation_queue.push(op);
}

void FrameSimulatorGPU::queue_M(size_t target) {
    QueuedOperation op = {OpType::M, 0, target};  // control unused for M
    operation_queue.push(op);
}

void FrameSimulatorGPU::execute_queued_operations() {
    if (operation_queue.empty()) return;
    
    if (g_debug_output) {
        std::cout << "Executing " << operation_queue.size() << " queued operations" << std::endl;
    }
    
    // Transfer operations to batch
    op_batch.clear();
    while (!operation_queue.empty()) {
        QueuedOperation op = operation_queue.front();
        operation_queue.pop();
        
        if (op.type == OpType::ZCX) {
            op_batch.add_two_qubit_op(op.control, op.target, op.type);
        } else {
            op_batch.add_single_qubit_op(op.target, op.type);
        }
    }
    
    // Upload batch to device
    op_batch.upload_to_device();
    
    // Launch kernel
    const int BLOCK_SIZE = 256;
    int num_blocks = op_batch.size();
    
    if (g_debug_output) {
        std::cout << "Launching kernel with " << num_blocks << " blocks of " << BLOCK_SIZE << " threads" << std::endl;
    }
    
    batch_operations_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_x_table, d_z_table, d_measurements,
        op_batch.get_device_controls(),
        op_batch.get_device_targets(),
        op_batch.get_device_op_types(),
        op_batch.size(),
        num_words,
        g_debug_output
    );
    
    // Check for kernel errors
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

std::vector<bool> FrameSimulatorGPU::get_measurement_results(size_t target_idx) {
    std::vector<bool> results(batch_size);
    std::vector<uint64_t> host_measurements(num_words);
    
    // Copy measurements for target qubit to host
    CHECK_CUDA(cudaMemcpy(host_measurements.data(),
                         d_measurements + target_idx * num_words,
                         num_words * sizeof(uint64_t),
                         cudaMemcpyDeviceToHost));
    
    // Extract individual measurement results
    for (size_t w = 0; w < num_words; w++) {
        uint64_t word = host_measurements[w];
        for (size_t b = 0; b < WORD_BITS && (w * WORD_BITS + b) < batch_size; b++) {
            results[w * WORD_BITS + b] = (word >> b) & 1;
        }
    }
    
    return results;
} 