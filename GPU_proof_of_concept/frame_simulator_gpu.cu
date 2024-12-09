#include "frame_simulator_gpu.h"
#include "cuda_kernels.cuh"
#include <cmath>

FrameSimulatorGPU::FrameSimulatorGPU(size_t num_qubits, size_t batch_size) 
    : FrameSimulatorBase(num_qubits, batch_size) {
    num_words = words_needed(batch_size);
    allocateMemory();
}

FrameSimulatorGPU::~FrameSimulatorGPU() {
    freeMemory();
}

void FrameSimulatorGPU::allocateMemory() {
    size_t size = num_qubits * num_words * sizeof(uint64_t);
    cudaMalloc(&d_x_table, size);
    cudaMalloc(&d_z_table, size);
    cudaMalloc(&d_measurements, size);
    
    // Initialize to false
    cudaMemset(d_x_table, 0, size);
    cudaMemset(d_z_table, 0, size);
    cudaMemset(d_measurements, 0, size);
}

void FrameSimulatorGPU::freeMemory() {
    cudaFree(d_x_table);
    cudaFree(d_z_table);
    cudaFree(d_measurements);
}

void FrameSimulatorGPU::queue_ZCX(size_t control, size_t target) {
    op_batch.add_two_qubit_op(control, target);
}

void FrameSimulatorGPU::queue_M(size_t target) {
    op_batch.add_single_qubit_op(target);
}

void FrameSimulatorGPU::queue_H(size_t target) {
    op_batch.add_single_qubit_op(target, OpType::H);
}

void FrameSimulatorGPU::execute_queued_operations() {
    if (op_batch.size() > 0) {
        op_batch.upload_to_device();
        
        dim3 grid(op_batch.size());
        dim3 block(num_words);
        
        batch_operations_kernel<<<grid, block>>>(
            d_x_table, d_z_table, d_measurements,
            op_batch.get_device_controls(),
            op_batch.get_device_targets(),
            op_batch.get_device_op_types(),
            op_batch.size(), num_words);
        
        op_batch.clear();
    }
}

void FrameSimulatorGPU::do_ZCX(size_t control, size_t target) {
    queue_ZCX(control, target);
    execute_queued_operations();  // For backward compatibility, execute immediately
}

void FrameSimulatorGPU::do_M(size_t target) {
    queue_M(target);
    execute_queued_operations();  // For backward compatibility, execute immediately
}

void FrameSimulatorGPU::do_H(size_t target) {
    queue_H(target);
    execute_queued_operations();  // For backward compatibility, execute immediately
}

std::vector<bool> FrameSimulatorGPU::get_measurement_results(size_t target_idx) {
    std::vector<bool> results(batch_size);
    std::vector<uint64_t> host_measurements(num_words);
    
    // Copy measurements for target qubit to host
    cudaMemcpy(host_measurements.data(), 
               d_measurements + target_idx * num_words,
               num_words * sizeof(uint64_t),
               cudaMemcpyDeviceToHost);
    
    // Extract individual bits
    for (size_t w = 0; w < num_words; w++) {
        uint64_t word = host_measurements[w];
        for (size_t b = 0; b < WORD_BITS && (w * WORD_BITS + b) < batch_size; b++) {
            results[w * WORD_BITS + b] = (word >> b) & 1;
        }
    }
    
    return results;
} 