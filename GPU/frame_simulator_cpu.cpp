#include "frame_simulator_cpu.h"
#include <cmath>

FrameSimulatorCPU::FrameSimulatorCPU(size_t num_qubits, size_t batch_size) 
    : FrameSimulatorBase(num_qubits, batch_size) {
    num_words = words_needed(batch_size);
    
    // Initialize tables
    x_table.resize(num_qubits, std::vector<uint64_t>(num_words, 0));
    z_table.resize(num_qubits, std::vector<uint64_t>(num_words, 0));
    measurements.resize(num_qubits, std::vector<uint64_t>(num_words, 0));
}

void FrameSimulatorCPU::do_ZCX(size_t control, size_t target) {
    // Process 64 simulations at once using bitwise operations
    for (size_t w = 0; w < num_words; w++) {
        z_table[control][w] ^= z_table[target][w];
        x_table[target][w] ^= x_table[control][w];
    }
}

void FrameSimulatorCPU::do_H(size_t target) {
    for (size_t w = 0; w < num_words; w++) {
        uint64_t temp = x_table[target][w];
        x_table[target][w] = z_table[target][w];
        z_table[target][w] = temp;
    }
}

void FrameSimulatorCPU::do_M(size_t target) {
    for (size_t w = 0; w < num_words; w++) {
        measurements[target][w] = x_table[target][w];
    }
}

std::vector<bool> FrameSimulatorCPU::get_measurement_results(size_t target_idx) {
    std::vector<bool> results(batch_size);
    
    for (size_t w = 0; w < num_words; w++) {
        uint64_t word = measurements[target_idx][w];
        for (size_t b = 0; b < WORD_BITS && (w * WORD_BITS + b) < batch_size; b++) {
            results[w * WORD_BITS + b] = (word >> b) & 1;
        }
    }
    return results;
} 