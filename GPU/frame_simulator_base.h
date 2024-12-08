#pragma once
#include <vector>
#include <cstdint>

// Base class for both CPU and GPU implementations
class FrameSimulatorBase {
protected:
    size_t num_qubits;
    size_t batch_size;
    static constexpr size_t WORD_BITS = 64;  // bits per word

    // Helper to calculate number of words needed for batch_size bits
    static size_t words_needed(size_t num_bits) {
        return (num_bits + WORD_BITS - 1) / WORD_BITS;
    }

public:
    FrameSimulatorBase(size_t num_qubits, size_t batch_size) 
        : num_qubits(num_qubits), batch_size(batch_size) {}
    
    virtual ~FrameSimulatorBase() = default;

    // Core operations that both CPU and GPU must implement
    virtual void do_ZCX(size_t control, size_t target) = 0;
    virtual void do_H(size_t target) = 0;
    virtual void do_M(size_t target) = 0;
    
    // Helper to get measurements
    virtual std::vector<bool> get_measurement_results(size_t target_idx) = 0;
}; 