#pragma once
#include "frame_simulator_base.h"
#include <vector>

class FrameSimulatorCPU : public FrameSimulatorBase {
private:
    // Each uint64_t word contains bits from different simulations
    // x_table[q][w] contains bit w of simulation s for qubit q
    // where s is the bit position within the word
    std::vector<std::vector<uint64_t>> x_table;  // [num_qubits][words_per_batch]
    std::vector<std::vector<uint64_t>> z_table;
    std::vector<std::vector<uint64_t>> measurements;
    
    size_t num_words;  // words needed for batch_size bits

public:
    FrameSimulatorCPU(size_t num_qubits, size_t batch_size);
    void do_ZCX(size_t control, size_t target) override;
    void do_H(size_t target) override;
    void do_M(size_t target) override;
    std::vector<bool> get_measurement_results(size_t target_idx) override;
}; 