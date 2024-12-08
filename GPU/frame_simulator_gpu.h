#pragma once
#include "frame_simulator_base.h"
#include <cuda_runtime.h>

class FrameSimulatorGPU : public FrameSimulatorBase {
private:
    // Device memory pointers to packed bit arrays
    uint64_t *d_x_table;    // [num_qubits][words_per_batch]
    uint64_t *d_z_table;
    uint64_t *d_measurements;
    
    size_t num_words;  // words needed for batch_size bits
    
    void allocateMemory();
    void freeMemory();

public:
    FrameSimulatorGPU(size_t num_qubits, size_t batch_size);
    ~FrameSimulatorGPU();
    
    void do_ZCX(size_t control, size_t target) override;
    void do_H(size_t target) override;
    void do_M(size_t target) override;
    std::vector<bool> get_measurement_results(size_t target_idx) override;
}; 