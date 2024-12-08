#pragma once
#include "frame_simulator_base.h"
#include "operation_batch.cuh"
#include <cuda_runtime.h>

class FrameSimulatorGPU : public FrameSimulatorBase {
private:
    uint64_t *d_x_table;
    uint64_t *d_z_table;
    uint64_t *d_measurements;
    
    size_t num_words;
    OperationBatch op_batch;
    
    void allocateMemory();
    void freeMemory();
    void flush_zcx_batch();
    void flush_h_batch();
    void flush_m_batch();

public:
    FrameSimulatorGPU(size_t num_qubits, size_t batch_size);
    ~FrameSimulatorGPU();
    
    void do_ZCX(size_t control, size_t target) override;
    void do_H(size_t target) override;
    void do_M(size_t target) override;
    std::vector<bool> get_measurement_results(size_t target_idx) override;
}; 