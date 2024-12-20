#pragma once
#include "frame_simulator_base.h"
#include "operation_batch.cuh"
#include <cuda_runtime.h>
#include <queue>

struct QueuedOperation;  // Forward declaration

class FrameSimulatorGPU : public FrameSimulatorBase {
private:
    // Device memory pointers to packed bit arrays
    uint64_t *d_x_table;    // [num_qubits][words_per_batch]
    uint64_t *d_z_table;
    uint64_t *d_measurements;
    
    size_t num_words;  // words needed for batch_size bits
    OperationBatch op_batch;
    std::queue<QueuedOperation> operation_queue;
    
    void allocateMemory();
    void freeMemory();

public:
    FrameSimulatorGPU(size_t num_qubits, size_t batch_size);
    ~FrameSimulatorGPU();
    
    void do_ZCX(size_t control, size_t target) override;
    void do_H(size_t target) override;
    void do_M(size_t target) override;
    std::vector<bool> get_measurement_results(size_t target_idx) override;
    void queue_ZCX(size_t control, size_t target);
    void queue_M(size_t target);
    void execute_queued_operations();
    void queue_H(size_t target);
}; 