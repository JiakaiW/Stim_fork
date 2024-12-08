#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>

class OperationBatch {
private:
    static constexpr size_t MAX_BATCH = 1024;
    std::vector<size_t> targets;
    std::vector<size_t> controls;
    size_t num_ops;

    size_t *d_targets;
    size_t *d_controls;

public:
    OperationBatch();
    ~OperationBatch();
    
    void add_single_qubit_op(size_t target);
    void add_two_qubit_op(size_t control, size_t target);
    void upload_to_device();
    void clear();
    
    size_t* get_device_targets() const { return d_targets; }
    size_t* get_device_controls() const { return d_controls; }
    size_t size() const { return num_ops; }
}; 