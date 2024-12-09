#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>

enum class OpType {
    ZCX,
    H,
    M
};

class OperationBatch {
public:
    static constexpr size_t MAX_BATCH = 1024;

private:
    std::vector<OpType> op_types;
    std::vector<size_t> targets;
    std::vector<size_t> controls;
    size_t num_ops;

    size_t *d_targets;
    size_t *d_controls;
    OpType *d_op_types;

public:
    OperationBatch();
    ~OperationBatch();
    
    void add_single_qubit_op(size_t target, OpType type = OpType::M);
    void add_two_qubit_op(size_t control, size_t target, OpType type = OpType::ZCX);
    void upload_to_device();
    void clear();
    
    size_t* get_device_targets() const { return d_targets; }
    size_t* get_device_controls() const { return d_controls; }
    OpType* get_device_op_types() const { return d_op_types; }
    size_t size() const { return num_ops; }
}; 