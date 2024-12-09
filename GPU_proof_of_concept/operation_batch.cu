#include "operation_batch.cuh"

OperationBatch::OperationBatch() : num_ops(0) {
    op_types.reserve(MAX_BATCH);
    targets.reserve(MAX_BATCH);
    controls.reserve(MAX_BATCH);
    
    cudaMalloc(&d_targets, MAX_BATCH * sizeof(size_t));
    cudaMalloc(&d_controls, MAX_BATCH * sizeof(size_t));
    cudaMalloc(&d_op_types, MAX_BATCH * sizeof(OpType));
}

OperationBatch::~OperationBatch() {
    cudaFree(d_targets);
    cudaFree(d_controls);
    cudaFree(d_op_types);
}

void OperationBatch::add_single_qubit_op(size_t target, OpType type) {
    if (num_ops < MAX_BATCH) {
        op_types.push_back(type);
        targets.push_back(target);
        num_ops++;
    }
}

void OperationBatch::add_two_qubit_op(size_t control, size_t target, OpType type) {
    if (num_ops < MAX_BATCH) {
        op_types.push_back(type);
        controls.push_back(control);
        targets.push_back(target);
        num_ops++;
    }
}

void OperationBatch::upload_to_device() {
    if (num_ops > 0) {
        cudaMemcpy(d_targets, targets.data(), num_ops * sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_controls, controls.data(), num_ops * sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_op_types, op_types.data(), num_ops * sizeof(OpType), cudaMemcpyHostToDevice);
    }
}

void OperationBatch::clear() {
    num_ops = 0;
    op_types.clear();
    targets.clear();
    controls.clear();
} 