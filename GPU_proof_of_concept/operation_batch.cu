#include "operation_batch.cuh"

OperationBatch::OperationBatch() : num_ops(0) {
    targets.reserve(MAX_BATCH);
    controls.reserve(MAX_BATCH);
    cudaMalloc(&d_targets, MAX_BATCH * sizeof(size_t));
    cudaMalloc(&d_controls, MAX_BATCH * sizeof(size_t));
}

OperationBatch::~OperationBatch() {
    cudaFree(d_targets);
    cudaFree(d_controls);
}

void OperationBatch::add_single_qubit_op(size_t target) {
    if (num_ops < MAX_BATCH) {
        targets.push_back(target);
        num_ops++;
    }
}

void OperationBatch::add_two_qubit_op(size_t control, size_t target) {
    if (num_ops < MAX_BATCH) {
        controls.push_back(control);
        targets.push_back(target);
        num_ops++;
    }
}

void OperationBatch::upload_to_device() {
    if (num_ops > 0) {
        cudaMemcpy(d_targets, targets.data(), num_ops * sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_controls, controls.data(), num_ops * sizeof(size_t), cudaMemcpyHostToDevice);
    }
}

void OperationBatch::clear() {
    num_ops = 0;
    targets.clear();
    controls.clear();
} 