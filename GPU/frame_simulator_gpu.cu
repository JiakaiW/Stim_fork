#include "frame_simulator_gpu.h"
#include <stdexcept>

// CUDA kernels operate on both packed bits and multiple qubits in parallel
__global__ void zcx_kernel(uint64_t *x_table, uint64_t *z_table, 
                          const size_t *controls, const size_t *targets,
                          size_t num_ops, size_t num_words) {
    // Thread handles one word of one operation
    size_t op_idx = blockIdx.x;
    size_t word_idx = threadIdx.x;
    
    if (op_idx < num_ops && word_idx < num_words) {
        size_t control = controls[op_idx];
        size_t target = targets[op_idx];
        
        // Process one word (64 shots) for this control-target pair
        z_table[control * num_words + word_idx] ^= z_table[target * num_words + word_idx];
        x_table[target * num_words + word_idx] ^= x_table[control * num_words + word_idx];
    }
}

__global__ void hadamard_kernel(uint64_t *x_table, uint64_t *z_table,
                               const size_t *targets, size_t num_ops, 
                               size_t num_words) {
    // Thread handles one word of one operation
    size_t op_idx = blockIdx.x;
    size_t word_idx = threadIdx.x;
    
    if (op_idx < num_ops && word_idx < num_words) {
        size_t target = targets[op_idx];
        size_t idx = target * num_words + word_idx;
        
        // Process one word (64 shots) for this target
        uint64_t temp = x_table[idx];
        x_table[idx] = z_table[idx];
        z_table[idx] = temp;
    }
}

__global__ void measure_kernel(uint64_t *x_table, uint64_t *measurements,
                             const size_t *targets, size_t num_ops,
                             size_t num_words) {
    // Thread handles one word of one operation
    size_t op_idx = blockIdx.x;
    size_t word_idx = threadIdx.x;
    
    if (op_idx < num_ops && word_idx < num_words) {
        size_t target = targets[op_idx];
        size_t idx = target * num_words + word_idx;
        
        // Process one word (64 shots) for this target
        measurements[idx] = x_table[idx];
    }
}

class OperationBatch {
private:
    static constexpr size_t MAX_BATCH = 1024;
    std::vector<size_t> targets;
    std::vector<size_t> controls;  // for two-qubit gates
    size_t num_ops;

    // Device memory for operation data
    size_t *d_targets;
    size_t *d_controls;

public:
    OperationBatch() : num_ops(0) {
        targets.reserve(MAX_BATCH);
        controls.reserve(MAX_BATCH);
        cudaMalloc(&d_targets, MAX_BATCH * sizeof(size_t));
        cudaMalloc(&d_controls, MAX_BATCH * sizeof(size_t));
    }

    ~OperationBatch() {
        cudaFree(d_targets);
        cudaFree(d_controls);
    }

    void add_single_qubit_op(size_t target) {
        if (num_ops < MAX_BATCH) {
            targets.push_back(target);
            num_ops++;
        }
    }

    void add_two_qubit_op(size_t control, size_t target) {
        if (num_ops < MAX_BATCH) {
            controls.push_back(control);
            targets.push_back(target);
            num_ops++;
        }
    }

    void upload_to_device() {
        if (num_ops > 0) {
            cudaMemcpy(d_targets, targets.data(), num_ops * sizeof(size_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_controls, controls.data(), num_ops * sizeof(size_t), cudaMemcpyHostToDevice);
        }
    }

    void clear() {
        num_ops = 0;
        targets.clear();
        controls.clear();
    }

    size_t* get_device_targets() const { return d_targets; }
    size_t* get_device_controls() const { return d_controls; }
    size_t size() const { return num_ops; }
};

// Add operation batch to GPU simulator class
class FrameSimulatorGPU : public FrameSimulatorBase {
private:
    // ... existing members ...
    OperationBatch op_batch;

public:
    // Modified methods to use operation batching
    void do_ZCX(size_t control, size_t target) override {
        op_batch.add_two_qubit_op(control, target);
        
        if (op_batch.size() == 1024) {  // Max batch size
            flush_zcx_batch();
        }
    }

    void flush_zcx_batch() {
        if (op_batch.size() == 0) return;
        
        op_batch.upload_to_device();
        
        // Each block handles one operation, each thread handles one word
        dim3 grid(op_batch.size());
        dim3 block(num_words);
        
        zcx_kernel<<<grid, block>>>(
            d_x_table, d_z_table,
            op_batch.get_device_controls(),
            op_batch.get_device_targets(),
            op_batch.size(), num_words);
        
        cudaDeviceSynchronize();
        op_batch.clear();
    }

    // Similar modifications for H and M operations...
};

FrameSimulatorGPU::FrameSimulatorGPU(size_t num_qubits, size_t batch_size) 
    : FrameSimulatorBase(num_qubits, batch_size) {
    allocateMemory();
}

FrameSimulatorGPU::~FrameSimulatorGPU() {
    freeMemory();
}

void FrameSimulatorGPU::allocateMemory() {
    size_t size = num_qubits * batch_size * sizeof(bool);
    cudaMalloc(&d_x_table, size);
    cudaMalloc(&d_z_table, size);
    cudaMalloc(&d_measurements, size);
    
    // Initialize to false
    cudaMemset(d_x_table, 0, size);
    cudaMemset(d_z_table, 0, size);
    cudaMemset(d_measurements, 0, size);
}

void FrameSimulatorGPU::freeMemory() {
    cudaFree(d_x_table);
    cudaFree(d_z_table);
    cudaFree(d_measurements);
}

void FrameSimulatorGPU::do_H(size_t target) {
    int blockSize = 256;
    int numBlocks = (batch_size + blockSize - 1) / blockSize;
    
    hadamard_kernel<<<numBlocks, blockSize>>>(
        d_x_table, d_z_table, target, num_qubits, num_words);
    cudaDeviceSynchronize();
}

void FrameSimulatorGPU::do_M(size_t target) {
    int blockSize = 256;
    int numBlocks = (batch_size + blockSize - 1) / blockSize;
    
    measure_kernel<<<numBlocks, blockSize>>>(
        d_x_table, d_measurements, target, batch_size);
    cudaDeviceSynchronize();
}

std::vector<bool> FrameSimulatorGPU::get_measurement_results(size_t target_idx) {
    std::vector<bool> results(batch_size);
    cudaMemcpy(results.data(), 
               d_measurements + target_idx * batch_size,
               batch_size * sizeof(bool), 
               cudaMemcpyDeviceToHost);
    return results;
} 