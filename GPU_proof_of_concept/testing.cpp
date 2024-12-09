#include <chrono>
#include <iostream>
#include <cassert>
#include <random>
#include "frame_simulator_cpu.h"
#include "frame_simulator_gpu.h"

// Helper function to measure execution time
template<typename Func>
double measureTime(Func f) {
    auto start = std::chrono::high_resolution_clock::now();
    f();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Helper to compare measurement results
bool compare_results(const std::vector<bool>& cpu_results, 
                    const std::vector<bool>& gpu_results) {
    if (cpu_results.size() != gpu_results.size()) {
        std::cout << "Result size mismatch! CPU: " << cpu_results.size() 
                  << " GPU: " << gpu_results.size() << "\n";
        return false;
    }
    
    for (size_t i = 0; i < cpu_results.size(); i++) {
        if (cpu_results[i] != gpu_results[i]) {
            std::cout << "Mismatch at index " << i 
                      << " CPU: " << cpu_results[i] 
                      << " GPU: " << gpu_results[i] << "\n";
            return false;
        }
    }
    return true;
}

void verifyImplementations(size_t num_qubits, size_t batch_size) {
    std::cout << "\nVerifying implementations with " << num_qubits 
              << " qubits and batch size " << batch_size << "...\n";
    
    FrameSimulatorCPU cpu_sim(num_qubits, batch_size);
    FrameSimulatorGPU gpu_sim(num_qubits, batch_size);
    
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::uniform_int_distribution<size_t> qubit_dist(0, num_qubits - 1);
    
    // Keep track of measured qubits for final verification
    std::vector<size_t> measured_qubits;
    
    // Test sequence of operations
    const size_t num_test_ops = 100;
    for (size_t i = 0; i < num_test_ops; i++) {
        // Randomly choose operation type and qubits
        int op_type = i % 3;  // Cycle through ZCX, H, M
        size_t q1 = qubit_dist(rng);
        size_t q2 = qubit_dist(rng);
        while (q2 == q1) q2 = qubit_dist(rng);  // Ensure different qubits for ZCX
        
        switch (op_type) {
        case 0:  // ZCX
            cpu_sim.do_ZCX(q1, q2);
            gpu_sim.do_ZCX(q1, q2);
            break;
        case 1:  // H
            cpu_sim.do_H(q1);
            gpu_sim.do_H(q1);
            break;
        case 2:  // M
            cpu_sim.do_M(q1);
            gpu_sim.do_M(q1);
            measured_qubits.push_back(q1);
            break;
        }
    }
    
    // Verify all measurement results at the end
    for (size_t qubit : measured_qubits) {
        auto cpu_results = cpu_sim.get_measurement_results(qubit);
        auto gpu_results = gpu_sim.get_measurement_results(qubit);
        if (!compare_results(cpu_results, gpu_results)) {
            std::cout << "Measurement mismatch for qubit " << qubit << "\n";
            return;
        }
    }
    
    std::cout << "Verification passed! CPU and GPU implementations produce identical results.\n";
}

void runBenchmark(size_t num_qubits, size_t batch_size, size_t num_operations) {
    FrameSimulatorCPU cpu_sim(num_qubits, batch_size);
    FrameSimulatorGPU gpu_sim(num_qubits, batch_size);
    auto* gpu = dynamic_cast<FrameSimulatorGPU*>(&gpu_sim);
    
    // Benchmark ZCX operations
    std::cout << "Benchmarking ZCX operations...\n";
    
    double cpu_time = measureTime([&]() {
        for (size_t i = 0; i < num_operations; i++) {
            cpu_sim.do_ZCX(i % (num_qubits-1), (i+1) % num_qubits);
        }
    });
    
    // Queue all ZCX operations first
    for (size_t i = 0; i < num_operations; i++) {
        gpu->queue_ZCX(i % (num_qubits-1), (i+1) % num_qubits);
    }
    
    // Measure only execution time
    double gpu_time = measureTime([&]() {
        gpu->execute_queued_operations();
    });
    
    std::cout << "CPU time: " << cpu_time << "ms\n";
    std::cout << "GPU time: " << gpu_time << "ms\n";
    std::cout << "Speedup: " << cpu_time/gpu_time << "x\n\n";
    
    // Benchmark H operations
    std::cout << "Benchmarking H operations...\n";
    
    cpu_time = measureTime([&]() {
        for (size_t i = 0; i < num_operations; i++) {
            cpu_sim.do_H(i % num_qubits);
        }
    });
    
    // Queue all H operations first
    for (size_t i = 0; i < num_operations; i++) {
        gpu->queue_H(i % num_qubits);
    }
    
    // Measure only execution time
    gpu_time = measureTime([&]() {
        gpu->execute_queued_operations();
    });
    
    std::cout << "CPU time: " << cpu_time << "ms\n";
    std::cout << "GPU time: " << gpu_time << "ms\n";
    std::cout << "Speedup: " << cpu_time/gpu_time << "x\n";
}

int main() {
    const size_t num_qubits = 1000;  // Fixed number of qubits
    const size_t base_batch_size = 64;
    const size_t max_power = 6;  // Will go up to 64 * 4^6
    
    std::vector<size_t> batch_sizes;
    size_t current_batch = base_batch_size;
    for (size_t i = 0; i <= max_power; i++) {
        batch_sizes.push_back(current_batch);
        current_batch *= 4;
    }
    
    // First verify correctness with smallest batch size
    verifyImplementations(num_qubits, base_batch_size);
    
    // Then run performance benchmarks
    std::cout << "\nRunning performance benchmarks...\n";
    std::cout << "Fixed number of qubits: " << num_qubits << "\n";
    
    for (size_t batch_size : batch_sizes) {
        std::cout << "\nTesting with batch size " << batch_size << "\n";
        std::cout << "----------------------------------------\n";
        
        FrameSimulatorCPU cpu_sim(num_qubits, batch_size);
        FrameSimulatorGPU gpu_sim(num_qubits, batch_size);
        auto* gpu = dynamic_cast<FrameSimulatorGPU*>(&gpu_sim);
        
        // Benchmark ZCX operations
        std::cout << "Benchmarking ZCX operations...\n";
        
        double cpu_time = measureTime([&]() {
            for (size_t i = 0; i < 10000; i++) {
                cpu_sim.do_ZCX(i % (num_qubits-1), (i+1) % num_qubits);
            }
        });
        
        // Queue all ZCX operations first
        for (size_t i = 0; i < 10000; i++) {
            gpu->queue_ZCX(i % (num_qubits-1), (i+1) % num_qubits);
        }
        
        // Measure only execution time
        double gpu_time = measureTime([&]() {
            gpu->execute_queued_operations();
        });
        
        std::cout << "CPU time: " << cpu_time << "ms\n";
        std::cout << "GPU time: " << gpu_time << "ms\n";
        std::cout << "Speedup: " << cpu_time/gpu_time << "x\n";
    }
    
    return 0;
} 