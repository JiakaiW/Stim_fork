#include <chrono>
#include <iostream>
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

void runBenchmark(size_t num_qubits, size_t batch_size, size_t num_operations) {
    FrameSimulatorCPU cpu_sim(num_qubits, batch_size);
    FrameSimulatorGPU gpu_sim(num_qubits, batch_size);
    
    // Benchmark ZCX operations
    std::cout << "Benchmarking ZCX operations...\n";
    
    double cpu_time = measureTime([&]() {
        for (size_t i = 0; i < num_operations; i++) {
            cpu_sim.do_ZCX(i % (num_qubits-1), (i+1) % num_qubits);
        }
    });
    
    double gpu_time = measureTime([&]() {
        for (size_t i = 0; i < num_operations; i++) {
            gpu_sim.do_ZCX(i % (num_qubits-1), (i+1) % num_qubits);
        }
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
    
    gpu_time = measureTime([&]() {
        for (size_t i = 0; i < num_operations; i++) {
            gpu_sim.do_H(i % num_qubits);
        }
    });
    
    std::cout << "CPU time: " << cpu_time << "ms\n";
    std::cout << "GPU time: " << gpu_time << "ms\n";
    std::cout << "Speedup: " << cpu_time/gpu_time << "x\n";
}

int main() {
    // Test with different sizes
    std::vector<std::pair<size_t, size_t>> test_cases = {
        {100, 1000},    // Small case
        {1000, 10000},  // Medium case
        {10000, 100000} // Large case
    };
    
    for (const auto& [num_qubits, batch_size] : test_cases) {
        std::cout << "\nTesting with " << num_qubits << " qubits and batch size " 
                  << batch_size << "\n";
        std::cout << "----------------------------------------\n";
        runBenchmark(num_qubits, batch_size, 10000);
    }
    
    return 0;
} 