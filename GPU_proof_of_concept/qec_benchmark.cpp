#include <chrono>
#include <iostream>
#include <vector>
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

// Simulates a repetition code cycle
void runRepCodeCycle(FrameSimulatorBase& sim, const std::vector<size_t>& data_qubits, const std::vector<size_t>& measurement_qubits) {
    // For GPU simulator, only queue operations
    if (auto* gpu_sim = dynamic_cast<FrameSimulatorGPU*>(&sim)) {
        // First round of CNOTs
        for (size_t i = 0; i < measurement_qubits.size(); i++) {
            gpu_sim->queue_ZCX(data_qubits[i], measurement_qubits[i]);
        }
        
        // Second round of CNOTs
        for (size_t i = 0; i < measurement_qubits.size(); i++) {
            gpu_sim->queue_ZCX(data_qubits[i + 1], measurement_qubits[i]);
        } 
        
        // Queue measurements
        for (size_t m : measurement_qubits) {
            gpu_sim->queue_M(m);
        }
    } else {
        // Original CPU implementation remains unchanged
        // First round of CNOTs
        for (size_t i = 0; i < measurement_qubits.size(); i++) {
            sim.do_ZCX(data_qubits[i], measurement_qubits[i]);
        }
        
        // Second round of CNOTs
        for (size_t i = 0; i < measurement_qubits.size(); i++) {
            sim.do_ZCX(data_qubits[i + 1], measurement_qubits[i]);
        }
        
        // Measure syndrome qubits
        for (size_t m : measurement_qubits) {
            sim.do_M(m);
        }
    }
}

void runQECBenchmark(size_t distance, size_t batch_size, size_t num_rounds) {
    // For distance d, we need d data qubits and (d-1) measurement qubits
    size_t num_data = distance;
    size_t num_measurements = distance - 1;
    size_t total_qubits = num_data + num_measurements;
    
    // Create simulators
    FrameSimulatorCPU cpu_sim(total_qubits, batch_size);
    FrameSimulatorGPU gpu_sim(total_qubits, batch_size);
    
    // Create qubit lists
    std::vector<size_t> data_qubits;
    std::vector<size_t> measurement_qubits;
    for (size_t i = 0; i < num_data; i++) {
        data_qubits.push_back(i * 2);  // Even indices for data qubits
    }
    for (size_t i = 0; i < num_measurements; i++) {
        measurement_qubits.push_back(i * 2 + 1);  // Odd indices for measurement qubits
    }
    
    // Run CPU benchmark
    double cpu_time = measureTime([&]() {
        for (size_t r = 0; r < num_rounds; r++) {
            runRepCodeCycle(cpu_sim, data_qubits, measurement_qubits);
        }
    });
    
    // Run GPU benchmark - only measure execution time
    auto* gpu = dynamic_cast<FrameSimulatorGPU*>(&gpu_sim);
    
    // Queue operations without timing
    for (size_t r = 0; r < num_rounds; r++) {
        runRepCodeCycle(gpu_sim, data_qubits, measurement_qubits);
    }
    
    // Now measure only the GPU execution time
    double gpu_time = measureTime([&]() {
        gpu->execute_queued_operations();
    });
    
    std::cout << "CPU time: " << cpu_time << "ms\n";
    std::cout << "GPU time: " << gpu_time << "ms\n";
    std::cout << "Speedup: " << cpu_time/gpu_time << "x\n";
}

int main() {
    // Test with different code distances and batch sizes
    std::vector<std::pair<size_t, size_t>> test_cases = {
        {25, 100000},     // Medium distance code
        {50, 100000},     // Large distance code
        {75, 100000},     // Very large distance code
    };
    
    for (const auto& test_case : test_cases) {
        size_t distance = test_case.first;
        size_t batch_size = test_case.second;
        std::cout << "\nTesting repetition code with distance " << distance 
                  << " and batch size " << batch_size << "\n";
        std::cout << "----------------------------------------\n";
        runQECBenchmark(distance, batch_size, 1000);  // Run 1000 QEC cycles
    }
    
    return 0;
} 