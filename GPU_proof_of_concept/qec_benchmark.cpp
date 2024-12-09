#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <cstdlib>
#include "frame_simulator_cpu.h"
#include "frame_simulator_gpu.h"
#include "../src/stim/gen/gen_rep_code.h"
#include "../src/stim/simulators/frame_simulator.h"
#include "debug_flags.h"

// Global debug flag definition
CUDA_CALLABLE bool g_debug_output = false;

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

double runOriginalFrameSimBenchmark(size_t distance, size_t batch_size, size_t num_rounds) {
    // Generate repetition code circuit
    stim::CircuitGenParameters params(
        num_rounds,                    // rounds: uint64_t
        static_cast<uint32_t>(distance), // distance: uint32_t
        std::string("memory"));         // task: std::string
    auto circuit = stim::generate_rep_code_circuit(params).circuit;
    
    // Initialize simulator
    std::mt19937_64 rng(0);  // Fixed seed for reproducibility
    auto stats = circuit.compute_stats();
    stim::FrameSimulator<64> sim(
        stats, 
        stim::FrameSimulatorMode::STORE_DETECTIONS_TO_MEMORY,
        batch_size,
        std::move(rng));

    // Run benchmark
    double time = measureTime([&]() {
        sim.do_circuit(circuit);
    });
    
    if (g_debug_output) {
        std::cout << "Original Frame Simulator time: " << time << "ms\n";
    }
    return time;  // Return the time for comparison
}

void runQECBenchmark(size_t distance, size_t batch_size, size_t num_rounds, double orig_time) {
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
    
    // Only print detailed timing information if debug output is enabled
    // if (g_debug_output) {
    std::cout << "CPU time: " << cpu_time << "ms\n";
    std::cout << "GPU time: " << gpu_time << "ms\n";
    std::cout << "GPU Speedup vs CPU: " << cpu_time/gpu_time << "x\n";
    std::cout << "GPU Speedup vs Original: " << orig_time/gpu_time << "x\n";
    // }
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [--debug] distance batch_size num_rounds\n";
    std::cout << "Options:\n";
    std::cout << "  --debug      Enable detailed timing and debug output\n";
    std::cout << "  distance     Code distance to test\n";
    std::cout << "  batch_size   Number of parallel simulations (default: 10000)\n";
    std::cout << "  num_rounds   Number of QEC cycles to run (default: 1000)\n";
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    bool debug_flag = false;
    size_t distance = 0;
    size_t batch_size = 10000;  // Default batch size
    size_t num_rounds = 1000;   // Default number of rounds
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--debug") {
            debug_flag = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (distance == 0) {
            try {
                distance = std::stoul(arg);
            } catch (const std::exception&) {
                std::cerr << "Invalid distance: " << arg << "\n";
                print_usage(argv[0]);
                return 1;
            }
        } else if (batch_size == 10000) {  // Only set if not already set
            try {
                batch_size = std::stoul(arg);
            } catch (const std::exception&) {
                std::cerr << "Invalid batch size: " << arg << "\n";
                print_usage(argv[0]);
                return 1;
            }
        } else if (num_rounds == 1000) {  // Only set if not already set
            try {
                num_rounds = std::stoul(arg);
            } catch (const std::exception&) {
                std::cerr << "Invalid number of rounds: " << arg << "\n";
                print_usage(argv[0]);
                return 1;
            }
        } else {
            std::cerr << "Too many arguments\n";
            print_usage(argv[0]);
            return 1;
        }
    }
    
    if (distance == 0) {
        std::cerr << "Distance must be specified\n";
        print_usage(argv[0]);
        return 1;
    }
    
    // Set debug flag
    g_debug_output = debug_flag;
    
    // Always print the test case header
    std::cout << "\nTesting repetition code with:\n"
              << "  distance: " << distance << "\n"
              << "  batch_size: " << batch_size << "\n"
              << "  num_rounds: " << num_rounds << "\n"
              << "----------------------------------------\n";
    
    // Run original frame simulator benchmark and get its time
    double orig_time = runOriginalFrameSimBenchmark(distance, batch_size, num_rounds);
    
    // Run our CPU/GPU implementation benchmark with the original time
    runQECBenchmark(distance, batch_size, num_rounds, orig_time);
    
    return 0;
} 