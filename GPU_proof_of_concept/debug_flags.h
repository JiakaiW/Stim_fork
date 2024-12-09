#pragma once

#ifdef __CUDACC__
#define CUDA_CALLABLE __device__ __host__
#else
#define CUDA_CALLABLE
#endif

// Host debug flag
extern bool g_debug_output;

// Device debug flag (only available in CUDA code)
#ifdef __CUDACC__
__device__ extern bool g_debug_output_device;
#endif