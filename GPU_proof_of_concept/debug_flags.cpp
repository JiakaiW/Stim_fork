#include "debug_flags.h"

// Host debug flag
bool g_debug_output = false;

// Device debug flag (defined in CUDA code)
#ifdef __CUDACC__
__device__ bool g_debug_output_device = false;
#endif 