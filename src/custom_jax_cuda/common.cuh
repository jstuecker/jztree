#ifndef CUSTOM_JAX_COMMON_H
#define CUSTOM_JAX_COMMON_H

struct PosMass {
    float x, y, z, mass;
};

#define MAXP 6

// Helper macro to dispatch a templated kernel on runtime integer p (1..6).
// Usage: LAUNCH_KERNEL_SWITCH(p, KernelName, grid_size, block_size, stream, arg1, arg2, ...)
#define LAUNCH_KERNEL_SWITCH(P, KERNEL, GRID, BLOCK, STREAM, ...) \
    switch(P) { \
        case 1: KERNEL<1><<<GRID, BLOCK, 0, STREAM>>>(__VA_ARGS__); break; \
        case 2: KERNEL<2><<<GRID, BLOCK, 0, STREAM>>>(__VA_ARGS__); break; \
        case 3: KERNEL<3><<<GRID, BLOCK, 0, STREAM>>>(__VA_ARGS__); break; \
        /*case 4: KERNEL<4><<<GRID, BLOCK, 0, STREAM>>>(__VA_ARGS__); break; \
        case 5: KERNEL<5><<<GRID, BLOCK, 0, STREAM>>>(__VA_ARGS__); break; \
        case 6: KERNEL<6><<<GRID, BLOCK, 0, STREAM>>>(__VA_ARGS__); break;*/ \
        default: break; \
    }

#endif