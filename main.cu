#include <cuda.h>
#include <iostream>
#include <chrono>
#include <thread>

// Kernel to simulate a 3-second sleep
__global__ void sleepKernel() {
    // Sleep for ~3 seconds (busy wait loop)
    unsigned long long startClock = clock64();
    unsigned long long waitClock = 9000000000; // 3 seconds in nanoseconds for a busy GPU clock wait
    while (clock64() - startClock < waitClock) {
        // Busy waiting
    }
}

// Kernel to perform a quick task
__global__ void quickKernel() {
    printf("Quick kernel executed!\n");
}

int main() {
    // Initialize CUDA Driver API
    cuInit(0);

    // Get the first CUDA device
    CUdevice device;
    cuDeviceGet(&device, 0);

    // Create a CUDA context
    CUcontext ctx;
    cuCtxCreate(&ctx, 0, device);

    cuCtxSetCurrent(ctx);

    // Create two CUDA streams with non-blocking behavior
    CUstream stream1, stream2;
    cuStreamCreate(&stream1, CU_STREAM_NON_BLOCKING);
    cuStreamCreate(&stream2, CU_STREAM_NON_BLOCKING);

    // Create CUDA events to measure timing
    CUevent start, event1, event2;
    cuEventCreate(&start, CU_EVENT_DEFAULT);
    cuEventCreate(&event1, CU_EVENT_DEFAULT);
    cuEventCreate(&event2, CU_EVENT_DEFAULT);

    // Record the start event on the default stream
    cuEventRecord(start, 0);

    // Launch the sleep kernel on stream1
    std::cout << "Launching sleep kernel..." << std::endl;
    sleepKernel<<<1, 1, 0, stream1>>>();
    cuEventRecord(event1, stream1);

    // Launch the quick kernel on stream2
    std::cout << "Launching quick kernel..." << std::endl;
    quickKernel<<<1, 1, 0, stream2>>>();
    cuEventRecord(event2, stream2);

    cuStreamSynchronize(stream1); // Synchronize only stream1
    cuStreamSynchronize(stream2); // Synchronize only stream2
    // Measure elapsed time
    float elapsed1, elapsed2;
    cuEventElapsedTime(&elapsed1, start, event1);
    cuEventElapsedTime(&elapsed2, start, event2);

    std::cout << "Sleep kernel finished at: " << elapsed1 << " ms" << std::endl;
    std::cout << "Quick kernel finished at: " << elapsed2 << " ms" << std::endl;

    // Cleanup
    cuEventDestroy(start);
    cuEventDestroy(event1);
    cuEventDestroy(event2);
    cuStreamDestroy(stream1);
    cuStreamDestroy(stream2);
    cuCtxDestroy(ctx);

    return 0;
}
