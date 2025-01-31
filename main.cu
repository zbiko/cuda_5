#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define ASSERT_DRV(...) __VA_ARGS__

__global__ void sleepKernel(int seconds) {
    // Calculate the time to stop (in clock cycles)
    clock_t start = clock();
    clock_t end = start + seconds * CLOCKS_PER_SEC * 3000;

    // Spin-wait until the required time has passed
    while (clock() < end);
    if(threadIdx.x == 0)
    {
        printf("sleepKernel slept for %d seconds\n", seconds);
    }
}

__global__ void accessMemory(CUdeviceptr d_data)
{
    // Sleep 1 second
    clock_t end = clock() + 1 * CLOCKS_PER_SEC * 3000;
    while (clock() < end);

    int* data = reinterpret_cast<int*>(d_data);
    data[0] = 0;
    if(threadIdx.x == 0)
    {
        printf("AccessMemory finished \n" );
    }
}

int main() {
    // Allocate memory on the device
    const int size = 1024;
    CUdeviceptr d_data, d_data2;

    // Initialize CUDA Driver API
    ASSERT_DRV(cuInit(0));

    // Get the first CUDA device
    CUdevice device;
    ASSERT_DRV(cuDeviceGet(&device, 0));

    // Create a CUDA context
    CUcontext ctx;
    ASSERT_DRV(cuCtxCreate(&ctx, 0, device));

    CUstream stream1;
    ASSERT_DRV(cuStreamCreate(&stream1, CU_STREAM_NON_BLOCKING));
    ASSERT_DRV(cuMemAllocAsync(&d_data, 2 * size * sizeof(int), stream1));
    sleepKernel<<<1,1,0,stream1>>>(1);
    accessMemory<<<1,1,0,stream1>>>(d_data);
    cudaStreamSynchronize(stream1);
    ASSERT_DRV(cuMemFreeAsync(d_data, stream1));

    // Cleanup
    std::cout << "Ctx synchronize... " << std::endl;
    ASSERT_DRV(cuCtxSynchronize());
    std::cout << "Ctx synchronize done. " << std::endl;
    ASSERT_DRV(cuCtxDestroy(ctx));
    return 0;
}
