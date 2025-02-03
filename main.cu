#include <cuda.h>
#include <cuda_runtime.h>

#define GPU_FREQ 2880000000

__global__ void sleepKernel(int seconds) {
    // Calculate the time to stop (in clock cycles)
    clock_t start = clock();
    clock_t end = start + seconds * GPU_FREQ;

    // Spin-wait until the required time has passed
    while (clock() < end);
}

__global__ void accessMemory(CUdeviceptr d_data)
{

    // Sleep a while
    clock_t end = clock() + 1 * GPU_FREQ / 2;
    while (clock() < end);

    int* data = reinterpret_cast<int*>(d_data);
    data[0] = 0;
}

int main() {
    const int size = 1024;
    CUdeviceptr d_data;

    // Initialize CUDA Driver API
    cuInit(0);

    // Get the first CUDA device
    CUdevice device;
    cuDeviceGet(&device, 0);

    // Create a CUDA context
    CUcontext ctx;
    cuCtxCreate(&ctx, 0, device);

    CUstream stream1, stream2;
    cuStreamCreate(&stream1, CU_STREAM_NON_BLOCKING);
    cuStreamCreate(&stream2, CU_STREAM_NON_BLOCKING);

    CUevent ctxEvent;
    cuEventCreate(&ctxEvent, CU_EVENT_DEFAULT);

    cuMemAllocAsync(&d_data, 2 * size * sizeof(int), stream1);  // schedule allocation on stream1
    sleepKernel<<<1,1,0,stream1>>>(1);    // schedule sleep for 1 second on stream1
    cuCtxRecordEvent(ctx, ctxEvent);
    sleepKernel<<<1,1,0,stream2>>>(1);    // schedule sleep for 1 second on stream2
    cuCtxWaitEvent(ctx, ctxEvent);
    accessMemory<<<1,1,0,stream2>>>(d_data);       // schedule access the memory on stream2 (should by synchronised with allocation)
    cuMemFreeAsync(d_data, stream2);                                      // schedule free the memory on stream2

    // Cleanup
    cuCtxSynchronize();
    cuCtxDestroy(ctx);
    return 0;
}
