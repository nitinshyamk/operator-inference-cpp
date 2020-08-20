#pragma once
#ifndef OPINF_GPU_TIMER_H
#define OPINF_GPU_TIMER_H

#include <cuda_runtime_api.h>
#include <iostream>

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        std::cout << "Elapsed time " << elapsed << std::endl;
        return elapsed;
    }
};

#endif OPINF_GPU_TIMER_H