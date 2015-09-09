#ifndef __CUDACC__
#define __CUDACC__
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>

#include <math.h>
#include <stdio.h>
#include <random>
#include <iomanip>
#include <iostream>

#define N 256
#define BLOCKSIZE 16

cudaError_t histogramCuda(int *freq, int *freq2, const int *vals, int bin, float &time, float &shared_time);

__global__ void naivehistKernel(int *freq, const int *vals) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(&freq[vals[id]], 1);
}

__global__ void sharedhistKernel(int *freq, const int *vals) {
    __shared__ int temp[BLOCKSIZE];
	unsigned int tid = threadIdx.x;
    temp[tid] = 0;
    __syncthreads();

    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(&temp[vals[id]], 1);
    __syncthreads();

    atomicAdd(&(freq[tid]), temp[tid]);
}

int main() {
    const int bin = 10 + 1;
	const int vals[N] = {1,9,0,6,10,1,2,4,6,9,9,5,10,10,8,3,
							7,7,5,0,4,8,5,10,0,8,9,10,10,5,1,0,
							0,3,10,5,4,3,5,1,4,4,1,8,1,6,10,3,
							0,2,5,7,10,4,1,6,6,5,4,0,5,0,4,4,
							4,1,4,6,8,9,0,0,9,4,10,10,10,1,4,9,
							0,1,7,9,7,10,10,0,5,9,1,6,7,0,3,9,
							8,5,4,8,4,1,0,6,9,2,1,2,3,6,10,6,
							4,9,6,0,2,6,2,6,3,8,6,0,2,2,1,1,
							3,10,6,7,4,5,3,8,4,9,5,9,7,9,5,8,
							9,6,8,8,7,10,10,7,9,6,3,7,5,3,8,10,
							2,5,8,6,9,1,1,2,3,7,7,8,2,2,10,5,
							7,3,9,4,1,9,7,7,6,9,3,5,8,8,8,2,
							7,2,7,6,1,6,8,7,10,1,2,6,5,6,1,0,
							6,8,9,6,1,9,10,4,1,7,1,8,5,0,9,10,
							5,6,2,9,6,3,10,0,0,6,1,8,7,0,6,2,
							3,10,1,1,10,10,5,6,9,0,2,8,5,5,10,4};
	int freq[bin] = { 0 };
	int freq2[bin] = { 0 };
	float time = 0.0f;
	float shared_time = 0.0f;

    cudaError_t cudaStatus = histogramCuda(freq, freq2, vals, bin, time, shared_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "histogramCuda failed!");
        return 1;
    }

	for(int i = 0; i < bin; i++) {
       std::cout << std::left;
       std::cout << std::setw(5) << i;
       std::cout << std::setw(5) << freq[i];
       std::cout << std::endl;
	}

	std::cout << "Histogram GPU Implementation" << std::endl;
	std::cout << "Execution Time : " << time / 1000 << " seconds" << std::endl;
	std::cout << "Effective Bandwidth : " << (N*sizeof(int)*2) / (time / 1000) << " GB/s" << std::endl;
	std::cout << std::endl;

	for(int i = 0; i < bin; i++) {
       std::cout << std::left;
       std::cout << std::setw(5) << i;
       std::cout << std::setw(5) << freq2[i];
       std::cout << std::endl;
	}

	std::cout << "Histogram Shared Memory GPU Implementation" << std::endl;
	std::cout << "Execution Time : " << shared_time / 1000 << " seconds" << std::endl;
	std::cout << "Effective Bandwidth : " << (N*sizeof(int)*2) / (shared_time / 1000) << " GB/s" << std::endl;
	std::cout << std::endl;

	cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

cudaError_t histogramCuda(int *freq, int *freq2, const int *vals, int bin, float &time, float &shared_time) {
    int *dev_vals = 0;
    int *dev_freq = 0;
	int *dev_freq2 = 0;
	float milliseconds = 0.0f;
	float milliseconds1 = 0.0f;
    cudaError_t cudaStatus;
	dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid(N/BLOCKSIZE);
	cudaEvent_t start, stop;
	cudaEvent_t start1, stop1;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_freq, bin * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMemset(dev_freq, 0, bin * sizeof(int));
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_freq2, bin * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMemset(dev_freq2, 0, bin * sizeof(int));
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_vals, N * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_vals, vals, N * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaEventRecord(start);
    naivehistKernel<<<dimGrid, dimBlock>>>(dev_freq, dev_vals);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaThreadSynchronize();

	cudaEventRecord(start1);
    sharedhistKernel<<<dimGrid, dimBlock>>>(dev_freq2, dev_vals);
	cudaEventRecord(stop1);
	cudaEventSynchronize(stop1);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "histKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching histKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(freq, dev_freq, bin * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(freq2, dev_freq2, bin * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventElapsedTime(&milliseconds1, start1, stop1);
	time = milliseconds;
	shared_time = milliseconds1;

Error:
    cudaFree(dev_freq);
	cudaFree(dev_freq2);
    cudaFree(dev_vals);
    
    return cudaStatus;
}
