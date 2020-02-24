#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <iostream>

// CUDA kernel for vector addition
__global__ void vectorAdd(int* a, int* b, int* c, int n) {
	// Calculate global thread ID (tid)
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	// Vector boundary guard
	if (tid < n) {
		// Each thread adds a single element
		c[tid] = a[tid] + b[tid];
	}
}

// Initialize vector of size n to int between 0-99
void vector_init(int* a, int n) {
	for (int i = 0; i < n; i++) {
		a[i] = rand() % 100;
	}
}

// Check vector add result
void check_answer(int* a, int* b, int* c, int n) {
	for (int i = 0; i < n; i++) {
		assert(c[i] == a[i] + b[i]);
	}
}

int main() {
	//Get the device id for cuda calls
	int id = cudaGetDevice(&id);
	// Vector size of 2^16 (65536 elements)
	int n = 1 << 16;
	//unified memory pointers
	int* a, * b, * c;
	// Allocation size for all vectors
	size_t bytes = sizeof(int) * n;

	// Allocate device memory
	cudaMallocManaged(&a, bytes);
	cudaMallocManaged(&b, bytes);
	cudaMallocManaged(&c, bytes);

	// Initialize vectors a and b with random values between 0 and 99
	vector_init(a, n);
	vector_init(b, n);

	// Threadblock size
	int BLOCKS = 256;

	// Grid size
	int GRID = (int)ceil(n / BLOCKS);
	
	//call cuda kernrl
	//for prefetching a and b vectors to device to make sure data gets copied before kernel call
	cudaMemPrefetchAsync(a, bytes, id);
	cudaMemPrefetchAsync(b, bytes, id);
	// Launch kernel on default stream w/o shmem
	vectorAdd <<<GRID, BLOCKS >>> (a, b, c, n);

	//wait for all the previous operations before using values
	cudaDeviceSynchronize();

	//for prefetching c vector to the host
	cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);
	// Check result for errors
	check_answer(a, b, c, n);

	printf("COMPLETED SUCCESFULLY\n");

	return 0;
}