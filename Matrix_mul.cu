#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <iostream>

// CUDA kernel for vector addition
__global__ void MatrixMul(int* a, int* b, int* c, int n) {
	// row
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	//col
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;
	int temp_sum = 0;
	// boundary guard
	if ((row < n) && (col < n)) {
		for (int k = 0; k < n; k++)
		{
			temp_sum += a[row*n+k]*b[k*n+col];
		}
		c[row*n+col] = temp_sum;
	}
}

// Initialize
void Mat_init(int* a, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			a[i * n + j] = rand() % 100;
		}
	}
}

// Check MatrixMul add result
void check_answer(int* a, int* b, int* c, int n) {
	int* result = (int*)malloc(n * n * sizeof(int));
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			for (int k = 0; k < n; k++)
			{
				result[i * n + j] += a[i * n + k] * b[k * n + j];
			}
		}
	}

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			assert(c[i * n + j] == result[i * n + j]);
		}
	}
}

int main() {
	
	// matrix of size 1024 x 1024
	int n = 1 << 10;
	//host memory pointers
	int* h_a, * h_b, * h_c;
	// Allocation size for all vectors
	size_t bytes = sizeof(int) * n * n;

	h_a = (int*)malloc(bytes);
	h_b = (int*)malloc(bytes);
	h_c = (int*)malloc(bytes);

	//device memory pointers
	int* d_a, * d_b, * d_c;
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	// Initialize vectors a and b with random values between 0 and 99
	Mat_init(h_a, n);
	Mat_init(h_b, n);

	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

	// Threadblock size
	int BLOCKS = 16;

	// Grid size
	int GRID = (int)ceil(n / BLOCKS);

	//use dim3 objects
	dim3 grid(GRID, GRID);
	dim3 threads(BLOCKS, BLOCKS);
	
	// Launch kernel on default stream w/o shmem
	MatrixMul <<<grid, threads >>> (d_a, d_b, d_c, n);

	//copy result back to host
	cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

	// Check result for errors
	check_answer(h_a, h_b, h_c, n);

	free(h_a);
	free(h_b);
	free(h_c);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	printf("COMPLETED SUCCESFULLY\n");

	return 0;
}