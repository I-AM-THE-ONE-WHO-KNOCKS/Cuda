#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <iostream>

#define Shared_Mem_Size 16*16*4

// CUDA kernel for vector addition
__global__ void tile_MatrixMul(int* a, int* b, int* c, int n, int tile_size) {
	//statically-sized memory
	__shared__ int A[Shared_Mem_Size];
	__shared__ int B[Shared_Mem_Size];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	//cal global row and col postions for this thread
	int row = by * tile_size + ty;
	int col = bx * tile_size + tx;

	//Intermidiate sum for element being written
	int temp_val = 0;

	//sweet tiles over entire matrix
	for (int i = 0; i < (n / tile_size); i++)
	{
		/*

			Every thread in a threadblock loads one element into shared memory
			The element location in shared memory corresponds to the thread's 
			position in the threadblock (e.g thread[0,0] loads for 
			A[0 * tile_size + 0] and B[0 * tile_size + 0])

			Explanation of indexing parameters
			for A:
					row*n: Indexes the global row for this thread (loop invariant)
					i*tile_size: Indexes new set of column each iteration 
					tx: Indexes the column within that set

			for B:
				    col: Indexes the global column this thread (loop invariant)
					i*tile_size*n: Indexes next set of rows each iteration 
					ty*n: Indexes the row within that set
		*/
		A[(ty * tile_size) + tx] = a[row * n + (i * tile_size + tx)];
		B[(ty * tile_size) + tx] = b[(i * tile_size * n + ty * n) + col];

		//Ensure all threads have loaded their data before proceeding
		__syncthreads();

		//cal all temp values for this tile
		for (int j = 0; j < tile_size; j++)
		{
			temp_val += A[(ty * tile_size) + j] * B[(j * tile_size) + tx];
		}

		//Ensure some threads dont progress and stomp current shared memory values
		__syncthreads();
	}
	c[(row * n) + col] = temp_val;
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
	tile_MatrixMul <<<grid, threads >>> (d_a, d_b, d_c, n, BLOCKS);

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