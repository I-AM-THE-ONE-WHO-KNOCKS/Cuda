#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <iostream>

#define MASK_LEN 8

/*as mask is never changing we can define a constant memory on the device side so that
 we do not have to copu again and again and loading from const cache is much much faster that 
 loading from d-ram.
*/
__constant__ int mask[MASK_LEN];

__global__ void conv_1d(int* a, int* c, int n) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	//cal the radius of the mask(mid point)
	int r = MASK_LEN / 2;
	//cal the start point of for the element
	int start = id - r;
	int temp = 0;
	for (int j = 0; j < MASK_LEN; j++)
	{
		if ((start + j >= 0) && (start + j < n))
		{
			temp += a[start + j] * mask[j];
		}
	}
	c[id] = temp;
}

// Initialize
void Array_init(int* a, int n, int div) {
	for (int i = 0; i < n; i++) {
			a[i] = rand() % div;
		}
}

void check_answer(int* a, int* b, int* c, int n, int m) {
	int radius = m / 2;
	int temp;
	int start;
	for (int i = 0; i < n; i++)
	{
		start = i - radius;
		temp = 0;
		for (int j = 0; j < m; j++)
		{
			if ((start + j >= 0) && (start + j < n))
			{
				temp += a[start + j] * b[j];
			}
		}
		assert(temp == c[i]);
	}
}

int main() {
	
	// number of elements in result array
	int n = 1 << 16;
	
	int n_bytes = n * sizeof(int);

	//num of elemets in mask
	int m = 8;

	int m_bytes = m * sizeof(int);

	//allocate the array
	int* h_arr = new int[n];

	Array_init(h_arr, n, 100);

	//allocate the mask and intialize it
	int* h_mask = new int[m];
	Array_init(h_mask, m, 10);

	//allocate space for result
	int* h_result = new int[n];

	//allocate space on device memory
	int* d_arr, * d_res;
	cudaMalloc(&d_arr, n_bytes);
	cudaMalloc(&d_res, n_bytes);

	cudaMemcpy(d_arr, h_arr, n_bytes, cudaMemcpyHostToDevice);
	//special function to copy to a symbol
	cudaMemcpyToSymbol(mask, h_mask, m_bytes);

	int threads = 256;

	int grid = (n + threads - 1) / threads;

	conv_1d <<<grid, threads>>> (d_arr, d_res, n);

	cudaMemcpy(h_result, d_res, n_bytes, cudaMemcpyDeviceToHost);

	check_answer(h_arr, h_mask, h_result, n, m);

	free(h_result);
	free(h_mask);
	free(h_arr);

	cudaFree(d_arr);
	cudaFree(d_res);

	printf("COMPLETED SUCCESFULLY\n");

	return 0;
}