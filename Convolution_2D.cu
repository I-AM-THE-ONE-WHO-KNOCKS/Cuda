#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <iostream>

//8 X 8 mask
#define MASK_LEN 8

#define MASK_OFFSET (MASK_LEN / 2)

/*as mask is never changing we can define a constant memory on the device side so that
 we do not have to copu again and again and loading from const cache is much much faster that 
 loading from d-ram.
*/
__constant__ int mask[MASK_LEN * MASK_LEN];

__global__ void conv_2d(int* Mat, int* res, int n) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int start_r = row - MASK_OFFSET;
	int start_c = col - MASK_OFFSET;

	int temp = 0;

	for (int i = 0; i < MASK_LEN; i++)
	{
		for (int j = 0; j < MASK_LEN; j++)
		{
			if ((start_r + i >= 0) && (start_r + i < n))
			{
				if ((start_c + j >= 0) && (start_c + j < n))
				{
					temp += Mat[(start_r + i) * n + (start_c + j)] * mask[i * MASK_LEN + j];
				}
			}
		}
	}

	res[row * n + col] = temp;
}

// Initialize
void Array_init(int* a, int n, int div) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			a[i*n + j] = rand() % div;
		}
	}
}

void check_answer(int* mat, int* mask, int* res, int n) {
	int temp;
	int offset_r;
	int offset_c;

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			temp = 0;

			for (int k = 0; k < MASK_LEN; k++)
			{
				offset_r = i - MASK_OFFSET + k;

				for (int l = 0; l < MASK_LEN; l++)
				{
					offset_c = j - MASK_OFFSET + l;

					if (offset_r >= 0 && offset_r < n)
					{
						if (offset_c >= 0 && offset_c < n)
						{
							temp += mat[offset_r * n + offset_c] * mask[k * MASK_LEN + l];
						}
					}
				}
			}
			assert(res[i * n + j] == temp);
		}
	}
}

int main() {
	
	// number of elements in result array
	int n = 1 << 12;
	
	int n_bytes = n * n * sizeof(int);

	int m_bytes = MASK_LEN * MASK_LEN * sizeof(int);

	//allocate the array
	int* h_Mat = new int[n * n];

	Array_init(h_Mat, n, 100);

	//allocate the mask and intialize it
	int* h_mask = new int[MASK_LEN * MASK_LEN];
	Array_init(h_mask, MASK_LEN, 10);

	//allocate space for result
	int* h_result = new int[n * n];

	//allocate space on device memory
	int* d_Mat, * d_res;
	cudaMalloc(&d_Mat, n_bytes);
	cudaMalloc(&d_res, n_bytes);

	cudaMemcpy(d_Mat, h_Mat, n_bytes, cudaMemcpyHostToDevice);
	//special function to copy to a symbol
	cudaMemcpyToSymbol(mask, h_mask, m_bytes);

	int threads = 16;

	int grid = (n + threads - 1) / threads;

	dim3 block_dim(threads, threads);
	dim3 grid_dim(grid, grid);

	conv_2d <<<grid_dim, block_dim >>> (d_Mat, d_res, n);

	cudaMemcpy(h_result, d_res, n_bytes, cudaMemcpyDeviceToHost);

	check_answer(h_Mat, h_mask, h_result, n);

	free(h_result);
	free(h_mask);
	free(h_Mat);

	cudaFree(d_Mat);
	cudaFree(d_res);

	printf("COMPLETED SUCCESFULLY\n");

	return 0;
}