#include <cuda_runtime.h>
#include <iostream>

using namespace std;

int main() {
	//Get the number of device in the system 
	int device_num;
	cudaGetDeviceCount(&device_num);
	cout << "There are " << device_num << " GPU's in this system" << endl;

	for (int i = 0; i < device_num; i++)
	{
		//set the device if we have multiple GPU's
		cudaSetDevice(i);

		//get device properties from GPU
		cudaDeviceProp device_prop;
		cudaGetDeviceProperties(&device_prop, i);
		cout << "Device " << i << " is a " << device_prop.name << endl;

		//get information about the driver and runtime
		int driver;
		int runtime;
		cudaDriverGetVersion(&driver);
		cudaRuntimeGetVersion(&runtime);
		cout << "Driver: " << driver << " Runtime: " << runtime << endl;

		// compare this against the device capabilities
		cout << "CUDA capability: " << device_prop.major << "." << device_prop.minor << endl;

		//can get the size of global memeory
		cout << "Global memory in GB: " << device_prop.totalGlobalMem / (1 << 30) << endl;

		//number of SMs
		cout << "Number of SMs: " << device_prop.multiProcessorCount << endl;

		//frequency
		cout << "Max clock rate: " << device_prop.clockRate * 1e-6 << "GHz" << endl;

		//The L2 cache size
		cout << "The L2 cache size in MB: " << device_prop.l2CacheSize / (1 << 20) << endl;

		//shared memory per block 
		cout << "Total shared memory per block in KB: " << device_prop.sharedMemPerBlock / (1 << 10) << endl;

		// similarly there are multiple other properties which can be checked.
		// The detailed list of properties which can be checked is listed here in the following link
		// https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/html/group__CUDART__DEVICE_g5aa4f47938af8276f08074d09b7d520c.html
	}

	return 0;
}