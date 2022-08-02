#include <iostream>
#include <math.h>

#include <stdlib.h>

// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
	//y[i] = x[i] + y[i];
	for (int j = 0; j < n; j++) {
		y[i] += x[i]*x[j];
	}
}

int main(void)
{
  int N = 1000000;
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  time_t t;
  srand((unsigned) time(&t));
  for (int i = 0; i < N; i++) {
    x[i] = (float) rand()/((float) RAND_MAX);
  }

  // Run kernel on 1M elements on the GPU
  add<<<8, 128>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float ETOT = 0.0f;
  for (int i = 0; i < N; i++)
    ETOT += y[i];
  std::cout << "ETOT: " << ETOT << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}
