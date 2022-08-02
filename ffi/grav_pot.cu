#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
//#include <omp.h>

//#include "grav_pot_cuda.h"

#define NTHREADS 256

// Kernel function to add the elements of two arrays
__global__
void grav_pot_kernel(double *m,
              double *x1,
              double *x2,
              double *x3,
              double *EPOT,
              int n)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
	  for (int j = 0; j< n; j++) {
		  if (i == j) continue;
          double dist = sqrt((x1[i] - x1[j])*(x1[i] - x1[j]) + (x2[i] - x2[j])*(x2[i] - x2[j]) + (x3[i] - x3[j])*(x3[i] - x3[j]));
          EPOT[i] += -m[i]*m[j]/dist;

	  }
  }
}
extern "C" void grav_pot(double *m,
              double *x1,
              double *x2,
              double *x3,
              double *EPOT,
              int n)
{
    double *l_m, *l_x1, *l_x2, *l_x3, *l_EPOT;
    cudaMallocManaged(&l_m, n*sizeof(double));
    cudaMallocManaged(&l_x1, n*sizeof(double));
    cudaMallocManaged(&l_x2, n*sizeof(double));
    cudaMallocManaged(&l_x3, n*sizeof(double));
    cudaMallocManaged(&l_EPOT, n*sizeof(double));

    //#pragma omp parallel for
    for (int i=0; i < n; i++) {
        l_m[i] = m[i];
        l_x1[i] = x1[i];
        l_x2[i] = x2[i];
        l_x3[i] = x3[i];
        l_EPOT[i] = 0.0;
    }
   
    int blocks = (int) n/NTHREADS + 1;
	//printf("Blocks: %d Threads: %d \n",blocks,NTHREADS);
    grav_pot_kernel<<<blocks, NTHREADS>>>(l_m, l_x1, l_x2, l_x3, l_EPOT, n);
    cudaDeviceSynchronize();
    
    //#pragma omp parallel for
    for (int i=0; i < n; i++) {
        EPOT[i] = l_EPOT[i];
    }

    cudaFree(l_m);
    cudaFree(l_x1);
    cudaFree(l_x2);
    cudaFree(l_x3);
    cudaFree(l_EPOT);

}

/*int main(void)
{
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  srand((unsigned int)time(NULL));
  for (int i = 0; i < N; i++) {
    x[i] = (float) rand()/((float)RAND_MAX);
    y[i] = 0.0f;
  }

  // Run kernel on 1M elements on the GPU
  int blocks = (int) N/NTHREADS + 1;
  printf("NBLOCKS: %d, NTHREADS: %d\n", blocks, NTHREADS);
  grav_pot<<<blocks, NTHREADS>>>(N, x, y);

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
}*/
