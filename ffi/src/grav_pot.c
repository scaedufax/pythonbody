#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>

#include "../include/grav_pot.h"

#if HAVE_CL_OPENCL_H == 1
#include "../include/ocl.h"
#endif

double grav_pot(float *m, float *x1, float *x2, float *x3, float *EPOT, int n) {
    #if HAVE_CL_OPENCL_H == 1
    return grav_pot_ocl(m,x1,x2,x3,EPOT,n);
    #elif HAVE_OMP_H == 1
    return grav_pot_omp(m,x1,x2,x3,EPOT,n);
    #else
    return grav_pot_unthreaded(m,x1,x2,x3,EPOT,n);
    #endif
}

#if HAVE_OMP_H == 1
double grav_pot_omp(float *m, float *x1, float *x2, float *x3, float *epot, int n) {
	__m256* M = (__m256*) m;
	__m256* X1 = (__m256*) x1;
	__m256* X2 = (__m256*) x2;
	__m256* X3 = (__m256*) x3;
	#pragma omp parallel
	{
		float epot_thread[n];
		for (int i = 0; i < n; i++) {
			epot_thread[i] = 0;
		}
		__m256* EPOT_thread = (__m256*) epot_thread;
		#pragma omp for
		for (int i = 0; i < n; i++) {
			__m256 X1_i = _mm256_set1_ps(x1[i]);
			__m256 X2_i = _mm256_set1_ps(x2[i]);
			__m256 X3_i = _mm256_set1_ps(x3[i]);
			__m256 M_i = _mm256_set1_ps(m[i]);
			for (int j = 0; j < (int) n/8; j++) {
				__m256 dist_x1 = _mm256_sub_ps(X1_i,X1[j]);
				__m256 dist_x2 = _mm256_sub_ps(X2_i,X2[j]);
				__m256 dist_x3 = _mm256_sub_ps(X3_i,X3[j]);
				dist_x1 = _mm256_mul_ps(dist_x1, dist_x1);
				dist_x2 = _mm256_mul_ps(dist_x2, dist_x2);
				dist_x3 = _mm256_mul_ps(dist_x3, dist_x3);
				__m256 dist = _mm256_add_ps(dist_x1, dist_x2);
				dist = _mm256_add_ps(dist, dist_x3);
				dist = _mm256_sqrt_ps(dist);

				__m256 epot_ij = _mm256_mul_ps(M_i, M[j]);
				epot_ij = _mm256_div_ps(epot_ij,dist);

				/* Disabling self gravity */			
				__m256 mask = _mm256_cmp_ps(dist, _mm256_set1_ps(0.0), _CMP_NEQ_OQ);
				epot_ij = _mm256_and_ps(epot_ij, mask);

				/* Summing all together */
				epot_ij = _mm256_hadd_ps(epot_ij, epot_ij);
				epot_ij = _mm256_hadd_ps(epot_ij, epot_ij);
				__m256 epot_ij_flip = _mm256_permute2f128_ps(epot_ij,epot_ij,1);
				epot_ij = _mm256_add_ps(epot_ij, epot_ij_flip);

				float res = epot_ij[0];
				epot[i] -= res;
			}
		}
		/*for (int i = 0; i < n; i++) {
			for (int j = i+1; j < n; j++) {
				float dist = sqrt((x1[i] - x1[j])*(x1[i] - x1[j]) + (x2[i] - x2[j])*(x2[i] - x2[j]) + (x3[i] - x3[j])*(x3[i] - x3[j]));
				float epot_ij = -m[i]*m[j]/dist;
				epot_thread[i] += epot_ij;
				epot_thread[j] += epot_ij;
			}
		}*/
		for (int i = 0; i < n; i++) {
			#pragma omp atomic
			epot[i] += epot_thread[i];
		}
	}
}
#endif
 
#if HAVE_CL_OPENCL_H == 1
cl_program ocl_program_grav_pot;
cl_kernel ocl_kernel_grav_pot;

const char *kernel_source_grav_pot =                             "\n" \
"__kernel void grav_pot_kernel(  __global float *m,               \n" \
"                                __global float *x1,              \n" \
"                                __global float *x2,              \n" \
"                                __global float *x3,              \n" \
"                                __global float *EPOT,            \n" \
"                                int n)                           \n" \
"{                                                                \n" \
"    //Get our global thread ID                                   \n" \
"    int id = get_global_id(0);                                   \n" \
"    int i  = id;                                                 \n" \
"    float EPOT_i = 0.0;                                          \n" \
"    //Make sure we do not go out of bounds                       \n" \
"    if (id < n) {                                                \n" \
"       for (int j = 0; j < n; j++) {                             \n" \
"		   if (i == j) continue;                                  \n" \
"           float dist = sqrt((x1[i] - x1[j])*(x1[i] - x1[j]) + (x2[i] - x2[j])*(x2[i] - x2[j]) + (x3[i] - x3[j])*(x3[i] - x3[j]));\n" \
"           EPOT_i +=  -m[i]*m[j]/dist;                           \n" \
"       }                                                         \n" \
"      EPOT[id] = EPOT_i;                                         \n" \
"    }                                                            \n" \
"}\n\n";
//"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                    \n" \

int ocl_init_grav_pot(void) {
	cl_int err;
    // Create the compute program from the source buffer
    ocl_program_grav_pot = clCreateProgramWithSource(ocl_context, 1,
                            (const char **) & kernel_source_grav_pot, NULL, &err);
    CL_SUCCESS_OR_RETURN(err, "clCreateProgramWithSource");

    // Build the program executable
    clBuildProgram(ocl_program_grav_pot, 0, NULL, NULL, NULL, NULL);

    /*char build_log[4096];
    err = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, (size_t) 4096, build_log, NULL);
    printf("%s", build_log);
    CL_SUCCESS_OR_RETURN(err, "clGetProgramBuildInfo");*/

    // Create the compute kernel in the program we wish to run
    ocl_kernel_grav_pot = clCreateKernel(ocl_program_grav_pot, "grav_pot_kernel", &err);
    CL_SUCCESS_OR_RETURN(err, "clCreateKernel");
    return CL_SUCCESS;
}
    
void ocl_free_grav_pot(void) {
    clReleaseProgram(ocl_program_grav_pot);
    clReleaseKernel(ocl_kernel_grav_pot);
}

double grav_pot_ocl(float *m,
                float *x1,
                float *x2,
                float *x3,
                float *EPOT,
                int n
                )
{
    cl_mem l_m;
    cl_mem l_x1;
    cl_mem l_x2;
    cl_mem l_x3;
    cl_mem l_EPOT;
    
	unsigned int N = (unsigned int) n;
    
    size_t bytes = n*sizeof(float);
    
    size_t globalSize, localSize;
    cl_int err;
 
    // Number of work items in each local work group
    localSize = 256;
 
    // Number of total work items - localSize must be devisor
    //globalSize = ceil(n/(float)localSize)*localSize;
    //globalSize = (((int) (n/localSize)) + 1)*localSize;
	globalSize = N;
     
    // Create the input and output arrays in device memory for our calculation
    l_m = clCreateBuffer(ocl_context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    l_x1 = clCreateBuffer(ocl_context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    l_x2 = clCreateBuffer(ocl_context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    l_x3 = clCreateBuffer(ocl_context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    l_EPOT = clCreateBuffer(ocl_context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
    
    err = clEnqueueWriteBuffer(ocl_queue, l_m, CL_TRUE, 0,
                                   bytes, m, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(ocl_queue, l_x1, CL_TRUE, 0,
                                   bytes, x1, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(ocl_queue, l_x2, CL_TRUE, 0,
                                   bytes, x2, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(ocl_queue, l_x3, CL_TRUE, 0,
                                   bytes, x3, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(ocl_queue, l_EPOT, CL_TRUE, 0,
                                   bytes, EPOT, 0, NULL, NULL);

	CL_SUCCESS_OR_RETURN(err, "clEngueueWriteBuffer");
    
    // Set the arguments to our compute kernel
    err  = clSetKernelArg(ocl_kernel_grav_pot, 0, sizeof(cl_mem), &l_m);
    err |= clSetKernelArg(ocl_kernel_grav_pot, 1, sizeof(cl_mem), &l_x1);
    err |= clSetKernelArg(ocl_kernel_grav_pot, 2, sizeof(cl_mem), &l_x2);
    err |= clSetKernelArg(ocl_kernel_grav_pot, 3, sizeof(cl_mem), &l_x3);
    err |= clSetKernelArg(ocl_kernel_grav_pot, 4, sizeof(cl_mem), &l_EPOT);
    err |= clSetKernelArg(ocl_kernel_grav_pot, 5, sizeof(int), &n);
	
	CL_SUCCESS_OR_RETURN(err, "clSetKernelArg");
 
    // Execute the kernel over the entire range of the data set 
    err = clEnqueueNDRangeKernel(ocl_queue, ocl_kernel_grav_pot, 1, NULL, &globalSize, NULL,
                                                              0, NULL, NULL);
	CL_SUCCESS_OR_RETURN(err, "clEnqueueNDRangeKernel");
 
    // Wait for the command queue to get serviced before reading back results
    clFinish(ocl_queue);
 
    // Read the results from the device
    clEnqueueReadBuffer(ocl_queue, l_EPOT, CL_TRUE, 0,
                                bytes, EPOT, 0, NULL, NULL );
 
    //Sum up vector c and print result divided by n, this should equal 1 within error
 
    // release OpenCL resources
    clReleaseMemObject(l_m);
    clReleaseMemObject(l_x1);
    clReleaseMemObject(l_x2);
    clReleaseMemObject(l_x3);
    clReleaseMemObject(l_EPOT);
}
#endif

double grav_pot_unthreaded(float *m, float *x1, float *x2, float *x3, float *epot, int n) {
	__m256* M = (__m256*) m;
	__m256* X1 = (__m256*) x1;
	__m256* X2 = (__m256*) x2;
	__m256* X3 = (__m256*) x3;
	__m256* EPOT = (__m256*) epot;

	for (int i = 0; i < n; i++) {
		__m256 X1_i = _mm256_set1_ps(x1[i]);
		__m256 X2_i = _mm256_set1_ps(x2[i]);
		__m256 X3_i = _mm256_set1_ps(x3[i]);
		__m256 M_i = _mm256_set1_ps(m[i]);
		for (int j = 0; j < (int) n/8; j++) {
			__m256 dist_x1 = _mm256_sub_ps(X1_i,X1[j]);
			__m256 dist_x2 = _mm256_sub_ps(X2_i,X2[j]);
			__m256 dist_x3 = _mm256_sub_ps(X3_i,X3[j]);
			dist_x1 = _mm256_mul_ps(dist_x1, dist_x1);
			dist_x2 = _mm256_mul_ps(dist_x2, dist_x2);
			dist_x3 = _mm256_mul_ps(dist_x3, dist_x3);
			__m256 dist = _mm256_add_ps(dist_x1, dist_x2);
			dist = _mm256_add_ps(dist, dist_x3);
			dist = _mm256_sqrt_ps(dist);

			__m256 epot_ij = _mm256_mul_ps(M_i, M[j]);
			epot_ij = _mm256_div_ps(epot_ij,dist);

			/* Disabling self gravity */			
			__m256 mask = _mm256_cmp_ps(dist, _mm256_set1_ps(0.0), _CMP_NEQ_OQ);
			epot_ij = _mm256_and_ps(epot_ij, mask);

			/* Summing all together */
			epot_ij = _mm256_hadd_ps(epot_ij, epot_ij);
			epot_ij = _mm256_hadd_ps(epot_ij, epot_ij);
			__m256 epot_ij_flip = _mm256_permute2f128_ps(epot_ij,epot_ij,1);
			epot_ij = _mm256_add_ps(epot_ij, epot_ij_flip);

			float res = epot_ij[0];
			epot[i] -= res;
		}
	}
}
