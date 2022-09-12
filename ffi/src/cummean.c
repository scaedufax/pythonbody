#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "ocl.h"
#include "cummean.h"

#if HAVE_OMP_H == 1
double cummean_omp(float *target, float *source, int n) {
	#pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                target[i] += source[j];
            }
            target[i] = target[i]/(i+1);
        }
    }
}
#endif

#if HAVE_CL_OPENCL_H == 1
cl_program ocl_program_cummean;
cl_kernel ocl_kernel_cummean;

// OpenCL kernel. Each work item takes care of one element of c
const char *kernel_source_cummean =                              "\n" \
"__kernel void cummean_kernel( __global float *target,            \n" \
"                              __global float *src,               \n" \
"                              int n)                             \n" \
"{                                                                \n" \
"    //Get our global thread ID                                   \n" \
"    int i = get_global_id(0);                                    \n" \
"    float mean = 0.0;                                           \n" \
"    //Make sure we do not go out of bounds                       \n" \
"    if (i < n) {                                                 \n" \
"       for (int j = 0; j <= i; j++) {                            \n" \
"           mean += src[j];                                       \n" \
"       }                                                         \n" \
"    target[i] = mean/(i+1);                                      \n" \
"    }                                                            \n" \
"}\n\n";

//"       printf('%d, %f', id, mean);                               \n" \
//"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                    \n" \

int ocl_init_cummean() {
	cl_int err;
    // Create the compute program from the source buffer
    ocl_program_cummean = clCreateProgramWithSource(ocl_context, 1,
                            (const char **) & kernel_source_cummean, NULL, &err);
	CL_SUCCESS_OR_RETURN(err, "clCreateProgramWithSource");
 
    // Build the program executable
    clBuildProgram(ocl_program_cummean, 0, NULL, NULL, NULL, NULL);

	/*char build_log[4096];
    err = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, (size_t) 4096, build_log, NULL);
	printf("%s", build_log);
	CL_SUCCESS_OR_RETURN(err, "clGetProgramBuildInfo");*/
 
    // Create the compute kernel in the program we wish to run
    ocl_kernel_cummean = clCreateKernel(ocl_program_cummean, "cummean_kernel", &err);
	CL_SUCCESS_OR_RETURN(err, "clCreateKernel");
	return CL_SUCCESS;

}

void ocl_free_cummean(void) {
    clReleaseProgram(ocl_program_cummean);
    clReleaseKernel(ocl_kernel_cummean);
}

double cummean_ocl(float *target,
                   float *src,
                   int n
                  )
{
    cl_mem l_src;
    cl_mem l_target;
    
	unsigned int N = (unsigned int) n;
    
    size_t bytes = n*sizeof(float);
    
    size_t globalSize, localSize;
    cl_int err;
 
    // Number of work items in each local work group
    localSize = 256;
 
    // Number of total work items - localSize must be devisor
    //globalSize = ceil(n/(float)localSize)*localSize;
    globalSize = (((int) (n/localSize)) + 1)*localSize;    
	 
 
    // Create the input and output arrays in device memory for our calculation
    l_src = clCreateBuffer(ocl_context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    l_target = clCreateBuffer(ocl_context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
    
    err = clEnqueueWriteBuffer(ocl_queue, l_src, CL_TRUE, 0,
                                   bytes, src, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(ocl_queue, l_target, CL_TRUE, 0,
                                   bytes, target, 0, NULL, NULL);

	CL_SUCCESS_OR_RETURN(err, "clEngueueWriteBuffer");
    
    // Set the arguments to our compute kernel
    err  = clSetKernelArg(ocl_kernel_cummean, 0, sizeof(cl_mem), &l_target);
    err |= clSetKernelArg(ocl_kernel_cummean, 1, sizeof(cl_mem), &l_src);
    err |= clSetKernelArg(ocl_kernel_cummean, 2, sizeof(int), &n);
	
	CL_SUCCESS_OR_RETURN(err, "clSetKernelArg");
 
    // Execute the kernel over the entire range of the data set 
    err = clEnqueueNDRangeKernel(ocl_queue, ocl_kernel_cummean, 1, NULL, &globalSize, NULL,
                                                              0, NULL, NULL);
	CL_SUCCESS_OR_RETURN(err, "clEnqueueNDRangeKernel");
 
    // Wait for the command queue to get serviced before reading back results
    clFinish(ocl_queue);
 
    // Read the results from the device
    clEnqueueReadBuffer(ocl_queue, l_target, CL_TRUE, 0,
                                bytes, target, 0, NULL, NULL );
 
    //Sum up vector c and print result divided by n, this should equal 1 within error
 
    // release OpenCL resources
    clReleaseMemObject(l_src);
    clReleaseMemObject(l_target);
}
#endif

double cummean_unthreaded(float *target, float *source, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j <= i; j++) {
			target[i] += source[j];
		}
		target[i] = target[i]/(i+1);
	}
}
