#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "grav_pot.h"

#if HAVE_CL_OPENCL_H == 1

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/opencl.h>
#endif

#define CL_SUCCESS_OR_RETURN(code, where) do { \
    if (code != CL_SUCCESS) {printf("Err (%d): %s\n",code,where); return code; } \
}while (0);

double grav_pot(double *m, double *x1, double *x2, double *x3, double *EPOT, int n) {
    #if HAVE_CL_OPENCL_H == 1
    return grav_pot_ocl(m,x1,x2,x3,EPOT,n);
    #elif HAVE_OMP_H == 1
    return grav_pot_omp(m,x1,x2,x3,EPOT,n);
    #else
    return grav_pot_omp(m,x1,x2,x3,EPOT,n);
    #endif
}

#if HAVE_OMP_H == 1
double grav_pot_omp(double *m, double *x1, double *x2, double *x3, double *EPOT, int n) {
	#pragma omp parallel
	{
		double EPOT_thread[n];
		for (int i = 0; i < n; i++) {
			EPOT_thread[i] = 0;
		}
		#pragma omp for
		for (int i = 0; i < n; i++) {
			for (int j = i+1; j < n; j++) {
				double dist = sqrt((x1[i] - x1[j])*(x1[i] - x1[j]) + (x2[i] - x2[j])*(x2[i] - x2[j]) + (x3[i] - x3[j])*(x3[i] - x3[j]));
				double epot_ij = -m[i]*m[j]/dist;
				EPOT_thread[i] += epot_ij;
				EPOT_thread[j] += epot_ij;
			}
		}
		for (int i = 0; i < n; i++) {
			#pragma omp atomic
			EPOT[i] += EPOT_thread[i];
		}
	}
}
#endif


 
#if HAVE_CL_OPENCL_H == 1
// OpenCL kernel. Each work item takes care of one element of c
const char *kernelSource =                                       "\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                    \n" \
"__kernel void grav_pot_kernel(  __global double *m,              \n" \
"                                __global double *x1,             \n" \
"                                __global double *x2,             \n" \
"                                __global double *x3,             \n" \
"                                __global double *EPOT,           \n" \
"                                int n)                           \n" \
"{                                                                \n" \
"    //Get our global thread ID                                   \n" \
"    int id = get_global_id(0);                                   \n" \
"    int i  = id;                                                 \n" \
"    double EPOT_i = 0.0;                                         \n" \
"    //Make sure we do not go out of bounds                       \n" \
"    if (id < n) {                                                \n" \
"       for (int j = 0; j < n; j++) {                             \n" \
"		   if (i == j) continue;                                  \n" \
"           double dist = sqrt((x1[i] - x1[j])*(x1[i] - x1[j]) + (x2[i] - x2[j])*(x2[i] - x2[j]) + (x3[i] - x3[j])*(x3[i] - x3[j]));\n" \
"           EPOT_i +=  -m[i]*m[j]/dist;                           \n" \
"       }                                                         \n" \
"      EPOT[id] = EPOT_i;                                         \n" \
"    }                                                            \n" \
"}\n\n";


double grav_pot_ocl(double *m,
                double *x1,
                double *x2,
                double *x3,
                double *EPOT,
                int n
                )
{
    cl_mem l_m;
    cl_mem l_x1;
    cl_mem l_x2;
    cl_mem l_x3;
    cl_mem l_EPOT;
    
    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_kernel kernel;                 // kernel
									  //
    //const char* kernelSource = Read_Source_File("./grav_pot_cl_kernel.c");

	unsigned int N = (unsigned int) n;
    
    size_t bytes = n*sizeof(double);
    
    size_t globalSize, localSize;
    cl_int err;
 
    // Number of work items in each local work group
    localSize = 256;
 
    // Number of total work items - localSize must be devisor
    //globalSize = ceil(n/(float)localSize)*localSize;
    globalSize = (((int) (n/localSize)) + 1)*localSize;
    
    // Bind to platform
    err = clGetPlatformIDs(1, &cpPlatform, NULL);
	
	CL_SUCCESS_OR_RETURN(err, "clGetPlatfromIDs");
 
    // Get ID for the device
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS) {
        err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    }

	/*char vendor[128];
	char name[128];
	clGetPlatformInfo(cpPlatform, CL_PLATFORM_VENDOR, 128*sizeof(char), vendor, NULL);
	clGetPlatformInfo(cpPlatform, CL_PLATFORM_NAME, 128*sizeof(char), name, NULL);
	printf("%s %s\n",vendor,name);*/
	
	CL_SUCCESS_OR_RETURN(err, "clGetDeviceIDs");
 
    // Create a context 
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
 
    // Create a command queue
    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
 
    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1,
                            (const char **) & kernelSource, NULL, &err);
	CL_SUCCESS_OR_RETURN(err, "clCreateProgramWithSource");
 
    // Build the program executable
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	/*char build_log[4096];
    err = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, (size_t) 4096, build_log, NULL);
	printf("%s", build_log);
	CL_SUCCESS_OR_RETURN(err, "clGetProgramBuildInfo");*/
 
    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "grav_pot_kernel", &err);
	CL_SUCCESS_OR_RETURN(err, "clCreateKernel");
 
    // Create the input and output arrays in device memory for our calculation
    l_m = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    l_x1 = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    l_x2 = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    l_x3 = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    l_EPOT = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
    
    err = clEnqueueWriteBuffer(queue, l_m, CL_TRUE, 0,
                                   bytes, m, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, l_x1, CL_TRUE, 0,
                                   bytes, x1, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, l_x2, CL_TRUE, 0,
                                   bytes, x2, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, l_x3, CL_TRUE, 0,
                                   bytes, x3, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, l_EPOT, CL_TRUE, 0,
                                   bytes, EPOT, 0, NULL, NULL);

	CL_SUCCESS_OR_RETURN(err, "clEngueueWriteBuffer");
    
    // Set the arguments to our compute kernel
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &l_m);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &l_x1);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &l_x2);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &l_x3);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &l_EPOT);
    err |= clSetKernelArg(kernel, 5, sizeof(int), &n);
	
	CL_SUCCESS_OR_RETURN(err, "clSetKernelArg");
 
    // Execute the kernel over the entire range of the data set 
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
                                                              0, NULL, NULL);
	CL_SUCCESS_OR_RETURN(err, "clEnqueueNDRangeKernel");
 
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);
 
    // Read the results from the device
    clEnqueueReadBuffer(queue, l_EPOT, CL_TRUE, 0,
                                bytes, EPOT, 0, NULL, NULL );
 
    //Sum up vector c and print result divided by n, this should equal 1 within error
 
    // release OpenCL resources
    clReleaseMemObject(l_m);
    clReleaseMemObject(l_x1);
    clReleaseMemObject(l_x2);
    clReleaseMemObject(l_x3);
    clReleaseMemObject(l_EPOT);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

}
#endif

double grav_pot_unthreaded(double *m, double *x1, double *x2, double *x3, double *EPOT, int n, int num_threads) {
	for (int i = 0; i < n; i++) {
		for (int j = i+1; j < n; j++) {
			double dist = sqrt((x1[i] - x1[j])*(x1[i] - x1[j]) + (x2[i] - x2[j])*(x2[i] - x2[j]) + (x3[i] - x3[j])*(x3[i] - x3[j]));
			double epot_ij = -m[i]*m[j]/dist;
			EPOT[i] += epot_ij;
			EPOT[j] += epot_ij;
		}
	}
}
