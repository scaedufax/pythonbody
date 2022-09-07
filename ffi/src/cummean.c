#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#if HAVE_CL_OPENCL_H == 1
#define CL_TARGET_OPENCL_VERSION 200
#include <CL/opencl.h>
#define CL_SUCCESS_OR_RETURN(code, where) do { \
    if (code != CL_SUCCESS) {printf("Err (%d): %s\n",code,where); return code; } \
}while (0);
#endif

#if HAVE_OMP_H == 1
double cummean_omp(double *target, double *source, int n) {
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
// OpenCL kernel. Each work item takes care of one element of c
const char *kernelSource =                                       "\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                    \n" \
"__kernel void grav_pot_kernel(  __global double *target,         \n" \
"                                __global double *src,            \n" \
"                                int n)                           \n" \
"{                                                                \n" \
"    //Get our global thread ID                                   \n" \
"    int id = get_global_id(0);                                   \n" \
"    int i  = id;                                                 \n" \
"    double mean = 0.0;                                           \n" \
"    //Make sure we do not go out of bounds                       \n" \
"    if (id < n) {                                                \n" \
"       for (int j = 0; j <= i; j++) {                            \n" \
"           mean += src[j];                                       \n" \
"       }                                                         \n" \
"       target[id] = mean/(i+1);                                     \n" \
"    }                                                            \n" \
"}\n\n";

//"       printf('%d, %f', id, mean);                               \n" \

double cummean_ocl(double *target,
                   double *src,
                   int n
                  )
{
    cl_mem l_src;
    cl_mem l_target;
    
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
		printf("Warning, need to use CPU as fallback device");
        err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    }
	
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
    l_src = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    l_target = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
    
    err = clEnqueueWriteBuffer(queue, l_src, CL_TRUE, 0,
                                   bytes, src, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, l_target, CL_TRUE, 0,
                                   bytes, target, 0, NULL, NULL);

	CL_SUCCESS_OR_RETURN(err, "clEngueueWriteBuffer");
    
    // Set the arguments to our compute kernel
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &l_target);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &l_src);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &n);
	
	CL_SUCCESS_OR_RETURN(err, "clSetKernelArg");
 
    // Execute the kernel over the entire range of the data set 
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
                                                              0, NULL, NULL);
	CL_SUCCESS_OR_RETURN(err, "clEnqueueNDRangeKernel");
 
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);
 
    // Read the results from the device
    clEnqueueReadBuffer(queue, l_target, CL_TRUE, 0,
                                bytes, target, 0, NULL, NULL );
 
    //Sum up vector c and print result divided by n, this should equal 1 within error
 
    // release OpenCL resources
    clReleaseMemObject(l_src);
    clReleaseMemObject(l_target);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

}
#endif

double cummean_unthreaded(double *target, double *source, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j <= i; j++) {
			target[i] += source[j];
		}
		target[i] = target[i]/(i+1);
	}
}
