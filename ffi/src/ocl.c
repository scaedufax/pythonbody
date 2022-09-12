#include "include/ocl.h"
#if HAVE_CL_OPENCL_H == 1

cl_platform_id ocl_platform_id;
cl_device_id ocl_device_id;
cl_context ocl_context;
cl_command_queue ocl_queue;

int ocl_init(void) {
	cl_int err;

	err = clGetPlatformIDs(1, &ocl_platform_id, NULL);
	CL_SUCCESS_OR_RETURN(err, "clGetPlatfromIDs");

	err = clGetDeviceIDs(ocl_platform_id, CL_DEVICE_TYPE_GPU, 1, &ocl_device_id, NULL);
    if (err != CL_SUCCESS) {
		printf("Warning, need to use CPU as fallback device");
        err = clGetDeviceIDs(ocl_platform_id, CL_DEVICE_TYPE_CPU, 1, &ocl_device_id, NULL);
    }
	CL_SUCCESS_OR_RETURN(err, "clGetDeviceIDs");

	ocl_context = clCreateContext(0, 1, &ocl_device_id, NULL, NULL, &err);
	CL_SUCCESS_OR_RETURN(err, "clCreateContext");

	ocl_queue = clCreateCommandQueueWithProperties(ocl_context, ocl_device_id, 0, &err);
	CL_SUCCESS_OR_RETURN(err, "clCreateQueue");

	return CL_SUCCESS;
}

void ocl_free(void) {
	clReleaseCommandQueue(ocl_queue);
    clReleaseContext(ocl_context);
}

#endif
