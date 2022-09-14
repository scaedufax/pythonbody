#include "../include/ocl.h"
#if HAVE_CL_OPENCL_H == 1

cl_platform_id ocl_platform_id[MAX_PLATFORMS];
cl_device_id ocl_device_id[MAX_DEVICES];
cl_context ocl_context;
cl_command_queue ocl_queue;

int ocl_init(int *p_id, int *d_id) {
	cl_int err;
	int p_id_was_null = 0;
	int d_id_was_null = 0;
	if (p_id == NULL) {
		p_id = (int *) malloc(sizeof(int));
		*p_id = 0;
		p_id_was_null = 1;
	}
	if (d_id == NULL) {
		d_id = (int *) malloc(sizeof(int));
		*d_id = 0;
		d_id_was_null = 1;
	}
	for (int i = 0; (i < MAX_PLATFORMS) || (i < MAX_DEVICES); i++) {
		if ( i < MAX_PLATFORMS ) { ocl_platform_id[i] = 0;}
		if ( i < MAX_DEVICES ) { ocl_device_id[i] = 0;}
	}

	err = clGetPlatformIDs(MAX_PLATFORMS, ocl_platform_id, NULL);
	CL_SUCCESS_OR_RETURN(err, "clGetPlatfromIDs");

	err = clGetDeviceIDs(ocl_platform_id[*p_id], CL_DEVICE_TYPE_ALL, MAX_DEVICES, ocl_device_id, NULL);
    if (err != CL_SUCCESS) {
		printf("Warning, need to use CPU as fallback device\n");
        err = clGetDeviceIDs(ocl_platform_id[*p_id], CL_DEVICE_TYPE_CPU, MAX_DEVICES, ocl_device_id, NULL);
    }
	CL_SUCCESS_OR_RETURN(err, "clGetDeviceIDs");

	ocl_context = clCreateContext(0, 1, &ocl_device_id[*d_id], NULL, NULL, &err);
	CL_SUCCESS_OR_RETURN(err, "clCreateContext");

	ocl_queue = clCreateCommandQueueWithProperties(ocl_context, ocl_device_id[*d_id], 0, &err);
	CL_SUCCESS_OR_RETURN(err, "clCreateQueue");

	if (p_id_was_null) {	
		free(p_id);
	}
	if (d_id_was_null) {
		free(d_id);
	}

	return CL_SUCCESS;
}

void ocl_free(void) {
	clReleaseCommandQueue(ocl_queue);
    clReleaseContext(ocl_context);
}

#endif
