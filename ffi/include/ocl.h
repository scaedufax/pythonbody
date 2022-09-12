#ifndef OCL_H
#define OCL_H

#if HAVE_CL_OPENCL_H == 1
#define CL_TARGET_OPENCL_VERSION 200
#include <CL/opencl.h>
#include <stdio.h>
#define CL_SUCCESS_OR_RETURN(code, where) do { \
    if (code != CL_SUCCESS) {printf("Err (%d): %s\n",code,where); return code; } \
}while (0);

extern cl_platform_id ocl_platform_id;
extern cl_device_id ocl_device_id;
extern cl_context ocl_context;
extern cl_command_queue ocl_queue;

int ocl_init(void);
void ocl_free(void);

#endif
#endif
