#ifndef OCL_H
#define OCL_H

#if HAVE_CL_OPENCL_H == 1
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/opencl.h>
#include <stdio.h>
#define CL_SUCCESS_OR_RETURN(code, where) do { \
    if (code != CL_SUCCESS) {printf("Err (%d): %s\n",code,where); return code; } \
}while (0);

#define MAX_PLATFORMS 10
#define MAX_DEVICES 10

extern cl_platform_id ocl_platform_id[MAX_PLATFORMS];
extern cl_device_id ocl_device_id[MAX_DEVICES];
extern cl_context ocl_context;
extern cl_command_queue ocl_queue;
extern int ocl_below_20;

int ocl_init(int *p_id, int *d_id);
void ocl_free(void);

#endif
#endif
