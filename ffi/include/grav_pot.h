#ifndef GRAV_POT_H
#define GRAV_POT_H

#include <immintrin.h>

#if HAVE_CL_OPENCL_H == 1
#include "ocl.h"
extern cl_program ocl_program_grav_pot;
extern cl_kernel ocl_kernel_grav_pot;
int ocl_init_grav_pot(void);
void ocl_free_grav_pot(void);
#endif

void _grav_pot_inner_loop_avx(
		__m256 *M,
		__m256 *X1,
		__m256 *X2,
		__m256 *X3,
		__m256 M_i,
		__m256 X1_i,
		__m256 X2_i,
		__m256 X3_i,
		float *epot,
		int i,
		int n
		);

double grav_pot_omp(float *m, float *x1, float *x2, float *x3, float *EPOT, int n);
#if HAVE_CL_OPENCL_H == 1
double grav_pot_ocl(float *m, float *x1, float *x2, float *x3, float *EPOT, int n);
#endif
double grav_pot_unthreaded(float *m, float *x1, float *x2, float *x3, float *EPOT, int n);

#endif
