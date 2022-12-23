#ifndef GRAV_POT_H
#define GRAV_POT_H

#if HAVE_CL_OPENCL_H == 1
#include "ocl.h"
extern cl_program ocl_program_neighbour_density;
extern cl_kernel ocl_kernel_neighbour_density;
int ocl_init_neighbour_density(void);
void ocl_free_neighbour_density(void);
#endif

int neighbour_density_omp(float *m,
	   					   float *x1,
						   float *x2,
						   float *x3,
						   float *neighbour_density_n,
						   float *neighbour_density_m,
						   int n_neigh,
						   int n_tot,
						   int *n_procs);
int neighbour_density_unthreaded(float *m,
	   					   float *x1,
						   float *x2,
						   float *x3,
						   float *neighbour_density_n,
						   float *neighbour_density_m,
						   int n_neigh,
						   int n_tot);
int neighbour_density_ocl(float *m,
	   					   float *x1,
						   float *x2,
						   float *x3,
						   float *neighbour_density_n,
						   float *neighbour_density_m,
						   int n_neigh,
						   int n_tot);
/*double grav_pot_ocl(float *m, float *x1, float *x2, float *x3, float *EPOT, int n);
double grav_pot_unthreaded(float *m, float *x1, float *x2, float *x3, float *EPOT, int n);*/

#endif
