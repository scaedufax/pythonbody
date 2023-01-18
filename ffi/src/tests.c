#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../include/ocl.h"
#include "../include/grav_pot.h"
#include "../include/cummean.h"
#include "../include/neighbour_density.h"

#define N 1000
#define SEED 314159

int main (void) {
	float M[N];
	float X1[N];
	float X2[N];
	float X3[N];
	float target[N];
	float EPOT[N];

	#if HAVE_CL_OPENCL_H
	ocl_init(NULL,NULL);
	ocl_init_neighbour_density();
	ocl_init_grav_pot();
	ocl_init_cummean();
	#endif
	srand(SEED);

	

	for (int i = 0; i < N; i++) {
		M[i] = (float) rand()/RAND_MAX;
		X1[i] = (float) rand()/RAND_MAX;
		X2[i] = (float) rand()/RAND_MAX;
		X3[i] = (float) rand()/RAND_MAX;
		EPOT[i] = 0;
	}
	grav_pot_unthreaded(M,X1,X2,X3,EPOT,N);
	for (int i = 0; i < 9; i++) {
	    printf("%03.02f ", EPOT[i]);
	}
	printf(" ... ");
	for (int i = 9; i > 0; i--) {
	    printf("%03.02f ", EPOT[N-i]);
	}
	printf("\n");
	
	
	for (int i = 0; i < N; i++) {
		EPOT[i] = 0;
	}
	#if HAVE_OMP_H
	grav_pot_omp(M,X1,X2,X3,EPOT,N);
	for (int i = 0; i < 9; i++) {
	    printf("%03.02f ", EPOT[i]);
	}
	printf(" ... ");
	for (int i = 9; i > 0; i--) {
	    printf("%03.02f ", EPOT[N-i]);
	}
	printf("\n");
	#endif
	
	#if HAVE_CL_OPENCL_H
	grav_pot_ocl(M,X1,X2,X3,EPOT,N);

	ocl_free_grav_pot();
	ocl_free_cummean();
	ocl_free();
	#endif

	
	return 0;
}
