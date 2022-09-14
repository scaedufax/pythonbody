#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../include/ocl.h"
#include "../include/grav_pot.h"
#include "../include/cummean.h"

#define N 100000
#define SEED 314159

int main (void) {
	float M[N];
	float X1[N];
	float X2[N];
	float X3[N];
	float target[N];
	float EPOT[N];

	ocl_init(NULL,NULL);
	ocl_init_grav_pot();
	ocl_init_cummean();

	

	for (int i = 0; i < N; i++) {
		M[i] = (float) rand()/RAND_MAX;
		X1[i] = (float) rand()/RAND_MAX;
		X2[i] = (float) rand()/RAND_MAX;
		X3[i] = (float) rand()/RAND_MAX;
	}

	grav_pot_ocl(M,X1,X2,X3,EPOT,N);

	ocl_free_grav_pot();
	ocl_free_cummean();
	ocl_free();

	
	return 0;
}
