#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "ocl.h"
#include "grav_pot.h"
#include "cummean.h"

#define N 100000
#define SEED 314159

int main (void) {
	double M[N];
	double X1[N];
	double X2[N];
	double X3[N];
	double target[N];

	ocl_init();
	ocl_init_grav_pot();
	ocl_init_cummean();

	

	for (int i = 0; i < N; i++) {
		M[i] = (float) rand()/RAND_MAX;
		X1[i] = (float) rand()/RAND_MAX;
		X2[i] = (float) rand()/RAND_MAX;
		X3[i] = (float) rand()/RAND_MAX;
	}
	
	return 0;
}
