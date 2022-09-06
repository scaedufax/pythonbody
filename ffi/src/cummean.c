#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double cummean(double *target, double *source, int n) {
    #if HAVE_OMP_H == 1
	#pragma omp parallel
    {
        #pragma omp for
        #endif
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                target[i] += source[j];
            }
            target[i] = target[i]/(i+1);
        }
    #if HAVE_OMP_H == 1
    }
    #endif
}

