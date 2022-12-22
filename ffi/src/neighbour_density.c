#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
//#include "ocl.h"
#include "cummean.h"

//#if HAVE_OMP_H == 1
double neighbour_density_omp(float *m,
							 float *x1,
							 float *x2,
							 float *x3,
							 float *neighbour_density_n,
							 float *neighbour_density_m,
							 int n_neigh,
							 int n_tot) {
	#pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < n; i++) {
			float *dist_list = (float *) malloc(sizeof(float) * n_neigh);
			int *dist_idx_list = (int *) malloc(sizeof(int) * n_neigh);

			/* initialize lists with -1 */
			for (int j = 0; j < n_neigh) {
				dist_list[j] = -1;
				dist_idx_list = -1;
			}
            for (int j = 0; j < n; j++) {
				dist = (x1[i] - x1[j]) * (x1[i] - x1[j]) + (x2[i] - x2[j]) * (x2[i] - x2[j]) + (x3[i] - x3[j]) * (x3[i] - x3[j]);
				
				/* continue if distance is too large! */
				if (dist_list[n_neigh - 1] != -1) && (dist > dist_list[n_neigh - 1]) {
					continue
				}
				for (int k = 0; k < n_neigh - 1; k++) {
					if (dist_list[k] == -1 and dist_idx_list[k] == -1) {
						dist_list[k] = dist;
						dist_idx_list[k] = j;
						printf("Star %d: inserting neightbour %d as number %d inserting distance %f\n",i,j,k,dist);
						break;
					}
					else if (dist_list[k] < dist) && (dist_list[k+1] > dist) {
						for (int l = n_neigh - 1; l > k + 1; l--) {
							dist_list[l] = dist_list[l-1];
							dist_idx_list[l] = dist_idx_list[l-1]
						}
						dist_list[k+1] = dist;
						dist_idx_list[k+1] = j;
						printf("Star %d: inserting neightbour %d as number %d inserting distance %f\n",i,j,k,dist);
						break
					}	
				}
				for (int k = 0; k < n_neigh; k++) {
				}
            }
        }
    }
}
//#endif

