#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include "ocl.h"
#include "neighbour_density.h"

#if HAVE_OMP_H == 1
void neighbour_density_omp(float *m,
							 float *x1,
							 float *x2,
							 float *x3,
							 float *neighbour_density_n,
							 float *neighbour_density_m,
							 int n_neigh,
							 int n_tot,
							 int *n_procs) {
	if (n_procs != NULL) {
		omp_set_dynamic(0);
		omp_set_num_threads(*n_procs);

	}
	#pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < n_tot; i++) {
			float *dist_list = (float *) malloc(sizeof(float) * n_neigh);
			int *dist_idx_list = (int *) malloc(sizeof(int) * n_neigh);

			/* initialize lists with -1 */
			for (int j = 0; j < n_neigh; j++) {
				dist_list[j] = -1;
				dist_idx_list[j] = -1;
			}
			/* sort all other stars into the neighbour list */
            for (int j = 0; j < n_tot; j++) {
				/* skip self distance */
				if (i == j) {
					continue;
				}
				float dist = (x1[i] - x1[j]) * (x1[i] - x1[j]) + (x2[i] - x2[j]) * (x2[i] - x2[j]) + (x3[i] - x3[j]) * (x3[i] - x3[j]);
				//printf("Star %d, looking at neighbour %d with distance %f\n",i,j,dist);
				
				/* continue if distance is too large! */
				if ((dist_list[n_neigh - 1] != -1) && (dist > dist_list[n_neigh - 1])) {
					
					//printf("Star %d: Dropping star %d with dist %f", i,j,dist);
					continue;
				}
				else if ((dist_list[0] != -1 ) && (dist < dist_list[0])) {
					for (int k = n_neigh - 1; k > 0; k--) {
						dist_list[k] = dist_list[k-1];
						dist_idx_list[k] = dist_idx_list[k-1];
					}
					dist_list[0] = dist;
					dist_idx_list[0] = j;
					//printf("Star %d: inserting neighbour %d at beginning with distance %f\n",i,j,dist);
					continue;
				}

				for (int k = 0; k < n_neigh - 1; k++) {
					if ((dist_list[k] == -1) && (dist_idx_list[k] == -1)) {
						dist_list[k] = dist;
						dist_idx_list[k] = j;
						//printf("Star %d: inserting neighbour %d as number %d inserting distance %f (Due to -1 found)\n",i,j,k,dist);
						break;
					}
					else if ((dist_list[k] < dist) && ((dist_list[k+1] > dist) || dist_list[k+1] == -1.0)) {
						/* Found position in index, now moving up */
						for (int l = n_neigh - 1; l > k + 1; l--) {
							dist_list[l] = dist_list[l-1];
							dist_idx_list[l] = dist_idx_list[l-1];
						}
						/* Insert positions */
						dist_list[k+1] = dist;
						dist_idx_list[k+1] = j;
						// printf("Star %d: inserting neightbour %d as number %d inserting distance %f (Found proper position)\n",i,j,k,dist);
						break;
					}
				}
            }
			/* calculate averages */
			float avg_dist = 0;
			float avg_mass = 0;
			for (int j = 0; j < n_neigh; j++) {
				avg_dist += dist_list[j]/n_neigh;
				avg_mass += m[dist_idx_list[j]]/n_neigh;
			}
			/* store averages */
			neighbour_density_n[i] = 1/(4./3.*3.14159*avg_dist*avg_dist*avg_dist);
			neighbour_density_m[i] = avg_mass/(4./3.*3.14159*avg_dist*avg_dist*avg_dist);
			
			/* free memory */
			free(dist_list);
			free(dist_idx_list);
        }
    }
}
#endif

void neighbour_density_unthreaded(float *m,
	    					 	  float *x1,
								  float *x2,
							 	  float *x3,
							 	  float *neighbour_density_n,
								  float *neighbour_density_m,
							 	  int n_neigh,
							 	  int n_tot
							 	 ) 
{
	for (int i = 0; i < n_tot; i++) {
		float *dist_list = (float *) malloc(sizeof(float) * n_neigh);
		int *dist_idx_list = (int *) malloc(sizeof(int) * n_neigh);

		/* initialize lists with -1 */
		for (int j = 0; j < n_neigh; j++) {
			dist_list[j] = -1;
			dist_idx_list[j] = -1;
		}
		/* sort all other stars into the neighbour list */
		for (int j = 0; j < n_tot; j++) {
			/* skip self distance */
			if (i == j) {
				continue;
			}
			float dist = (x1[i] - x1[j]) * (x1[i] - x1[j]) + (x2[i] - x2[j]) * (x2[i] - x2[j]) + (x3[i] - x3[j]) * (x3[i] - x3[j]);
			//printf("Star %d, looking at neighbour %d with distance %f\n",i,j,dist);
			
			/* continue if distance is too large! */
			if ((dist_list[n_neigh - 1] != -1) && (dist > dist_list[n_neigh - 1])) {
				
				//printf("Star %d: Dropping star %d with dist %f", i,j,dist);
				continue;
			}
			else if ((dist_list[0] != -1 ) && (dist < dist_list[0])) {
				for (int k = n_neigh - 1; k > 0; k--) {
					dist_list[k] = dist_list[k-1];
					dist_idx_list[k] = dist_idx_list[k-1];
				}
				dist_list[0] = dist;
				dist_idx_list[0] = j;
				//printf("Star %d: inserting neighbour %d at beginning with distance %f\n",i,j,dist);
				continue;
			}

			for (int k = 0; k < n_neigh - 1; k++) {
				if ((dist_list[k] == -1) && (dist_idx_list[k] == -1)) {
					dist_list[k] = dist;
					dist_idx_list[k] = j;
					//printf("Star %d: inserting neighbour %d as number %d inserting distance %f (Due to -1 found)\n",i,j,k,dist);
					break;
				}
				else if ((dist_list[k] < dist) && ((dist_list[k+1] > dist) || dist_list[k+1] == -1.0)) {
					/* Found position in index, now moving up */
					for (int l = n_neigh - 1; l > k + 1; l--) {
						dist_list[l] = dist_list[l-1];
						dist_idx_list[l] = dist_idx_list[l-1];
					}
					/* Insert positions */
					dist_list[k+1] = dist;
					dist_idx_list[k+1] = j;
					// printf("Star %d: inserting neightbour %d as number %d inserting distance %f (Found proper position)\n",i,j,k,dist);
					break;
				}
			}
		}
		/* calculate averages */
		float avg_dist = 0;
		float avg_mass = 0;
		for (int j = 0; j < n_neigh; j++) {
			avg_dist += dist_list[j]/n_neigh;
			avg_mass += m[dist_idx_list[j]]/n_neigh;
		}
		/* store averages */
		neighbour_density_n[i] = 1/(4./3.*3.14159*avg_dist*avg_dist*avg_dist);
		neighbour_density_m[i] = avg_mass/(4./3.*3.14159*avg_dist*avg_dist*avg_dist);
		
		/* free memory */
		free(dist_list);
		free(dist_idx_list);
	}
}
