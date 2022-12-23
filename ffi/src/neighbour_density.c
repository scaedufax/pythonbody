#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include "ocl.h"
#include <CL/opencl.h>
#include "neighbour_density.h"

#if HAVE_OMP_H == 1
int neighbour_density_omp(float *m,
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

int neighbour_density_unthreaded(float *m,
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


#if HAVE_CL_OPENCL_H == 1
cl_program ocl_program_neighbour_density;
cl_kernel ocl_kernel_neighbour_density;

char *kernel_source_neighbour_density;
int ocl_init_neighbour_density(void) {
	cl_int err;
    // Create the compute program from the source buffer
	/*int kernel_source_allocated = 0;
	printf("2\n");
	if (kernel_source_neighbour_density == NULL) {
		FILE *f;
		f = fopen("/media/stuff_win/nbody/pythonbody/ffi/src/neighbour_density.cl", "r");
		if (!f) {
			printf("Error opening file!");
			exit(1);
		}
		fseek(f,0,SEEK_END);
		size_t source_size = ftell(f);
		printf("size of file %lu %lu %lu\n", source_size, source_size/sizeof(char), sizeof(char));

		kernel_source_neighbour_density = (char*) malloc(source_size + 1);
		kernel_source_neighbour_density[source_size] = "\0";
		fread(kernel_source_neighbour_density, sizeof(char), source_size, f);
		fclose(f);

		printf("%s", kernel_source_neighbour_density);
		
	}*/
    ocl_program_neighbour_density = clCreateProgramWithSource(ocl_context, 1,
                            (const char **) & kernel_source_neighbour_density, NULL, &err);
    CL_SUCCESS_OR_RETURN(err, "clCreateProgramWithSource");

    // Build the program executable
    clBuildProgram(ocl_program_neighbour_density, 0, NULL, NULL, NULL, NULL);

    /*char build_log[4096];
    err = clGetProgramBuildInfo(ocl_program_neighbour_density, *ocl_device_id, CL_PROGRAM_BUILD_LOG, (size_t) 4096, build_log, NULL);
    printf("%s", build_log);
    CL_SUCCESS_OR_RETURN(err, "clGetProgramBuildInfo");*/

    // Create the compute kernel in the program we wish to run
    ocl_kernel_neighbour_density = clCreateKernel(ocl_program_neighbour_density, "neighbour_density_kernel", &err);
    CL_SUCCESS_OR_RETURN(err, "clCreateKernel");
    return CL_SUCCESS;
}


    
void ocl_free_neighbour_density(void) {
    clReleaseProgram(ocl_program_neighbour_density);
    clReleaseKernel(ocl_kernel_neighbour_density);
}

int neighbour_density_ocl(float *m,
						   float *x1,
						   float *x2,
						   float *x3,
						   float *neighbour_density_n,
						   float *neighbour_density_m,
						   int n_neigh,
						   int n_tot
						  ) 
{
    cl_mem l_m;
    cl_mem l_x1;
    cl_mem l_x2;
    cl_mem l_x3;
    cl_mem l_nd_n;
    cl_mem l_nd_m;
    
	unsigned int N_TOT = (unsigned int) n_tot;
	unsigned int N_neigh = (unsigned int) n_neigh;
    
    size_t bytes = n_tot*sizeof(float);
    
    size_t globalSize, localSize;
    cl_int err;
 
    // Number of work items in each local work group
    localSize = 256;
 
    // Number of total work items - localSize must be devisor
    //globalSize = ceil(n/(float)localSize)*localSize;
    //globalSize = (((int) (n/localSize)) + 1)*localSize;
	globalSize = N_TOT;
     
    // Create the input and output arrays in device memory for our calculation
    l_m = clCreateBuffer(ocl_context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    l_x1 = clCreateBuffer(ocl_context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    l_x2 = clCreateBuffer(ocl_context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    l_x3 = clCreateBuffer(ocl_context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    l_nd_n = clCreateBuffer(ocl_context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
    l_nd_m = clCreateBuffer(ocl_context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
    
    err = clEnqueueWriteBuffer(ocl_queue, l_m, CL_TRUE, 0,
                                   bytes, m, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(ocl_queue, l_x1, CL_TRUE, 0,
                                   bytes, x1, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(ocl_queue, l_x2, CL_TRUE, 0,
                                   bytes, x2, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(ocl_queue, l_x3, CL_TRUE, 0,
                                   bytes, x3, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(ocl_queue, l_nd_n, CL_TRUE, 0,
                                   bytes, neighbour_density_n, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(ocl_queue, l_nd_m, CL_TRUE, 0,
                                   bytes, neighbour_density_m, 0, NULL, NULL);

	CL_SUCCESS_OR_RETURN(err, "clEngueueWriteBuffer");
    
    // Set the arguments to our compute kernel
    err  = clSetKernelArg(ocl_kernel_neighbour_density, 0, sizeof(cl_mem), &l_m);
    err |= clSetKernelArg(ocl_kernel_neighbour_density, 1, sizeof(cl_mem), &l_x1);
    err |= clSetKernelArg(ocl_kernel_neighbour_density, 2, sizeof(cl_mem), &l_x2);
    err |= clSetKernelArg(ocl_kernel_neighbour_density, 3, sizeof(cl_mem), &l_x3);
    err |= clSetKernelArg(ocl_kernel_neighbour_density, 4, sizeof(cl_mem), &l_nd_n);
    err |= clSetKernelArg(ocl_kernel_neighbour_density, 5, sizeof(cl_mem), &l_nd_m);
    err |= clSetKernelArg(ocl_kernel_neighbour_density, 6, sizeof(int), &n_neigh);
    err |= clSetKernelArg(ocl_kernel_neighbour_density, 7, sizeof(int), &n_tot);
	
	CL_SUCCESS_OR_RETURN(err, "clSetKernelArg");
 
    // Execute the kernel over the entire range of the data set 
    err = clEnqueueNDRangeKernel(ocl_queue, ocl_kernel_neighbour_density, 1, NULL, &globalSize, NULL,
                                                              0, NULL, NULL);
	CL_SUCCESS_OR_RETURN(err, "clEnqueueNDRangeKernel");
 
    // Wait for the command queue to get serviced before reading back results
    clFinish(ocl_queue);
 
    // Read the results from the device
    clEnqueueReadBuffer(ocl_queue, l_nd_n, CL_TRUE, 0,
                                bytes, neighbour_density_n, 0, NULL, NULL );
    clEnqueueReadBuffer(ocl_queue, l_nd_m, CL_TRUE, 0,
                                bytes, neighbour_density_m, 0, NULL, NULL );
 
    //Sum up vector c and print result divided by n, this should equal 1 within error
 
    // release OpenCL resources
    clReleaseMemObject(l_m);
    clReleaseMemObject(l_x1);
    clReleaseMemObject(l_x2);
    clReleaseMemObject(l_x3);
    clReleaseMemObject(l_nd_n);
    clReleaseMemObject(l_nd_m);
}

char *kernel_source_neighbour_density =                         "\n" \
"__kernel void neighbour_density_kernel(  __global float *m, \n" \
"                                __global float *x1, \n" \
"                                __global float *x2, \n" \
"                                __global float *x3, \n" \
"                                __global float *neighbour_density_n, \n" \
"                                __global float *neighbour_density_m, \n" \
"                                int n_neigh, \n" \
"                                int n_tot) \n" \
"{ \n" \
"    int i = get_global_id(0); \n" \
"	//float *dist_list = (float *) malloc(sizeof(float) * n_neigh); \n" \
"	//int *dist_idx_list = (int *) malloc(sizeof(int) * n_neigh); \n" \
"	float dist_list[80]; //= (float *) malloc(sizeof(float) * n_neigh); \n" \
"	int dist_idx_list[80]; //= (int *) malloc(sizeof(int) * n_neigh); \n" \
" \n" \
"	/* initialize lists with -1 */ \n" \
"	for (int j = 0; j < n_neigh; j++) { \n" \
"		dist_list[j] = -1; \n" \
"		dist_idx_list[j] = -1; \n" \
"	} \n" \
"	/* sort all other stars into the neighbour list */ \n" \
"	for (int j = 0; j < n_tot; j++) { \n" \
"		/* skip self distance */ \n" \
"		if (i == j) { \n" \
"			continue; \n" \
"		} \n" \
"		float dist = (x1[i] - x1[j]) * (x1[i] - x1[j]) + (x2[i] - x2[j]) * (x2[i] - x2[j]) + (x3[i] - x3[j]) * (x3[i] - x3[j]); \n" \
"		\n" \
"		/* continue if distance is too large! */ \n" \
"		if ((dist_list[n_neigh - 1] != -1) && (dist > dist_list[n_neigh - 1])) { \n" \
"			 \n" \
"			continue; \n" \
"		} \n" \
"		else if ((dist_list[0] != -1 ) && (dist < dist_list[0])) { \n" \
"			for (int k = n_neigh - 1; k > 0; k--) { \n" \
"				dist_list[k] = dist_list[k-1]; \n" \
"				dist_idx_list[k] = dist_idx_list[k-1]; \n" \
"			} \n" \
"			dist_list[0] = dist; \n" \
"			dist_idx_list[0] = j; \n" \
"			continue; \n" \
"		} \n" \
" \n" \
"		for (int k = 0; k < n_neigh - 1; k++) { \n" \
"			if ((dist_list[k] == -1) && (dist_idx_list[k] == -1)) { \n" \
"				dist_list[k] = dist; \n" \
"				dist_idx_list[k] = j; \n" \
"				break; \n" \
"			} \n" \
"			else if ((dist_list[k] < dist) && ((dist_list[k+1] > dist) || dist_list[k+1] == -1.0)) { \n" \
"				/* Found position in index, now moving up */ \n" \
"				for (int l = n_neigh - 1; l > k + 1; l--) { \n" \
"					dist_list[l] = dist_list[l-1]; \n" \
"					dist_idx_list[l] = dist_idx_list[l-1]; \n" \
"				} \n" \
"				/* Insert positions */ \n" \
"				dist_list[k+1] = dist; \n" \
"				dist_idx_list[k+1] = j; \n" \
"				break; \n" \
"			} \n" \
"		} \n" \
"	} \n" \
"	/* calculate averages */  \n" \
"	float avg_dist = 0; \n" \
"	float avg_mass = 0; \n" \
"	for (int j = 0; j < n_neigh; j++) { \n" \
"		avg_dist += dist_list[j]/n_neigh; \n" \
"		avg_mass += m[dist_idx_list[j]]/n_neigh; \n" \
"	} \n" \
"	/* store averages */ \n" \
"	neighbour_density_n[i] = 1/(4./3.*3.14159*avg_dist*avg_dist*avg_dist); \n" \
"	neighbour_density_m[i] = avg_mass/(4./3.*3.14159*avg_dist*avg_dist*avg_dist); \n" \
"	 \n" \
"	/* free memory */ \n" \
"	//free(dist_list); \n" \
"	//free(dist_idx_list); \n" \
"} \n\n" ;

#endif
