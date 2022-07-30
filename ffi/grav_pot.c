#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <pthread.h>

#include "grav_pot.h"

struct thread_args {
	int n;
	int num_threads;
	int thread_id;
	double *m;
	double *x1;
	double *x2;
	double *x3;
	double *EPOT;
};

double grav_pot_omp(double *m, double *x1, double *x2, double *x3, double *EPOT, int n, int num_threads) {
	#pragma omp parallel for
	for (int i = 0; i < n; i++) {
		for (int j = i+1; j < n; j++) {
			double dist = sqrt((x1[i] - x1[j])*(x1[i] - x1[j]) + (x2[i] - x2[j])*(x2[i] - x2[j]) + (x3[i] - x3[j])*(x3[i] - x3[j]));
			double epot_ij = -m[i]*m[j]/dist;
			EPOT[i] += epot_ij;
			EPOT[j] += epot_ij;
		}
	}
}

void *grav_pot_thread(void *args) {

	/* Loading arguments */
	struct thread_args *targs = (struct thread_args *) args;
	
	double *m = targs->m;
	double *x1 = targs->x1;
	double *x2 = targs->x2;
	double *x3 = targs->x3;
	double *EPOT = targs->EPOT;
	int n = targs->n;
	int num_threads = targs->num_threads;
	int thread_id =  targs->thread_id;
	/* do calculations */
	for (int i = thread_id; i < n; i = i + num_threads) {
		for (int j = i+1; j < n; j++) {
			double dist = sqrt((x1[i] - x1[j])*(x1[i] - x1[j]) + (x2[i] - x2[j])*(x2[i] - x2[j]) + (x3[i] - x3[j])*(x3[i] - x3[j]));
			double epot_ij = -m[i]*m[j]/dist;
			//printf("Thread %d, i=%d, j=%d\n", thread_id,i,j);
			//printf("i: m=%f x1=%f, x2=%f, x3=%f\n", m[i], x1[i], x2[i],x3[i]);
			//printf("j: m=%f x1=%f, x2=%f, x3=%f\n", m[j], x1[j], x2[j],x3[j]);
			EPOT[i] += epot_ij;
			EPOT[j] += epot_ij;
		}
	}
}

double grav_pot_threaded(double *m, double *x1, double *x2, double *x3, double *EPOT, int n, int num_threads) {
	
	/* Load thread_id for each thread */
	pthread_t *thread_ids = (pthread_t *) malloc(sizeof(pthread_t) * num_threads);
	/* Each thread shall get it's own list of args */
	struct thread_args *args_list = (struct thread_args *) malloc(sizeof(struct thread_args) * num_threads );
	
	/* In order to avoid race conditions assign */
	/* an array for EPOT to each thread, summing up later on */
	double *EPOT_threads = (double*) malloc(sizeof(double)*n*num_threads);
	for (int i = 0; i < n*num_threads;i++) {
		EPOT_threads[i] = 0;
	}
	
	/* start threads */
	for (int i = 0; i < num_threads; i++) {
		
		/* init arguments for each thread */
		struct thread_args *args = &args_list[i];
		args->m = m;
		args->x1 = x1;
		args->x2 = x2;
		args->x3 = x3;
		args->EPOT = &EPOT_threads[i*n]; /* own EPOT list */
		args->n = n;
		args->num_threads = num_threads;
		args->thread_id = i;

		pthread_create(&thread_ids[i], NULL, grav_pot_thread, args);
	}

	/* wait for threads to finish */
	for (int i = 0; i < num_threads; i++) {
		pthread_join(thread_ids[i], NULL);
	}

	/* after threads are finished, we can now sum up
	 * all values from the threads */
	for (int i = 0; i < num_threads; i++) {
		for (int j = 0; j < n; j++) {
			EPOT[j] += EPOT_threads[i*n+j];
		}
	}

	/* free allocated memory */
	free(args_list);
	free(thread_ids);
	free(EPOT_threads);
	
	/* calculate total energy */
	/*double epot_tot = 0;
	for (int i = 0; i < n; i++) {
		epot_tot += EPOT[i];
	}
	return epot_tot;	*/

}
double grav_pot_unthreaded(double *m, double *x1, double *x2, double *x3, double *EPOT, int n, int num_threads) {
	for (int i = 0; i < n; i++) {
		for (int j = i+1; j < n; j++) {
			double dist = sqrt((x1[i] - x1[j])*(x1[i] - x1[j]) + (x2[i] - x2[j])*(x2[i] - x2[j]) + (x3[i] - x3[j])*(x3[i] - x3[j]));
			double epot_ij = -m[i]*m[j]/dist;
			EPOT[i] += epot_ij;
			EPOT[j] += epot_ij;
		}
	}
}


