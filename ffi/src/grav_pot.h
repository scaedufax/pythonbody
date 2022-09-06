double grav_pot(double *m, double *x1, double *x2, double *x3, double *EPOT, int n, int num_threads);
double grav_pot_threaded(double *m, double *x1, double *x2, double *x3, double *EPOT, int n, int num_threads);
void * grav_pot_thread(void * args);
