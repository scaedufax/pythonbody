#pragma OPENCL EXTENSION cl_khr_fp64 : enable                    
__kernel void grav_pot_kernel(  __global double *m,              
                                __global double *x1,             
                                __global double *x2,             
                                __global double *x3,             
                                __global double *EPOT,           
                                __global int *n)                 
{                                                               
    //Get our global thread ID                                  
    int id = get_global_id(0);                                  
    double EPOT_i = 0.0;                                                            
    //Make sure we do not go out of bounds                      
    if (id < n) {
       for (int i = 0; i < n, i++) {
           double dist = sqrt((x1[i] - x1[j])*(x1[i] - x1[j]) + (x2[i] - x2[j])*(x2[i] - x2[j]) + (x3[i] - x3[j])*(x3[i] - x3[j]));
           EPOT_i +=  -m[i]*m[j]/dist;
       }
      EPOT[ID] = EPOT_i; 
    }                                                           
}
