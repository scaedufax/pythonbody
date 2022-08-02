from ctypes import cdll, c_double, c_int
import pandas as pd
import numpy as np

def grav_pot(data: pd.DataFrame,
             G: float = 1,
             num_threads: int = None,
             c_func = "omp"
             ):
    if num_threads is None:
        from multiprocessing import cpu_count
        num_threads = cpu_count()
    N = data.shape[0]

    EPOT = (c_double * N)(*np.zeros(N))

    #lib = cdll.LoadLibrary("pythonbody/ffi/grav_pot.so")  
    lib = cdll.LoadLibrary("pythonbody/ffi/grav_pot_cuda.so")  
    #func = eval(f"lib.grav_pot_{c_func}")
    func = eval(f"lib.grav_pot")
    func.argtypes = [
            c_double * N, # M
            c_double * N, # X1 
            c_double * N, # X2
            c_double * N, # X3
            c_double * N, # EPOT (results)
            c_int,        # N
            ]#c_int,        # num_threads
            #]
    func.restype = c_int 

    func((c_double * N)(*data["M"].values),
                (c_double * N)(*data["X1"].values),
                (c_double * N)(*data["X2"].values),
                (c_double * N)(*data["X3"].values),
                EPOT,
                N,
                num_threads
                )
    return np.array(EPOT)

