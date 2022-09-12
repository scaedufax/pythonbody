from ctypes import cdll, c_double, c_int
import numpy as np
import pandas as pd

class FFI:
    def __init__(self):
        self.lib = cdll.LoadLibrary("pythonbody/ffi/.libs/libpythonbody.so")
        self._ocl_init()
        self._ocl_init_cummean()
        self._ocl_init_grav_pot()
    def __del__(self):
        self._ocl_free_grav_pot()
        self._ocl_free_cummean()
        self._ocl_free()

    def _ocl_init(self):
        func = self.lib.ocl_init
        func.argtypes = []
        func.restype = c_int

        err = func()
        if err != 0:
            raise Exception("Error initialize OpenCL")
    def _ocl_init_cummean(self):
        func = self.lib.ocl_init_cummean
        func.restype = c_int

        err = func()
        if err != 0:
            raise Exception("Error initialize OpenCL")
    def _ocl_init_grav_pot(self):
        func = self.lib.ocl_init_grav_pot
        func.restype = c_int

        err = func()
        if err != 0:
            raise Exception("Error initialize OpenCL")

    def _ocl_free_grav_pot(self):
        func = self.lib.ocl_free_grav_pot
        func()
    def _ocl_free_cummean(self):
        func = self.lib.ocl_free_cummean
        func()
    def _ocl_free(self):
        func = self.lib.ocl_free
        func()

    def cummean(self, data: np.array, c_func="ocl"):
        if c_func not in ["ocl","omp","unthreaded", None]:
            raise ValueError("c_func must be either cuda, ocl, omp, unthreaded or None")
        N = data.shape[0]
        target = (c_double * N)(*np.zeros(N))
        if c_func is not None:
            func = eval(f"self.lib.cummean_{c_func}")
        else:
            func = eval(f"self.lib.cummean")
        func.argtypes = [ 
                c_double * N, # target
                c_double * N, # source
                c_int,        # N 
                ]   
        func.restype = c_int 

        func(target,
                (c_double * N)(*data),
                N,  
            )   
        return np.array(target)
    def grav_pot(self,
                 data: pd.DataFrame,
                 G: float = 1,
                 num_threads: int = None,
                 c_func = "ocl"
                 ):
        if c_func not in ["ocl","omp","unthreaded", None]:
            raise ValueError("c_func must be either cuda, ocl, omp, unthreaded or None")

        N = data.shape[0]

        EPOT = (c_double * N)(*np.zeros(N))
        if c_func is not None:
            func = eval(f"self.lib.grav_pot_{c_func}")
        else:
            func = eval(f"self.lib.grav_pot")
        
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

ffi = FFI()