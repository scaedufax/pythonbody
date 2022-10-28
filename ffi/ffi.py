from ctypes import cdll, c_int, c_float, c_void_p, byref
import os
import numpy as np
import pandas as pd
import pathlib


class FFI:
    def __init__(self, p_id: int = None, d_id: int = None):
        cwd = os.path.realpath(__file__)
        cwd = cwd[:cwd.rfind("/") + 1]
        if pathlib.Path("pythonbody/ffi/.libs/libpythonbody.so").is_file():
            self.lib = cdll.LoadLibrary(
                    "pythonbody/ffi/.libs/libpythonbody.so"
                    )
        elif pathlib.Path("ffi/.libs/libpythonbody.so").is_file():
            self.lib = cdll.LoadLibrary("ffi/.libs/libpythonbody.so")
        elif pathlib.Path(".libs/libpythonbody.so").is_file():
            self.lib = cdll.LoadLibrary(".libs/libpythonbody.so")
        elif pathlib.Path(cwd + ".libs/libpythonbody.so").is_file():
            self.lib = cdll.LoadLibrary(cwd + ".libs/libpythonbody.so")
        else:
            raise Exception("Couldn't load libpythonbody.so!")
        try:
            self._ocl_init(p_id, d_id)
            self.default_cfunc = "ocl"
            self.OpenCL = True
        except:
            print("FFI: Couldn't load OpenCL, defaulting back to OpenMP")
            self.default_cfunc = "omp"
        

    def __del__(self):
        if self.OpenCL:
            self._ocl_free_grav_pot()
            self._ocl_free_cummean()
            self._ocl_free()

    def _ocl_init(self, p_id: int = None, d_id: int = None):
        func = self.lib.ocl_init
        func.argtypes = [c_void_p, c_void_p]
        func.restype = c_int

        err = func(c_void_p(None) if p_id is None else byref(c_int(p_id)),
                   c_void_p(None) if d_id is None else byref(c_int(d_id)))
        if err != 0:
            raise Exception("Error initialize OpenCL")
        self._ocl_init_cummean()
        self._ocl_init_grav_pot()

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

    def cummean(self, data: np.array, c_func="auto"):
        if c_func not in ["ocl","omp","unthreaded","auto", None]:
            raise ValueError("c_func must be either cuda, ocl, omp, unthreaded or None")
        if c_func == "auto":
            c_func = self.default_cfunc
        N = data.shape[0]
        target = (c_float * N)(*np.zeros(N))
        if c_func is not None:
            func = eval(f"self.lib.cummean_{c_func}")
        else:
            func = eval(f"self.lib.cummean")
        func.argtypes = [ 
                c_float * N, # target
                c_float * N, # source
                c_int,        # N 
                ]   
        func.restype = c_int 

        func(target,
                (c_float * N)(*data),
                N,  
            )   
        return np.array(target)
    def grav_pot(self,
                 data: pd.DataFrame,
                 G: float = 1,
                 num_threads: int = None,
                 c_func ="auto"
                 ):
        if c_func not in ["ocl","omp","unthreaded","auto", None]:
            raise ValueError("c_func must be either cuda, ocl, omp, unthreaded or None")
        if c_func == "auto":
            c_func = self.default_cfunc

        N = data.shape[0]

        EPOT = (c_float * N)(*np.zeros(N))
        if c_func is not None:
            func = eval(f"self.lib.grav_pot_{c_func}")
        else:
            func = eval(f"self.lib.grav_pot")
        
        func.argtypes = [
                c_float * N, # M
                c_float * N, # X1 
                c_float * N, # X2
                c_float * N, # X3
                c_float * N, # EPOT (results)
                c_int,        # N
                ]#c_int,        # num_threads
                #]
        func.restype = c_int 

        func((c_float * N)(*data["M"].values),
                    (c_float * N)(*data["X1"].values),
                    (c_float * N)(*data["X2"].values),
                    (c_float * N)(*data["X3"].values),
                    EPOT,
                    N,
                    num_threads
                    )
        return np.array(EPOT)

ffi = FFI()
