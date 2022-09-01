from ctypes import cdll, c_double, c_int
import pandas as pd
import numpy as np

def cummean(data: np.array):
    N = data.shape[0]
    target = (c_double * N)(*np.zeros(N))
    lib = cdll.LoadLibrary("pythonbody/ffi/cummean.so")
    func = lib.cummean
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

