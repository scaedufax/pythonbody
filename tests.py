import sys
import os
sys.path.insert(0, "../")
sys.path.insert(0, "./")
os.chdir("../")


import numpy as np
import pandas as pd
from ffi import ffi
#import timeit
import datetime as dt

N = 100000
np.random.seed(314159)

if __name__ == "__main__":
    df = pd.DataFrame({
        "M": np.random.rand(N),
        "X1": np.random.rand(N),
        "X2": np.random.rand(N),
        "X3": np.random.rand(N),
        })
    EPOT = {}
    #EPOT_c_funcs = ["unthreaded", "omp", "ocl", "ocl_cpu", "cuda"]
    EPOT_c_funcs = ["unthreaded", "omp"]
    RHO_N_c_funcs = ["unthreaded", "omp"]
    print("Testing grav_pot")
    for c_func in EPOT_c_funcs:
        reinit = False
        if c_func == "ocl_cpu":
            ffi._ocl_init(2,0)
            reinit = True
        try:
            s = dt.datetime.now()
            #eval(f"EPOT_{c_func} = None")
            EPOT[c_func] = eval(f"ffi.grav_pot(df, c_func=\"{c_func.replace('ocl_cpu','ocl')}\")")
            print(f"{c_func.replace('unthreaded','unthrd')}:\t{(dt.datetime.now() - s).microseconds/1000000 + (dt.datetime.now() - s).seconds}")
        except:
            continue
        if EPOT_c_funcs.index(c_func) != 0:
          np.testing.assert_allclose(EPOT[EPOT_c_funcs[EPOT_c_funcs.index(c_func) - 1]], EPOT[c_func],rtol=1e-4)
        if reinit:
            ffi._ocl_init()
    
    print()
    print("Testing cummean")
    CUMMEAN = {}
    for c_func in EPOT_c_funcs:
        try:
            s = dt.datetime.now()
            #eval(f"EPOT_{c_func} = None")
            CUMMEAN[c_func] = eval(f"ffi.cummean(np.array(df['M']), c_func='{c_func}')")
            print(f"{c_func.replace('unthreaded','unthrd')}:\t{(dt.datetime.now() - s).microseconds/1000000 + (dt.datetime.now() - s).seconds}")
        except:
            continue
        if EPOT_c_funcs.index(c_func) != 0:
          np.testing.assert_allclose(CUMMEAN[EPOT_c_funcs[EPOT_c_funcs.index(c_func) - 1]], CUMMEAN[c_func],rtol=1e-4)
    
    print()
    print("Testing RHO_N")
    RHO_N = {}
    for c_func in RHO_N_c_funcs:
        #try:
        s = dt.datetime.now()
        #eval(f"EPOT_{c_func} = None")
        RHO_N[c_func] = eval(f"ffi.neighbour_density(df[['M', 'X1', 'X2', 'X3']], c_func='{c_func}')")
        print(f"{c_func.replace('unthreaded','unthrd')}:\t{(dt.datetime.now() - s).microseconds/1000000 + (dt.datetime.now() - s).seconds}")
        #except:
        #    continue
        if RHO_N_c_funcs.index(c_func) != 0:
          np.testing.assert_allclose(RHO_N[RHO_N_c_funcs[RHO_N_c_funcs.index(c_func) - 1]], RHO_N[c_func],rtol=10e-2)
    print()
    print(RHO_N)
