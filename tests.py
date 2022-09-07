import sys
import os
sys.path.insert(0, "../")
sys.path.insert(0, "./")
os.chdir("../")


import numpy as np
import pandas as pd
from utils import grav_pot,cummean
#import timeit
import datetime as dt

N = 10000
np.random.seed(314159)

if __name__ == "__main__":
    df = pd.DataFrame({
        "M": np.random.rand(N),
        "X1": np.random.rand(N),
        "X2": np.random.rand(N),
        "X3": np.random.rand(N),
        })
    EPOT = {}
    EPOT_c_funcs = [ "unthreaded", "omp", "ocl", "cuda"]
    print("Testing grav_pot")
    for c_func in EPOT_c_funcs:
        try:
            s = dt.datetime.now()
            #eval(f"EPOT_{c_func} = None")
            EPOT[c_func] = eval(f"grav_pot(df, c_func='{c_func}')")
            print(f"{c_func.replace('unthreaded','unthrd')}:\t{(dt.datetime.now() - s).microseconds/1000000 + (dt.datetime.now() - s).seconds}")
        except:
            continue
        if EPOT_c_funcs.index(c_func) != 0:
          np.testing.assert_allclose(EPOT[EPOT_c_funcs[EPOT_c_funcs.index(c_func) - 1]], EPOT[c_func])

    print()
    print("Testing cummean")

    CUMMEAN = {}
    for c_func in EPOT_c_funcs:
        try:
            s = dt.datetime.now()
            #eval(f"EPOT_{c_func} = None")
            CUMMEAN[c_func] = eval(f"cummean(np.array(df['M']), c_func='{c_func}')")
            print(f"{c_func.replace('unthreaded','unthrd')}:\t{(dt.datetime.now() - s).microseconds/1000000 + (dt.datetime.now() - s).seconds}")
        except:
            continue
        if EPOT_c_funcs.index(c_func) != 0:
          np.testing.assert_allclose(CUMMEAN[EPOT_c_funcs[EPOT_c_funcs.index(c_func) - 1]], CUMMEAN[c_func])
    print()
