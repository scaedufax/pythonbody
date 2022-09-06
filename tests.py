import sys
import os
sys.path.insert(0, "../")
sys.path.insert(0, "./")
os.chdir("../")


import numpy as np
import pandas as pd
from utils.grav_pot import grav_pot
#import timeit
import datetime as dt

N = 100000

if __name__ == "__main__":
    df = pd.DataFrame({
        "M": np.random.rand(N),
        "X1": np.random.rand(N),
        "X2": np.random.rand(N),
        "X3": np.random.rand(N),
        })
    for c_func in ["ocl", "omp", "unthreaded"]:
        s = dt.datetime.now()
        #eval(f"EPOT_{c_func} = None")
        eval(f"grav_pot(df, c_func='{c_func}')")
        print(f"{c_func.replace('unthreaded','unthrd')}:\t{(dt.datetime.now() - s).microseconds + 1000000 * (dt.datetime.now() - s).seconds}")

