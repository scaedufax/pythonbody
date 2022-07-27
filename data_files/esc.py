import pandas as pd
import pathlib
import os
import errno

COLS = ["TTOT", "BODY", "RI", "VI", "STEP T[Myr]", "M[M*]", "EESC", "VI[km/s]", "K*", "NAME", "ANGLE PHI", "ANGLE THETA", "M1[M*]", "RADIUS[RSun]", "LUM[LSun]", "TEFF", "AGE[Myr]", "EPOCH"]

REGEX = None

FILES = ["esc.11"]

def load(data_path="."):
    global COLS,REGEX,FILES
    if REGEX is not None:
        FILES = load_files()
    else:
        for f in FILES:
            if not pathlib.Path(data_path + "/" + f).is_file():
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), data_path + "/" + f)
    dfs = []
    for file in FILES:
        if COLS is not None:
            df = pd.read_csv(data_path + "/" + file,
                    delim_whitespace=True,
                    skiprows=1,
                    header=None,
                    names=COLS
                    )
        else :
            df =  df = pd.read_csv(data_path + "/" + file,
                    delim_whitespace=True,
                    skiprows=1,
                    header=None
                    )
        dfs += [df]
    return pd.concat(dfs,ignore_index=True)

