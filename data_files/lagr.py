import pandas as pd
import pathlib
import os
import errno

from .data_file import DataFile as pbdf
"""
load lagr.7 files
"""

AUTO_LOAD = True

COLS = ["TIME", "0.001", "0.003", "0.005", "0.01" , "0.03" , "0.05" , "0.1"  ,
        "0.2"  ,"0.3"  , "0.4"  , "0.5"  , "0.6"  , "0.7"  , "0.8"  , "0.9"  , 
        "0.95" , "0.99" , "1."  ,"<RC"]

REGEX = None

FILES = ["lagr.7"]

COL_MAP = {
        "RLAGR": ([0] + list(range(1, 20)), COLS),
        "RLAGR_S": ([0] + list(range(20, 38)), COLS[:-1]),
        "RLAGR_B": ([0] + list(range(38, 56)), COLS[:-1]),
        "<M>": ([0] + list(range(56, 75)), COLS),
        "N_SHELL": ([0] + list(range(75, 94)), COLS),
        "<V_x>": ([0] + list(range(94, 113)), COLS),
        "<V_y>": ([0] + list(range(113, 132)), COLS),
        "<V_z>": ([0] + list(range(132, 151)), COLS),
        "<V>": ([0] + list(range(151, 170)), COLS),
        "<V_r>": ([0] + list(range(170, 189)), COLS),
        "<V_t>": ([0] + list(range(189, 208)), COLS),
        "SIG2": ([0] + list(range(208, 227)), COLS),
        "SIGR2": ([0] + list(range(227, 246)), COLS),
        "SIGT2": ([0] + list(range(246, 265)), COLS),
        "VROT": ([0] + list(range(265, 284)), COLS),
        }

def load(data_path="."):
    global COLS, REGEX, FILES

    data = {}
    
    if REGEX is None and FILES is not None:
        for f in FILES:
            if not pathlib.Path(data_path + "/" + f).is_file():
                raise FileNotFoundError(errno.ENOENT,
                                        os.strerror(errno.ENOENT),
                                        data_path + "/" + f)

    for key in COL_MAP.keys():
        data[key] = pd.read_csv(data_path + FILES[0],
                                header=0,
                                usecols=COL_MAP[key][0],
                                names=COL_MAP[key][1],
                                skiprows=1,
                                delim_whitespace=True,
                                index_col=0
                                )
    return data

