import pandas as pd
import pathlib
import os
import errno
import numpy as np
import re

from .data_file import DataFile as pbdf

"""COLS = ["TTOT",
        "BODY",
        "RI",
        "VI",
        "STEP",
        "T[Myr]", 
        "M[M*]", 
        "EESC", 
        "VI[km/s]",
        "K*",
        "NAME",
        "ANGLE PHI",
        "ANGLE THETA",
        "M1[M*]",
        "RADIUS[RSun]",
        "LUM[LSun]",
        "TEFF",
        "AGE[Myr]",
        "EPOCH"]"""
COLS = None

REGEX = None

FILES = ["esc.11"]

def load(data_path="."):
    global COLS,REGEX,FILES

    with open(data_path + "/" + FILES[0],"r") as data_file:
        header = data_file.readline().rstrip()
        # for newer versions, get rid of spaces
        header = header.replace("ANGLES PHI, THETA", "ANGLE_PHI ANGLE_THETA")
        # for older versions get rid of spaces
        header = header.replace("ANGLE PHI", "ANGLE_PHI")
        header = header.replace("ANGLE THETA", "ANGLE_THETA")

        # extract COLS from header
        header = re.sub("\s+", " ", header).strip()
        COLS = header.split(" ")

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
                    names=COLS,
                    index_col=False
                    )
        else :
            df = pd.read_csv(data_path + "/" + file,
                    delim_whitespace=True,
                    #skiprows=1,
                    #header=None
                    )
        dfs += [df]
    return pbdf("esc",pd.concat(dfs,ignore_index=True))

def calc_EESC_TOT(df: pbdf):
    arr = np.zeros(df.shape[0])
    for i in df.index:
        arr[i] = np.sum(df["EESC"][:i])
    return arr
