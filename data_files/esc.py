import pandas as pd
import pathlib
import os
import errno
import numpy as np
import re

from .data_file import DataFile as pbdf
"""
handle esc.11 output file, containing info about escapers.
"""

COLS = None

REGEX = None

FILES = ["esc.11"]

def load(data_path="."):
    global COLS, REGEX, FILES

    for f in FILES:
        if not pathlib.Path(data_path + "/" + f).is_file():
            raise FileNotFoundError(errno.ENOENT,
                                    os.strerror(errno.ENOENT),
                                    data_path + "/" + f)

    with open(data_path + "/" + FILES[0],"r") as data_file:
        header = data_file.readline().rstrip()
        # get rid of spaces
        header = header.replace("ANGLE PHI", "ANGLE_PHI")
        header = header.replace("ANGLE THETA", "ANGLE_THETA")
        header = header.replace("TIDAL(1-4)", "TIDAL1 TIDAL2 TIDAL3 TIDAL4")

        # extract COLS from header
        header = re.sub("\s+", " ", header).strip()
        COLS = header.split(" ")

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
        else:
            df = pd.read_csv(data_path + "/" + file,
                    delim_whitespace=True,
                    #skiprows=1,
                    #header=None
                    )
        dfs += [df]
    return pbdf("esc",pd.concat(dfs,ignore_index=True))

def calc_EESC_TOT(df: pbdf, eesc_col="EESC"):
    arr = df[eesc_col].cumsum().values
    return arr
