import pandas as pd
import pathlib
import os
import errno
import re
from tqdm import tqdm
import numpy as np

COLS = None


FILES = None

def load(stdout_files):
    #global COLS,REGEX,FILES
    FILES=stdout_files 

    data = {}
    for out_file in FILES:
        if not pathlib.Path(out_file).is_file():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), out_file)

        with open(out_file, "r") as myfile:
            cols = None
            for line in tqdm(myfile):
                if re.search("TIME.*M/MT:", line):
                    line = re.sub("\s+", " ", line).strip()
                    line = line.split(" ")
                    cols = [float(i) for i in line[2:len(line)-1]] + [line[len(line)-1]]
                    cols = np.array(cols, dtype=str)
                elif re.search("RLAGR:|RSLAGR:|RBLAGR:|AVMASS:|NPARTC:|SIGR2:|SIGT2:|VROT:", line):
                    line = re.sub("\s+", " ",line.replace("\n","")).strip()
                    line_data = line.split(" ")

                    which = line_data[1]
                    which = which[:len(which) - 1]

                    if which not in data.keys():
                        # Some don't include < RC, needs to be filter with len(line_...)
                        data[which] = pd.DataFrame(columns=cols[:len(line_data[2:])])
                    idx = np.float64(line_data[0].replace("D", "E"))
                    data[which].loc[idx] = np.float64(line_data[2:])

                elif re.search("ADJUST", line):
                    line = re.sub("\s+", " ", line).strip()
                    line = line.split(" ")
                    if not "ADJUST" in data.keys():
                        cols = [line[i] for i in range(3, len(line), 2)]
                        data["ADJUST"] = pd.DataFrame(columns=cols)
                    idx = np.float64(line[2])
                    data["ADJUST"].loc[idx] = np.float64([line[i] for i in range(4, len(line), 2)])

                # the something after ADJUST lines, where we can get the total Mass
                elif re.search("TIME\[NB\].+N.+<NB>.+NPAIRS.+NMERGE.+MULT.+NS.+NSTEP\(I,B,R,U\).+DE.+E.+M.+", line):
                    line = re.sub("\s+", " ", line).strip()
                    line = line.split(" ")

                    # replace NSTEP(I,B,R,U) with NSTEP_I, NSTEP_B, ...
                    line[line.index("NSTEP(I,B,R,U)")] = "NSTEP_I"
                    line.insert(line.index("NSTEP_I") + 2, "NSTEP_B")
                    line.insert(line.index("NSTEP_B") + 2, "NSTEP_R")
                    line.insert(line.index("NSTEP_R") + 2, "NSTEP_U")
                    if not "OTHER" in data.keys():
                        cols = [line[i] for i in range(0, len(line), 2)]
                        data["OTHER"] = pd.DataFrame(columns=cols)
                    idx = np.float64(line[1])
                    data["OTHER"].loc[idx] = np.float64([line[i] for i in range(1, len(line), 2)])
    return data

