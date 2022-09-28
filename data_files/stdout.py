import pandas as pd
import pathlib
import os
import errno
import re
from tqdm import tqdm
import numpy as np
import multiprocessing as mp

COLS = None


FILES = None

data = {}
def analyze_line(line):
    global data
    cols = None

def load(stdout_files):
    #global COLS,REGEX,FILES
    FILES=stdout_files 

    data = {}
    for out_file in FILES:
        if not pathlib.Path(out_file).is_file():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), out_file)

        with open(out_file, "r") as myfile:
            cols = None
            lines = myfile.readlines()
            current_time = None
            line_count = 0
            RTIDE_line = None
            for line in tqdm(lines):
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
                    current_time = idx
                    data[which].loc[idx] = np.float64(line_data[2:])

                elif re.search("ADJUST", line):
                    line = re.sub("\s+", " ", line).strip()
                    line = line.split(" ")
                    if not "ADJUST" in data.keys():
                        cols = [line[i] for i in range(3, len(line), 2)]
                        data["ADJUST"] = pd.DataFrame(columns=cols)
                    idx = np.float64(line[2])
                    current_time = idx
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
                    cols = [line[i] for i in range(0, len(line), 2)]
                    if "OTHER" not in data.keys():
                        data["OTHER"] = pd.DataFrame(columns=cols)
                    idx = np.float64(line[1])
                    current_time = idx
                    data["OTHER"].loc[idx] = np.float64([line[i] for i in range(1, len(line), 2)])
                elif re.search("<R>.*RTIDE.*RDENS.*RC.*NC.*MC.*RHOD.*RHOM.*CMAX.*<Cn>.*Ir/R.*UN.*NP.*RCM.*VCM.*AZ.*EB/E.*EM/E.*TCR.*T6.*NESC.*VRMS", line):
                    line = re.sub("\s+", " ", line).strip()
                    line = line.split(" ")
                    cols = line
                    if "OTHER_R" not in data.keys():
                        data["OTHER_R"] = pd.DataFrame(columns=line)   
                    RTIDE_line = line_count + 1
                elif RTIDE_line == line_count:
                    line = re.sub("\s+", " ", line).strip()
                    line = line.split(" ")
                    data["OTHER_R"].loc[current_time] = np.float64(line[1:])
                line_count += 1
    return data

