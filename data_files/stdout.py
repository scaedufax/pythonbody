import pandas as pd
import pathlib
import os
import errno
import re
from tqdm import tqdm
import numpy as np

#COLS = ["TIME[NB]", "TIME[Myr]", "TCR[Myr]", "DE", "BE(3)", "RSCALE[PC]", "RTIDE[PC]", "RDENS[PC]", "RC[PC]", "RHOD[M*/PC3]", "RHOM[M*/PC3]", "MC[M*]", "CMAX", "⟨Cn⟩", "Ir/R", "RCM[NB]", "VCM[NB]", "AZ", "EB/E", "EM/E", "VRMS[km/s]", "N", "NS", "NPAIRS", "NUPKS", "NPKS", "NMERGE", "MULT", "⟨NB⟩", "NC", "NESC", "NSTEPI", "NSTEPB", "NSTEPR", "NSTEPU", "NSTEPT", "NSTEPQ", "NSTEPC", "NBLOCK", "NBLCKR", "NNPRED", "NIRRF", "NBCORR", "NBFLUX", "NBFULL", "NBVOID", "NICONV", "NLSMIN", "NBSMIN", "NBDIS", "NBDIS2", "NCMDER", "NFAST", "NBFAST", "NKSTRY", "NKSREG", "NKSHYP", "NKSPER", "NKSMOD", "NTTRY", "NTRIP", "NQUAD", "NCHAIN", "NMERG", "NEWHI"]

COLS = ["RLAGR", "AVMASS", "VROT", "EKIN", "EPOT", "ETOT", "Q"] 

#REGEX = None

#FILES = ["global.30"]

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
                elif re.search("RLAGR:|AVMASS:|VROT:", line):
                    line = re.sub("\s+", " ",line.replace("\n","")).strip()
                    line_data = line.split(" ")

                    which = line_data[1]
                    which = which[:len(which) - 1]

                    if which not in data.keys():
                        data[which] = pd.DataFrame(columns=cols)
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
    return data

