import pandas as pd
import pathlib
import os
import errno

AUTO_LOAD = True

COLS = ["TIME[NB]", "TIME[Myr]", "TCR[Myr]", "DE", "BE(3)", "RSCALE[PC]", "RTIDE[PC]", "RDENS[PC]", "RC[PC]", "RHOD[M*/PC3]", "RHOM[M*/PC3]", "MC[M*]", "CMAX", "⟨Cn⟩", "Ir/R", "RCM[NB]", "VCM[NB]", "AZ", "EB/E", "EM/E", "VRMS[km/s]", "N", "NS", "NPAIRS", "NUPKS", "NPKS", "NMERGE", "MULT", "⟨NB⟩", "NC", "NESC", "NSTEPI", "NSTEPB", "NSTEPR", "NSTEPU", "NSTEPT", "NSTEPQ", "NSTEPC", "NBLOCK", "NBLCKR", "NNPRED", "NIRRF", "NBCORR", "NBFLUX", "NBFULL", "NBVOID", "NICONV", "NLSMIN", "NBSMIN", "NBDIS", "NBDIS2", "NCMDER", "NFAST", "NBFAST", "NKSTRY", "NKSREG", "NKSHYP", "NKSPER", "NKSMOD", "NTTRY", "NTRIP", "NQUAD", "NCHAIN", "NMERG", "NEWHI"]

REGEX = None

FILES = ["global.30"]

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

