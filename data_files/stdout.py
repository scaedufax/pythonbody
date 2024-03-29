"""
Handle the nbody stdout file, and load (most of) its content.
"""
import pandas as pd
import pathlib
import os
import errno
import re
from tqdm import tqdm
import numpy as np
# import multiprocessing as mp

AUTO_LOAD = True

COLS = None


FILES = None

data = {}


"""def analyze_line(line):
    global data
    cols = None"""

def load(stdout_files, max_nb_time: float = None):
    """
    load the contents from stdout files

    :param stdout_files: Names of the stdout files
    :type stdout_files: list[str] or str
    :param max_nb_time: stop when NB time unit is reached.
    :type max_nb_time: float or None
    """

    if type(stdout_files) == str:
        stdout_files = [stdout_files]

    FILES = stdout_files 

    data = {}
    lines = []
    for out_file in FILES:
        if not pathlib.Path(out_file).is_file():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), out_file)

        """if len(FILES) != 1:
            print(f"Loading {out_file[out_file.rfind('/'):]} [{FILES.index(out_file) + 1}/{len(FILES)}]")"""

        with open(out_file, "r") as myfile:
            lines += myfile.readlines()

    cols = None
    current_time = -1
    line_count = 0
    RTIDE_line = None

    block_RLAGR = False
    block_ELLAN = False

    for line in tqdm(lines):
        if max_nb_time is not None and current_time > max_nb_time:
            break

        line = line.strip()

        # Skip KS-reg data for enhanced performance
        if (line.startswith("NEW KSREG") 
            or line.startswith("BEG KSRECT")
            or line.startswith("END KSREG")
            ):
            continue

        # LAGR Block
        elif re.search("TIME.*M/MT:", line):
            line = re.sub("\s+", " ", line).strip()
            line = line.split(" ")
            cols = [float(i) for i in line[2:len(line)-1]] + [line[len(line)-1]]
            cols = np.array(cols, dtype=str)
            block_RLAGR = True
            block_ELLAN = False
        elif block_RLAGR and re.search("RLAGR:|RSLAGR:|RBLAGR:|AVMASS:|NPARTC:|SIGR2:|SIGT2:|VROT:", line):
            line = re.sub("\s+", " ",line.replace("\n","")).strip()
            line_data = line.split(" ")

            # drop ":"
            which = line_data[1]
            which = which[:len(which) - 1]

            if which not in data.keys():
                # Some don't include < RC, needs to be filter with len(line_...)
                data[which] = pd.DataFrame(columns=cols[:len(line_data[2:])])
            idx = np.float64(line_data[0].replace("D", "E"))
            current_time = idx
            data[which].loc[idx] = np.float64(line_data[2:])
        
        # ELLAN Block
        elif re.search("TIME.*E/ET:", line):
            block_ELLAN = True
            block_RLAGR = False
            line = re.sub("\s+", " ", line).strip()
            line = line.split(" ")
            cols = [float(i) for i in line[2:]]
            cols = np.array(cols, dtype=str)
        elif block_ELLAN and re.search("MSHELL:|MCUM:|EROT:|EROTCM:|ANGM:|ANGMCM:|NPART:|NCUM:|AVMASS:|R3AV:|R2AV:|ZAV:|VROTEQ:|VRAV:|VZAV:|SGR2EQ:|SIGPH2:|SIGZ2:|B/A:|C/A:|TAU:|THETA[1-3]:|PHI[1-3]:", line):
            line = re.sub("\s+", " ",line.replace("\n","")).strip()
            line_data = line.split(" ")

            # drop ":"
            which = line_data[1]
            which = which[:len(which) - 1]

            if "ellan" not in data.keys():
                data["ellan"] = {}

            if which not in data["ellan"].keys():
                # Some don't include < RC, needs to be filter with len(line_...)
                data["ellan"][which] = pd.DataFrame(columns=cols[:len(line_data[2:])])
            idx = np.float64(line_data[0].replace("D", "E"))
            current_time = idx
            data["ellan"][which].loc[idx] = np.float64(line_data[2:])

        elif re.search("ADJUST", line):
            line = re.sub("\s+", " ", line).strip()
            line = line.split(" ")
            if not "ADJUST" in data.keys():
                cols = [line[i] for i in range(3, len(line), 2)]
                data["ADJUST"] = pd.DataFrame(columns=cols)
            idx = np.float64(line[2])
            current_time = idx
            try:
                data["ADJUST"].loc[idx] = np.float64([line[i] for i in range(4, len(line), 2)])
            except ValueError:
                continue

        # the something after ADJUST lines, where we can get the total Mass
        elif re.search("TIME\[NB\].+N.+<NB>.+NPAIRS.+NMERGE.+MULT.+NS.+NSTEP\(I,B,R,U\).+DE.+E.+M.+", line):
            block_RLAGR = False
            block_ELLAN = False
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
            try:
                data["OTHER_R"].loc[current_time] = np.float64(line[1:])
            except ValueError:
                continue
        # Scalar & Scalings blocks
        elif re.search("SCALING:.*SX =.*E =.*M\(1\) =.*M\(N\) =.*<M> =.*Q =.*", line):
            line = re.sub("\s+", " ", line).strip()
            line = line.split(" ")
            if "SCALARS" not in data.keys():
                data["SCALARS"] = {}
            if "SCALING" not in data["SCALARS"].keys():
                data["SCALARS"]["SCALING"] = {}

            for i, (key, value) in enumerate(zip(line[1::3], line[3::3])):
                data["SCALARS"]["SCALING"][key] = np.float64(value)
        elif re.search("TIME SCALES:.*TRH =.* TCR = .* 2<R>/<V> = .*", line):
            line = re.sub("\s+", " ", line).strip()
            line = line.split(" ")
            if "SCALARS" not in data.keys():
                data["SCALARS"] = {}
            if "TIME_SCALES" not in data["SCALARS"].keys():
                data["SCALARS"]["TIME_SCALES"] = {}

            for i, (key, value) in enumerate(zip(line[2::3], line[4::3])):
                data["SCALARS"]["TIME_SCALES"][key] = np.float64(value)
        elif re.search("PHYSICAL SCALING:.* R\* = .* M\* = .* V\* = .* T\* = .* <M> = .* SU = .*", line):
            line = re.sub("\s+", " ", line).strip()
            line = line.split(" ")
            if "SCALARS" not in data.keys():
                data["SCALARS"] = {}
            if "PHYSICAL_SCALING" not in data["SCALARS"].keys():
                data["SCALARS"]["PHYSICAL_SCALING"] = {}

            for i, (key, value) in enumerate(zip(line[2::3], line[4::3])):
                data["SCALARS"]["PHYSICAL_SCALING"][key] = np.float64(value)
        elif re.search("GR SCALING: .* V\* = .* C = .* RZ = .*", line):
            line = re.sub("\s+", " ", line).strip()
            line = line.split(" ")
            if "SCALARS" not in data.keys():
                data["SCALARS"] = {}
            if "GR_SCALING" not in data["SCALARS"].keys():
                data["SCALARS"]["GR_SCALING"] = {}

            for i, (key, value) in enumerate(zip(line[2::3], line[4::3])):
                data["SCALARS"]["GR_SCALING"][key] = np.float64(value)
        elif re.search("POINT-MASS MODEL \(NB unit\):.*MG = .* RG = .* OMEGA = .* RTIDE = .* RBAR = .*", line):
            line = re.sub("\s+", " ", line).strip()
            line = line.split(" ")
            if "SCALARS" not in data.keys():
                data["SCALARS"] = {}
            if "POINT-MASS_MODEL" not in data["SCALARS"].keys():
                data["SCALARS"]["POINT-MASS_MODEL"] = {}

            for i, (key, value) in enumerate(zip(line[4::3], line[6::3])):
                data["SCALARS"]["POINT-MASS_MODEL"][key] = np.float64(value)
        elif re.search("TIDAL PARAMETERS:.*TSCALE =.*  RTIDE =.*", line):
            line = re.sub("\s+", " ", line).strip()
            line = line.split(" ")
            if "SCALARS" not in data.keys():
                data["SCALARS"] = {}

            data["SCALARS"]["TIDAL"] = np.float64(line[2:6])
            data["SCALARS"]["TSCALE"] = np.float64(line[8])
            data["SCALARS"]["RTIDE"] = np.float64(line[11])

        line_count += 1
    return data

