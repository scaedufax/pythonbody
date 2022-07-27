import logging
import re
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys

from nbody.snap import snap
from nbody.cluster import cluster


# List of supported Versions of this Script
#SUPPORTED_VERSIONS = ["6++gpu"]

# List of data which can be loaded via the load(...) function
SUPPORTED_DATA_LOAD =  [
    "global",
    "sev",
    "RLAGR",
    "AVMASS",
    "VROT",
    "EKIN",
    "EPOT",
]

# List of Column names associated with a type of data
DATA_COLUMNS = {    
    "sev": ["TIME[NB]", "I", "NAME", "K*", "RI[RC]", "M[M*]", "log10(L[L*])", "log10(RS[R*])", "log10(Teff[K])"],
    "global": ["TIME[NB]", "TIME[Myr]", "TCR[Myr]", "DE", "BE(3)", "RSCALE[PC]", "RTIDE[PC]", "RDENS[PC]", "RC[PC]", "RHOD[M*/PC3]", "RHOM[M*/PC3]", "MC[M*]", "CMAX", "⟨Cn⟩", "Ir/R", "RCM[NB]", "VCM[NB]", "AZ", "EB/E", "EM/E", "VRMS[km/s]", "N", "NS", "NPAIRS", "NUPKS", "NPKS", "NMERGE", "MULT", "⟨NB⟩", "NC", "NESC", "NSTEPI", "NSTEPB", "NSTEPR", "NSTEPU", "NSTEPT", "NSTEPQ", "NSTEPC", "NBLOCK", "NBLCKR", "NNPRED", "NIRRF", "NBCORR", "NBFLUX", "NBFULL", "NBVOID", "NICONV", "NLSMIN", "NBSMIN", "NBDIS", "NBDIS2", "NCMDER", "NFAST", "NBFAST", "NKSTRY", "NKSREG", "NKSHYP", "NKSPER", "NKSMOD", "NTTRY", "NTRIP", "NQUAD", "NCHAIN", "NMERG", "NEWHI"]
}

# data files that can be loaded, matched with regex expression to find
# corresponding data files
SUPPORTED_DATA_FILES = {
    "global": "global\.30",
    "sev": "sev\.",
    "inp_files": "\.inp",
    "out_files": "\.out",
    "nbody_stdout_files": "JuSt NeVer m4Tch anYth!ng!",
}

class VersionNotSupported(Exception):
    pass
class DataImportNotSupported(Exception):
    pass

class nbody:
    """
    Class for handling nbody results
    
    Attributes:
        version (str): nbody version used for the simulation
        data_path (str): Path where output files of nbody can be found
        job_id (list): Not required, but helps finding the correct (main out) file(s)
        
        _data (dict): contains data from nbody simulation
        _files (dict): contains files related to the simulation
    
    """
    def __init__(self, data_path: str = None, nb_stdout_files: list = None):
        """
        Initializes class with data
        
        Parameters:
            data_path (str): Path where output files of nbody can be found
            nbody_stdout_files (list): list of file from nbody stdout
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
         
        self.data_path = data_path
        # make sure self.data_path ends with "/"
        if (self.data_path and 
            self.data_path[len(self.data_path) -1 ] != "/"):            
            self.data_path += "/"

        self.snap = snap(self.data_path)
        self.cluster = cluster(self.data_path, s=self.snap)
         
        self._data = {data_type: None for data_type in SUPPORTED_DATA_LOAD}
        self._files = { data_type: [] for data_type in SUPPORTED_DATA_FILES }

        self._files["nb_stdout_files"] = nb_stdout_files if type(nb_stdout_files) == list else [nb_stdout_files]
        
        self._analyze_files()
        
    def __setitem__(self, key, item):
        self._data[key] = item

    def __getitem__(self, key):
        return self._data[key]

    def __repr__(self):
        return repr(self._data)

    def __len__(self):
        return len(self._data)

    def __delitem__(self, key):
        del self._data[key]
        
    def load(self, what: list = "all"):
        """
        Loads data into the self._data from files associated with nbody.
        
        If a data type for loading is associated with a file containing more info then just the required
        one, all data associated with a filetype is loaded. E.g. if RLAGR is loaded, also AVMASS, VROT, ...
        will be loaded.
        
        Parameters:        
            what (list): Type of data the function is supposed to load.
                         See SUPPORTED_DATA_LOAD
        """
        
        # handle multiple datatype load
        if type(what) == list:
            # check whether everything passed can actually be imported
            for item in what:
                if item not in SUPPORTED_DATA_LOAD:
                    raise DataImportNotSupported(
                        "Can not import data type \"%s\". Available data type: %s" % (item, SUPPORTED_DATA_LOAD)
                    )
                    
            # if everything is fine with passed list, load data individually
            for item in what:
                self.load(item)      
        
        # handle all data to be imported
        elif what == "all":
            self.load(SUPPORTED_DATA_LOAD)
        
        # handle single data type load    
        elif type(what) == str:
            if what not in SUPPORTED_DATA_LOAD:
                    raise DataImportNotSupported(
                        "Can not import data type \"%s\". Available data type: %s" % (what, SUPPORTED_DATA_LOAD)
                    )
            self.logger.info("Loading data %s" % what)
            
            if what in ["RLAGR", "AVMASS", "VROT", "E", "main_stdout"]:
                out_files = self._files["nb_stdout_files"]
                rows = {"RLAGR": [], "AVMASS": [], "VROT": [], "EKIN": [], "EPOT": [], "ETIDE":[], "ETOT": [], "EBIN": [], "EMERGE": [], "ESUB": [], "ECOLL": [], "Q": []}
                cols = None
                for out_file in out_files:
                    with open(self.data_path + "/" + out_file, "r") as myfile:
                        for line in tqdm(myfile):
                            if re.search("TIME.*M/MT:", line):
                                line = re.sub("\s+", " ", line).strip()
                                line = line.split(" ")
                                cols = [float(i) for i in line[2:len(line)-1]] + [line[len(line)-1]]
                                cols = np.array(cols, dtype=str)        
                            elif re.search("RLAGR:|AVMASS:|VROT:", line):
                                line = re.sub("\s+", " ",line.replace("\n","")).strip()                            
                                if re.search("RLAGR", line):
                                    rows["RLAGR"] += [line.split(" ")]
                                elif re.search("AVMASS", line):
                                    rows["AVMASS"] += [line.split(" ")]
                                elif re.search("VROT", line):
                                    rows["VROT"] += [line.split(" ")]
                            elif re.search("ADJUST", line):
                                line = re.sub("\s+", " ", line).strip()
                                line = line.split(" ")
                                rows["EKIN"] += [[line[2], "EKIN", line[16]]]
                                rows["EPOT"] += [[line[2], "EPOT", line[18]]] 
                                rows["ETIDE"] += [[line[2], "ETIDE", line[20]]] 
                                rows["ETOT"] += [[line[2], "ETOT", line[22]]] 
                                rows["EBIN"] += [[line[2], "EBIN", line[24]]] 
                                rows["EMERGE"] += [[line[2], "EMERGE", line[26]]] 
                                rows["ESUB"] += [[line[2], "ESUB", line[26]]] 
                                rows["ECOLL"] += [[line[2], "ECOLL", line[26]]] 
                                rows["Q"] += [[line[2], "ECOLL", line[6]]] 
                                
                for key in rows.keys():
                    rows[key] = np.array(rows[key])
                #return rows["EKIN"],rows["EPOT"]
                self._data["RLAGR"] = pd.DataFrame(rows["RLAGR"][:,2:], 
                                                    index=np.array([i.replace("D","E") for i in rows["RLAGR"][:,0]], dtype=np.float64),
                                                    columns = cols,
                                                    dtype=np.float64)
                self._data["AVMASS"] = pd.DataFrame(rows["AVMASS"][:,2:],
                                                    index=np.array([i.replace("D","E") for i in rows["AVMASS"][:,0]], dtype=np.float64),
                                                    columns = cols,
                                                    dtype=np.float64)
                self._data["VROT"] = pd.DataFrame(rows["VROT"][:,2:],
                                                    index=np.array([i.replace("D","E") for i in rows["VROT"][:,0]], dtype=np.float64),
                                                    columns = cols,
                                                    dtype=np.float64)
                self._data["E"] = pd.DataFrame({"EKIN": rows["EKIN"][:,2], 
                                                "EPOT": rows["EPOT"][:,2],
                                                "ETIDE": rows["ETIDE"][:,2],
                                                "ETOT": rows["ETOT"][:,2],
                                                "EBIN": rows["EBIN"][:,2],
                                                "EMERGE": rows["EMERGE"][:,2],
                                                "ESUB": rows["ESUB"][:,2],
                                                "ECOLL": rows["ECOLL"][:,2],
                                                "Q": rows["Q"][:,2],
                                                },
                                                index=np.array([i.replace("D","E") for i in rows["EKIN"][:,0]], dtype=np.float64),
                                                    #columns = ["EKIN", "EPOT"],
                                                dtype=np.float64)                
                pass
            
            else:
                if len(self._files[what]) == 0:
                    self.logger.error("Couldn't load %s, files seem to be missing!" % what)
                    return -1
                dfs = []
                for file in tqdm(self._files[what]):
                    cols = DATA_COLUMNS[what] if what in DATA_COLUMNS else None
                    if cols is None:
                        df = pd.read_csv(file, delim_whitespace=True, skiprows=1, header=None)
                    else:                        
                        df = pd.read_csv(file, delim_whitespace=True, skiprows=1, header=None, names=cols)
                    dfs += [df]
                self._data[what] = pd.concat(dfs,ignore_index=True)
                del dfs

    def calculate_energy_evolution(self):
        if self.snap.shape == (0.3):
            self.snap._load_files()
        self._data["E"] = pd.DataFrame() 
        #self._data["E"] = pd.DataFrame() 

        for i in tqdm(self.snap.reduced.index):
            self.cluster.load(i)

            # TODO: change to pandas version
            if sys.version_info.minor >= 10:
                    self._data["E"].loc[i,["EPOT", "EKIN"]] = [self.cluster.EPOT, self.cluster.EKIN]
            else:
                self._data["E"].__init__(self._data["E"].append(pd.DataFrame(
                                                                 [[ self.cluster.EPOT,
                                                                    self.cluster.EKIN]],
                                                                 index=[i]
                                                            )))
        self._data["E"].sort_index(inplace=True)

        #self._data["EKIN"] = np.array(EKIN)
        #self._data["EPOT"] = np.array(EPOT)

    
    def show_mem_usage(self):
        """
        Shows memory usage of nbody class instance.
    
        The memory usage is shown for each data type, sorted from highest memory usage to lowest
        """
        mem_usage = []
        for key in self._data.keys():
            mem_usage += [(key, "%.03f M" % (sys.getsizeof(self._data[key])/1024/1024))]
        mem_usage = sorted(mem_usage, key=lambda item: item[1], reverse=True)
        print("Memory usage:")
        print("Total: %.03f M" % sum([float(i[1].replace(" M", " ")) for i in mem_usage]))
        for data in mem_usage:
            print("%s: %s" % data)
                        
            
    def _analyze_files(self):
        """
        Scans the nbody data dirctory for usable files. Called upon at the end of __init__()
        """
        files = sorted(glob.glob(self.data_path + "*"))
        self.logger.debug("Found data files %s", files)
        
        self.logger.info("Analyzing data files in %s", self.data_path)
        
        for file in tqdm(files):
            for file_type in SUPPORTED_DATA_FILES.keys():
                if re.search(SUPPORTED_DATA_FILES[file_type], file):
                    self._files[file_type] += [file]
                    break            
