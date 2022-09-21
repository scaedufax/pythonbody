import pandas as pd
import numpy as np
import glob
import logging
import h5py
import sys
from tqdm import tqdm
import pathlib

from pythonbody.ffi import ffi
from pythonbody.nbdf import nbdf
from pythonbody.snap.binaries import Binaries
#from pythonbody.snap.singles import singles


class snap(pd.DataFrame):
    def __init__(self, data_path=None):
        super().__init__(columns=["time", "file", "step"])
        if not pathlib.Path(data_path).is_dir():
            raise IOError(f"Couldn't find {data_path}. Does it exist?")
        self.data_path = data_path

        self.files = None
        self.time = None
        if self.data_path is not None:
            self.files = sorted(glob.glob(self.data_path + "/snap*"))
            self._load_files()

        self.cluster_data = None
        self.binaries_data = None
        self.binaries_mask = None
        self.singles_mask = None
        self.time_evolution_data = None

    def __getitem__(self, value):
        """
        checks if passed value(s) are in currently loaded cluster data,
        otherwise returns snap list data
        """
        if type(value) != list:
            value = [value]

        missing_list = []
        if self.cluster_data is not None:
            for val in value:
                if val not in self.cluster_data.columns:
                    missing_list += [val]

            if len(missing_list) == 0:
                return self.cluster_data[value]
            elif len(missing_list) > 0 and np.sum([f"calc_{val}".replace("/", "_over_") not in dir(self) for val in missing_list]) == 0:
                for missing in missing_list:
                    if missing not in self.cluster_data.columns:
                        eval(f"self.calc_{missing}()".replace("/", "_over_"))
                return self.cluster_data[value]
            else:
                return super().__getitem__(value)
        else:
            return super().__getitem__(value)

    """def __repr__(self):
        if self.cluster_data is not None:
            return self.cluster_data.__repr__()
        else:
            super().__repr__()"""

    @property
    def reduced(self):
        if self.shape[0] == 0:
            self._load_files()
        return self[self["time"] == self["time"].values.astype(int)]

    @property
    def binaries(self, t=0):
        if self.binaries_mask is None:
            self.load_cluster(t)
        return self.cluster_data[self.binaries_mask]

    @property
    def singles(self, t=0):
        if self.singles_mask is None:
            self.load_cluster(t)
        return self.cluster_data[self.singles_mask]

    @property
    def time_evolution(self):
        if self.time_evolution_data is None:
            self.calc_time_evolution_data()
        return self.time_evolution_data

    def calculate_time_evolution(self, RLAGRS=None, stepsize=1):
        if RLAGRS is None:
            RLAGRS = [0.001,
                      0.003,
                      0.005,
                      0.01,
                      0.03,
                      0.05,
                      0.1,
                      0.2,
                      0.3,
                      0.4,
                      0.5,
                      0.6,
                      0.7,
                      0.8,
                      0.9,
                      0.95,
                      0.99,
                      1.0]
        self.time_evolution_data = {
                "RLAGR_BH": nbdf(),
                "E": nbdf(),
                }
        for idx in tqdm(self.index[::stepsize]):
            try:
                self.load_cluster(idx)
            except:
                continue
            self.calc_R()
            self.calc_M_over_MT()
            self.binaries_data.calc_Eb()
            for rlagr in RLAGRS:
                self.time_evolution_data["RLAGR_BH"].loc[idx,str(rlagr)] = float(self.filter("SINGLE_BH")[self.filter("SINGLE_BH")["M/MT"] < rlagr]["R"].max())
                self.time_evolution_data["E"].loc[idx,"BH-BH_N"] = self.binaries_data.filter("BH-BH").shape[0]
                self.time_evolution_data["E"].loc[idx,"BH-BH_Eb_tot"] = self.binaries_data.filter("BH-BH")["Eb"].sum()
                self.time_evolution_data["E"].loc[idx,"BH-BH_Eb_mean"] = self.binaries_data.filter("BH-BH")["Eb"].mean()
                self.time_evolution_data["E"].loc[idx,"BH-BH_Eb_std"] = self.binaries_data.filter("BH-BH")["Eb"].std()


    def load_cluster(self, time):
        if self.shape == (0,3):
            self._load_files()

        self.time = time
        
        default_cols = {
                "M": "023 M",
                "X1": "001 X1",
                "X2": "002 X2",
                "X3": "003 X3",
                "V1": "004 V1",
                "V2": "005 V2",
                "V3": "006 V3",                
                "A1": "007 A1",
                "A2": "008 A2",
                "A3": "009 A3",                
                "POT": "025 POT",                
                "K*": "031 KW",
                "NAME": "032 Name",
                "Type": "033 Type",

                }

        f = h5py.File(self.loc[time]["file"],"r")
        self.cluster_data = nbdf(columns=[key for key in default_cols.keys() if default_cols[key] in f["Step#" + self.loc[time]["step"]].keys()])
        for col in default_cols.keys():
            if default_cols[col] in f["Step#" + self.loc[time]["step"]].keys():
                self.cluster_data[col] = f["Step#" + self.loc[time]["step"]][default_cols[col]][:]
       
        binary_cols = {
                "M1": "123 Bin M1*",
                "M2": "124 Bin M2*",
                "cmX1": "101 Bin cm X1",
                "cmX2": "102 Bin cm X2",
                "cmX3": "103 Bin cm X3",
                "cmV1": "104 Bin cm V1",
                "cmV2": "105 Bin cm V2",
                "cmV3": "106 Bin cm V3",
                "relX1": "125 Bin rel X1",
                "relX2": "126 Bin rel X2",
                "relX3": "127 Bin rel X3",
                "relV1": "128 Bin rel V1",
                "relV2": "129 Bin rel V2",
                "relV3": "130 Bin rel V3",
                "K*1": "158 Bin KW1",
                "K*2": "159 Bin KW2",
                "NAME1": "161 Bin Name1",
                "NAME2": "162 Bin Name2", 
                }
        self.binaries_data =  Binaries(columns=[key for key in binary_cols.keys() if binary_cols[key] in f["Step#" + self.loc[time]["step"]].keys()])
        for col in binary_cols.keys():
            if binary_cols[col] in f["Step#" + self.loc[time]["step"]].keys():
                self.binaries_data[col] = f["Step#" + self.loc[time]["step"]][binary_cols[col]][:]

        #self.singles_data = singles(self.cluster_data, self.binary_data)
        self.singles_mask = ~self.cluster_data["NAME"].isin(self.binaries_data["NAME1"]) & ~self.cluster_data["NAME"].isin(self.binaries_data["NAME2"])
        self.binaries_mask = ~ self.singles_mask

        return self.cluster_data

    def calc_spherical_coords(self):
        """
        calculates spherical coordinates from cartesian ones.
        See https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates
        """

        self.cluster_data["R"] = np.sqrt(self.cluster_data["X1"]**2 + self.cluster_data["X2"]**2 + self.cluster_data["X3"]**2)
        self.cluster_data["THETA"] = np.arccos(self.cluster_data["X3"]/self.cluster_data["R"])
        
        mask = self.cluster_data["X1"] > 0
        self.cluster_data.loc[mask,"PHI"] = np.arctan(self.cluster_data.loc[mask,"X2"]/self.cluster_data.loc[mask,"X1"])
        mask = (self.cluster_data["X1"] < 0) & (self.cluster_data["X2"] >= 0)
        self.cluster_data.loc[mask,"PHI"] = np.arctan(self.cluster_data.loc[mask,"X2"]/self.cluster_data.loc[mask,"X1"]) + np.pi
        mask = (self.cluster_data["X1"] < 0) & (self.cluster_data["X2"] < 0)
        self.cluster_data.loc[mask,"PHI"] = np.arctan(self.cluster_data.loc[mask,"X2"]/self.cluster_data.loc[mask,"X1"]) - np.pi
        mask = (self.cluster_data["X1"] == 0) & (self.cluster_data["X2"] > 0)
        self.cluster_data.loc[mask,"PHI"] = np.pi/2
        mask = (self.cluster_data["X1"] == 0) & (self.cluster_data["X2"] < 0)
        self.cluster_data.loc[mask,"PHI"] = -np.pi/2
        self.cluster_data["R/Rt"] = self.cluster_data["R"]/self.cluster_data["R"].max()

        self.cluster_data.sort_values("R",ignore_index=True,inplace=True)

    def calc_R(self):
        return self.calc_spherical_coords()
    def calc_THETA(self):
        return self.calc_spherical_coords()
    def calc_PHI(self):
        return self.calc_spherical_coords()
    def calc_EKIN(self):
        self.cluster_data["EKIN"] = 0.5*self.cluster_data["M"]*np.linalg.norm(self.cluster_data[["V1", "V2", "V3"]], axis=1)**2

    def calc_M_over_MT(self):
        if "R" not in self.cluster_data.columns:
            self.calc_R()

        self.cluster_data["M/MT"] = self.cluster_data["M"].cumsum()/self.cluster_data["M"].sum()

    def calc_VROT(self):
        if "R" not in self.cluster_data.columns:
            self.calc_R()

        """rvxy = self.cluster_data["X1"]*self.cluster_data["V1"] + self.cluster_data["X2"]*self.cluster_data["V2"]
        rxy2 = self.cluster_data["X1"] **2 + self.cluster_data["X2"]**2
        vrot1 = self.cluster_data["V1"] - rvxy* self.cluster_data["X1"]/rxy2
        vrot2 = self.cluster_data["V2"] - rvxy* self.cluster_data["X2"]/rxy2
        xsign = np.sign(vrot1*self.cluster_data["X2"]-vrot2*self.cluster_data["X1"])
        self.cluster_data["VROT"] = xsign*np.sqrt(vrot1**2+vrot2**2)"""

        """RR12 = self.cluster_data["X1"]**2 + self.cluster_data["X2"]**2
        XR12 = (self.cluster_data["V1"] * (self.cluster_data["X1"] - self.RDENS[0])) + \
                (self.cluster_data["V2"] * (self.cluster_data["X2"] -  self.RDENS[1]))

        VROT1 = self.cluster_data["V1"] - XR12/RR12 * self.cluster_data["X1"]
        VROT2 = self.cluster_data["V2"] - XR12/RR12 * self.cluster_data["X2"]

        VROTM = np.sqrt(VROT1**2 + VROT2**2)
        XSIGN = np.sign(VROT1*self.cluster_data["X2"]/np.sqrt(RR12) - VROT2*self.cluster_data["X1"]/np.sqrt(RR12))

        self.cluster_data["VROT"] = XSIGN*self.cluster_data["M"]*VROTM"""
        VROT = np.cross( 
                    np.cross(
                        self.cluster_data.loc[:,["X1","X2","X3"]],
                        self.cluster_data.loc[:,["V1","V2","V3"]]
                    )/(self.cluster_data["R"]**2).values.reshape(self.cluster_data.shape[0],1),
                    self.cluster_data.loc[:,["X1","X2","X3"]]
                )
        XSIGN = np.sign(VROT[:,0]*self.cluster_data["X2"]/np.sqrt(self.cluster_data["X1"]**2 + self.cluster_data["X2"]**2) \
                            - VROT[:,1]*self.cluster_data["X1"]/np.sqrt(self.cluster_data["X1"]**2 + self.cluster_data["X2"]**2))
        self.cluster_data["VROT"] = XSIGN * np.sqrt(VROT[:,0]**2 + VROT[:,1]**2)


        """self.cluster_data["VROT"] = np.cross(
                self.cluster_data.loc[:,["X1","X2","X3"]],
                self.cluster_data.loc[:,["V1","V2","V3"]]
            )[:,2]/(self.cluster_data["R"]**2)"""

        
        """
        self.cluster_data["VROT"] = np.linalg.norm(np.cross(
                self.cluster_data.loc[:,["X1","X2","X3"]],
                self.cluster_data.loc[:,["V1","V2","V3"]]
            ), axis=1)/(self.cluster_data["R"]**2)
        """

    def calc_VROT_CUMMEAN(self):
        if not "VROT" in self.cluster_data.columns:
            self.calc_VROT()

        self.cluster_data["VROT_CUMMEAN"] = ffi.cummean(self.cluster_data["VROT"].values)
        #self.cluster_data["VROT_CUMMEAN"] = self.cluster_data["VROT"]

    def filter(self, value):
        if value == "BH":
            return self.cluster_data[self.singles["K*"] == 14]
        elif value == "SINGLE_BH":
            return self.cluster_data[(self.singles["K*"] == 14) & self.singles_mask]
        elif value == "BH-Any":
            return self.binaries_data[(self.binaries_data["K*1"] == 14) | (self.binaries_data["K*2"] == 14)]
        elif value == "BH-BH":
            return self.binaries_data[(self.binaries_data["K*1"] == 14) & (self.binaries_data["K*2"] == 14)]
        else:
            raise KeyError(f"Couldn't filter by value \"{value}\"")


    def _load_files(self):
        if self.files is None:
            logging.error("Couldn't find any snap files to load")
            return 0
        for file in self.files:
            # if else clause should do the same, but kepler complains about setting new row with loc
            # newer python versions complain about appending... so either way...
            # TODO: change python ver to pandas ver
            if sys.version_info.minor >= 10:
                self.loc[float(file[file.rfind("/")+1:].replace("snap.40_","").replace(".h5part",""))] = [float(file[file.rfind("/") + 1:].replace("snap.40_","").replace(".h5part","")),file, "0"]
            else:
                super().__init__(self.append({"time": float(file[file.rfind("/")+1:].replace("snap.40_","").replace(".h5part","")),
                                              "file": file,
                                              "step": "0"}, ignore_index=True))
                self.index = self["time"].values
        self.sort_index(inplace=True)

    def _analyze_files(self):
        if self.files is None:
            logging.error("Couldn't find any snap files to load")
            return 0

        super().__init__(columns=["time","file","step"])

        for file in tqdm(self.files):
            f = h5py.File(file,"r")

            for step in f.keys():

                # if else clause should do the same, but kepler complains about setting new row with loc
                # newer python versions complain about appending... so either way...
                # TODO: change python ver to pandas ver
                if sys.version_info.minor >= 10:
                    self.loc[f[step]['000 Scalars'][0]] = [f[step]['000 Scalars'][0],file, step.replace("Step#","")]
                else:
                    super().__init__(self.append({"time": f[step]['000 Scalars'][0],
                                                  "file": file,
                                                  "step": step.replace("Step#","")}, ignore_index=True))
                    self.index = self["time"].values
            f.close() 
